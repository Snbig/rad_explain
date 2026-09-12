# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiny local OpenAI-compatible chat-completions server for MedGemma 1.5.

Implements just enough of ``POST /v1/chat/completions`` to serve the
rad_explain Flask app (which calls ``MEDGEMMA_ENDPOINT_URL``) from inside a
single Colab instance: the app runs on :7860 and this server on :7861.

The multimodal wire format matches the HF MedGemma message style used by the
official notebooks: ``content`` items of ``{"type": "text", "text": ...}`` and
``{"type": "image", "image": "<data: URL>"}``. Volumetric CT/MRI slices and
X-rays are sent as ``image`` items with inline PNG data URLs.

Run standalone::

    pip install -r requirements.txt
    HF_TOKEN=<your-token> python serve_medgemma.py
"""
import gc
import json
import logging
import os
import threading

import torch
import transformers
from flask import Flask, Response, jsonify, request

logger = logging.getLogger("medgemma_server")

MODEL_ID = "google/medgemma-1.5-4b-it"

# Reject prompts whose estimated SigLIP image-token count exceeds this: past
# this the image prefill reliably OOMs a 16 GB T4 (see _estimate_image_tokens).
MAX_PREFILL_IMAGE_TOKENS = int(os.environ.get("MEDGEMMA_MAX_PREFILL_TOKENS", "9000"))

_processor = None
_model = None
_lock = threading.Lock()


def load_model(load_in_4bit=True):
    """Load MedGemma 1.5 once (4-bit by default so it fits a 16 GB GPU)."""
    global _processor, _model
    if _model is not None:
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if load_in_4bit:
        model_kwargs = dict(
            device_map="auto",
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        )
    else:
        model_kwargs = dict(dtype=torch.bfloat16, device_map="auto",
                            offload_buffers=True)
    _processor = transformers.AutoProcessor.from_pretrained(
        MODEL_ID, use_fast=True)
    _model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, **model_kwargs)
    torch.cuda.empty_cache()
    logger.info("MedGemma loaded (4-bit=%s). free Mem: %.1f GiB",
                load_in_4bit, torch.cuda.mem_get_info()[0] / 1024 ** 3)


def _normalize_messages(chat_messages):
    """Convert OpenAI-style messages into MedGemma ``content`` item lists."""
    out = []
    for m in chat_messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m.get("role", "user"),
                        "content": [{"type": "text", "text": content}]})
            continue
        items = []
        for item in content or []:
            item_type = item.get("type")
            if item_type == "text":
                items.append({"type": "text", "text": item.get("text", "")})
            elif item_type in ("image", "image_url"):
                if item_type == "image":
                    src = item.get("image")
                else:
                    src = item.get("image_url", {})
                    if not isinstance(src, str):
                        src = src.get("url", "")
                if not isinstance(src, str) or not src.startswith("data:"):
                    raise ValueError(
                        "Only inline data: URLs are supported for images.")
                items.append({"type": "image", "image": src})
            else:
                raise ValueError(
                    f"Unsupported message content item: {item_type}")
        out.append({"role": m.get("role", "user"), "content": items})
    return out


def _estimate_image_tokens(chat_messages):
    """Approximate SigLIP tokens the image prefill will consume (~(side/14)^2/image)."""
    total = 0
    for m in chat_messages:
        for item in (m.get("content") or []):
            if item.get("type") == "image":
                src = item.get("image") or ""
                total += 1 if "data:image/png" in src else 0
    if total == 0:
        return 0
    import base64
    import io
    try:
        img = next(
            item["image"]
            for m in chat_messages for item in (m.get("content") or [])
            if item.get("type") == "image" and "data:image/png" in item.get("image", "")
        )
        from PIL import Image
        w, h = Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[1]))).size
        per_img = ((w // 14) + 1) * ((h // 14) + 1)
    except Exception:  # noqa: BLE001 - best-effort estimate
        per_img = 950
    return total * per_img


def generate(messages, max_new_tokens=600):
    """Tokenize image/text content and generate a response (deterministic).

    Mirrors the official high-dimensional CT notebook path exactly (including
    ``continue_final_message=False``): apply_chat_template -> generate ->
    post_process_image_text_to_text, stripping any echoed prompt prefix.

    Every request frees its input/generated tensors and empties the CUDA
    cache in a ``finally`` block, so the GPU never carries memory from a
    previous sample into the next one.
    """
    import time

    with _lock:
        start = time.time()
        inputs = None
        generated_sequence = None
        try:
            with torch.inference_mode():
                inputs = _processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    return_tensors="pt",
                    tokenize=True,
                    return_dict=True,
                )
                _est = _estimate_image_tokens(messages)
                logger.info("prefill: %d slice images (~%d image tokens)",
                            sum(1 for m in messages
                                for it in (m.get("content") or [])
                                if it.get("type") == "image"), _est)
                inputs = inputs.to(_model.device, dtype=torch.bfloat16)
                torch.cuda.empty_cache()
                generated_sequence = _model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_new_tokens)
            response = _processor.post_process_image_text_to_text(
                generated_sequence, skip_special_tokens=True)[0]
            decoded_inputs = _processor.post_process_image_text_to_text(
                inputs["input_ids"], skip_special_tokens=True)[0]
        finally:
            # Release this request's GPU tensors so the next sample starts
            # from a clean slate (KV caches die with the forward pass, but
            # fragmentation can linger without an explicit empty_cache).
            del inputs, generated_sequence
            gc.collect()
            torch.cuda.empty_cache()
    index = response.find(decoded_inputs)
    if 0 <= index <= 2:
        response = response[index + len(decoded_inputs):]
    logger.info("generation took %.1fs (%d output tokens)",
                time.time() - start, max_new_tokens)
    return response


app = Flask("medgemma")


@app.post("/v1/chat/completions")
def chat_completions():
    payload = request.get_json(force=True)
    try:
        messages = _normalize_messages(payload.get("messages", []))
        max_tokens = int(payload.get("max_tokens", 600))
        max_new_tokens = max(1, min(max_tokens, 1536))
        stream = bool(payload.get("stream", False))
    except (ValueError, KeyError) as e:
        return jsonify({"error": {"message": str(e)}}), 400

    # Guard against a prefill so big it is guaranteed to OOM the GPU: a clear
    # 413 beats a 503/torch OOM after minutes of vision encoding.
    est_tokens = _estimate_image_tokens(messages)
    if est_tokens > MAX_PREFILL_IMAGE_TOKENS:
        logger.warning("Rejecting prompt: ~%d image tokens > %d allowed",
                       est_tokens, MAX_PREFILL_IMAGE_TOKENS)
        return jsonify({"error": {
            "message": (f"The prompt encodes ~{est_tokens:,} image tokens "
                        "which exceeds this GPU. Lower the Slices value "
                        "(or set MEDGEMMA_IMAGE_SIDE=256) and retry.")}}), 413

    try:
        text = generate(messages, max_new_tokens=max_new_tokens)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        logger.exception("CUDA OOM during generation")
        return jsonify({"error": {
            "message": ("CUDA out of memory during generation. "
                        "Lower the Slices value (e.g. 2-4) or set "
                        "MEDGEMMA_IMAGE_SIDE=256 and restart.")}}), 503
    except Exception as e:  # noqa: BLE001
        logger.exception("Generation failed")
        return jsonify({"error": {"message": str(e)}}), 500

    if stream:
        first = {"choices": [{"index": 0, "delta": {"role": "assistant",
                                                     "content": text}}]}

        def sse():
            yield f"data: {json.dumps(first)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(sse(), mimetype="text/event-stream")

    return jsonify({
        "id": "cmpl-colab",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_ID,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text}}],
    })


def run_server(host="127.0.0.1", port=7861, load_in_4bit=True):
    """Load the model, then run the server on the given port."""
    load_model(load_in_4bit=load_in_4bit)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _port = int(os.environ.get("PORT", "7861"))
    _4bit = os.environ.get("LOAD_IN_4BIT", "1") == "1"
    run_server(port=_port, load_in_4bit=_4bit)