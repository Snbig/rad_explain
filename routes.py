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

import logging
from flask import Blueprint, render_template, request, jsonify, send_from_directory
from pathlib import Path
import shutil # For zipping the cache directory
import json # For parsing streamed JSON data
import re

import os
import random
import threading
import uuid
import tempfile
import config
import utils
import imaging
from llm_client import make_chat_completion_request, is_initialized as llm_is_initialized
from cache_store import cache
from cache_store import cache_directory
import requests

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

# LLM client is initialized in app.py create_app()


class LLMServiceError(Exception):
    """The MedGemma call itself failed; carries a client-safe message."""

    def __init__(self, message, status=503):
        super().__init__(message)
        self.message = message
        self.status = status


def _build_messages(result, question=""):
    """Build the multimodal MedGemma message list from processed previews."""
    modality = result["modality"]
    total_slices = result["total_slices"]
    previews = result["previews"]

    system_prompt = (
        "You are an expert radiologist explaining medical images to a "
        "non-specialist in plain language. Be thorough and detailed; do NOT "
        "give a one-line summary. Structure your answer with AT LEAST these "
        "three sections, using headings on their own lines wrapped in **bold**, "
        "like this:\n\n"
        "**Findings**\n- list each finding you observe (or state clearly that "
        "the study looks normal), with simple explanations\n\n"
        "**Impression**\n- one short paragraph giving the overall interpretation "
        "and whether the study appears normal or abnormal\n\n"
        "**Recommendations**\n- any suggested next steps, e.g. clinical "
        "correlation, follow-up imaging, or comparison with prior studies\n\n"
        "Use bullet points and short paragraphs; write enough detail to be "
        "useful, and clearly flag any uncertainty. This is for educational "
        "purposes only and is not a diagnosis."
    )

    if total_slices > 1:
        instruction = (
            f"You are reviewing a {modality} series with {total_slices} "
            "slices. The slices below were selected automatically (attention-"
            "based) and represent the key content of the volume. "
            "Review them as a radiologist would a full study"
        )
        content = [{"type": "text", "text": instruction}]
        for prev in previews:
            content.append({"type": "image", "image": prev["data_url"]})
            content.append({"type": "text", "text": f"SLICE {prev['label']}"})
        refs = result.get("slice_references")
        if refs:
            content.append({"type": "text", "text": (
                "Reference: the images above are slices of one DICOM series "
                "converted to NIfTI with the calibration removed, blank/empty "
                "slices deleted, and the key slices selected automatically. "
                "The original DICOM instance (slice) numbers shown are: "
                + ", ".join(str(r) for r in refs) + ".")})
    else:
        instruction = f"You are reviewing a {modality} image."
        content = [{"type": "text", "text": instruction},
                   {"type": "image", "image": previews[0]["data_url"]}]

    if question:
        content.append({"type": "text",
                        "text": f"Question from the user: {question}"})
    else:
        content.append({"type": "text",
                        "text": ("Describe the most important findings in "
                                 "simple terms and state whether the study "
                                 "appears normal or abnormal.")})

    return [
        {"role": "system",
         "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def _dedupe_repeated_sentences(text):
    """Drop repeated sentences (model loop artifact).

    MedGemma occasionally loops and emits the same sentence periodically.
    Splits on newlines and sentence boundaries, keeps the original separators
    so markdown structure survives, and removes a sentence whose normalized
    text was already emitted. Short fragments/headers (no sentence terminator
    or <=8 chars normalized) are never treated as duplicates.
    """
    if not text:
        return text
    tokens = re.split(r"(\n|(?<=[.!?\u2026])\s+)", text.strip())
    seen = set()
    out = []
    for i in range(0, len(tokens), 2):
        part = tokens[i]
        sep = tokens[i + 1] if i + 1 < len(tokens) else ""
        if re.search(r"[.!?\u2026]", part):
            key = re.sub(r"[^a-z0-9]", "", part.lower())
            if len(key) > 8 and key in seen:
                continue
            seen.add(key)
        out.append(part + sep)
    return "".join(out).strip()


def _stream_explanation(messages, max_tokens=1536):
    """Send the multimodal messages and collect the streamed answer.

    Turns endpoint failures into LLMServiceError with the real reason (HTTP
    status / error body / OOM / connection problem) so callers can surface a
    useful message instead of a generic "service is starting up".
    """
    logger.info("Sending (uploaded / IDC sample) request to LLM API (REST)...")
    try:
        response = make_chat_completion_request(
            model="tgi",
            messages=messages,
            top_p=None,
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            seed=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
        )
    except requests.exceptions.HTTPError as e:
        detail = str(e)
        status = e.response.status_code if e.response is not None else 502
        if e.response is not None:
            try:
                detail = (e.response.json().get("error", {}).get("message")
                          or detail)
            except Exception:  # noqa: BLE001
                detail = (e.response.text[:300] if e.response.text else detail)
        raise LLMServiceError(
            f"LLM endpoint returned HTTP {status}: {detail}", status=status)
    except requests.exceptions.RequestException as e:
        raise LLMServiceError(
            f"LLM endpoint is unreachable: {e}", status=503)

    explanation_parts = []
    for line in response.iter_lines():
        if not line:
            continue
        decoded_line = line.decode('utf-8')
        if decoded_line.startswith('data: '):
            json_data_str = decoded_line[len('data: '):].strip()
            if json_data_str == "[DONE]":
                break
            try:
                chunk = json.loads(json_data_str)
                if (chunk.get("choices") and
                        chunk["choices"][0].get("delta") and
                        chunk["choices"][0]["delta"].get("content")):
                    explanation_parts.append(
                        chunk["choices"][0]["delta"]["content"])
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode JSON from stream chunk: {json_data_str}")
        elif decoded_line.strip() == "[DONE]":
            break
    return "".join(explanation_parts).strip()


# --- Pre-fetched IDC sample pool --------------------------------------------
# The "Fetch samples from IDC" button stages a handful of series here; the
# X-ray/CT/MRI buttons then analyze a random sample from this pool instead of
# downloading on every click.
_IDC_POOL = {}
_IDC_POOL_LOCK = threading.Lock()
_IDC_POOL_ROOT = Path(tempfile.gettempdir()) / "radexplain-idc-pool"
# How many samples to stage per modality when fetching the pool.
IDC_POOL_COUNT = {"X-ray": 5, "CT": 5, "MRI": 5}


def _idc_pool_summary():
    with _IDC_POOL_LOCK:
        return {m: len(_IDC_POOL.get(m, [])) for m in ("X-ray", "CT", "MRI")}


def _pick_pooled_sample(modality):
    """Return a random staged sample for the modality, or None."""
    with _IDC_POOL_LOCK:
        group = _IDC_POOL.get(modality) or []
        if not group:
            return None
        return random.choice(group)


def _clean_pool_dirs(keep_dirs):
    """Remove series staging dirs that are no longer in the pool."""
    _IDC_POOL_ROOT.mkdir(parents=True, exist_ok=True)
    for child in _IDC_POOL_ROOT.iterdir():
        if child.is_dir() and str(child) not in keep_dirs:
            shutil.rmtree(child, ignore_errors=True)


@main_bp.route('/idc_fetch_samples', methods=['POST'])
def idc_fetch_samples():
    """Stage ~15 public IDC samples (5 per modality) for instant analysis."""
    collected = {}
    for modality, count in IDC_POOL_COUNT.items():
        try:
            samples = imaging.fetch_idc_sample_pool(
                modality, count, _IDC_POOL_ROOT)
        except (ValueError, requests.exceptions.RequestException) as e:
            logger.warning("Could not stage %s samples: %s", modality, e)
            samples = []
        collected[modality] = samples

    with _IDC_POOL_LOCK:
        _IDC_POOL.clear()
        for modality, samples in collected.items():
            _IDC_POOL[modality] = []
            for sample in samples:
                sample["id"] = uuid.uuid4().hex[:8]
                _IDC_POOL[modality].append(sample)
        keep_dirs = {s["dir"]
                     for group in _IDC_POOL.values() for s in group}
    _clean_pool_dirs(keep_dirs)

    by_modality = {m: len(collected[m]) for m in collected}
    staged = sum(by_modality.values())
    if staged == 0:
        return jsonify({"error": "No IDC samples could be staged."}), 502
    logger.info("Staged %d IDC samples: %s", staged, by_modality)
    return jsonify({"samples": staged, "by_modality": by_modality})

# --- Serve the cache directory as a zip file ---
@main_bp.route('/download_cache')
def download_cache_zip():
    """Zips the cache directory and serves it for download."""
    zip_filename = "radexplain-cache.zip"
    # Create the zip file in a temporary directory
    # Using /tmp is common in containerized environments
    temp_dir = "/tmp"
    zip_base_path = os.path.join(temp_dir, "radexplain-cache") # shutil adds .zip
    zip_filepath = zip_base_path + ".zip"

    # Ensure the cache directory exists before trying to zip it
    if not os.path.isdir(cache_directory):
        logger.error(f"Cache directory not found at {cache_directory}")
        return jsonify({"error": f"Cache directory not found on server: {cache_directory}"}), 500

    try:
        logger.info(f"Creating zip archive of cache directory: {cache_directory} to {zip_filepath}")
        shutil.make_archive(
            zip_base_path, # This is the base name, shutil adds the .zip extension
            "zip",
            cache_directory, # This is the root directory to archive
        )
        logger.info("Zip archive created successfully.")
        # Send the file and then clean it up
        return send_from_directory(temp_dir, zip_filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error creating or sending zip archive of cache directory: {e}", exc_info=True)
        return jsonify({"error": f"Error creating or sending zip archive: {e}"}), 500
@main_bp.route('/')
def index():
    """Serves the main HTML page."""
    # The backend now only provides the list of available reports.
    # The frontend will be responsible for selecting a report,
    # fetching its details (text, image path), and managing the current state.
    if not config.AVAILABLE_REPORTS:
        logger.warning("No reports found in config. AVAILABLE_REPORTS is empty.")

    return render_template(
        'index.html',
        available_reports=config.AVAILABLE_REPORTS
    )

@main_bp.route('/get_report_details/<report_name>')
def get_report_details(report_name):
    """Fetches the text content and image path for a given report name."""
    selected_report_info = next((item for item in config.AVAILABLE_REPORTS if item['name'] == report_name), None)

    if not selected_report_info:
        logger.error(f"Report '{report_name}' not found when fetching details.")
        return jsonify({"error": f"Report '{report_name}' not found."}), 404

    report_file = selected_report_info.get('report_file')
    image_file = selected_report_info.get('image_file') 

    report_text_content = "" # Default to empty if no report file is configured.

    if report_file:
        actual_server_report_path = config.BASE_DIR / report_file

        try:
            report_text_content = actual_server_report_path.read_text(encoding='utf-8').strip()
        except Exception as e:
            logger.error(f"Error reading report file {actual_server_report_path} for report '{report_name}': {e}", exc_info=True)
            return jsonify({"error": "Error reading report file."}), 500
    # If report_file was empty, report_text_content remains "".

    image_type_from_config = selected_report_info.get('image_type')
    display_image_type = 'Chest X-Ray' if image_type_from_config == 'CXR' else ('CT' if image_type_from_config == 'CT' else 'Medical Image')

    return jsonify({"text": report_text_content, "image_file": image_file, "image_type": display_image_type})



@main_bp.route('/explain', methods=['POST'])
def explain_sentence():
    """Handles the explanation request using LLM API with base64 encoded image."""
    if not llm_is_initialized():
         logger.error("LLM client (REST API) not initialized. Cannot process request.")
         return jsonify({"error": "LLM client (REST API) not initialized. Check API key and base URL."}), 500

    data = request.get_json()
    if not data or 'sentence' not in data or 'report_name' not in data:
        logger.warning("Missing 'sentence' or 'report_name' in request payload.")
        return jsonify({"error": "Missing 'sentence' or 'report_name' in request"}), 400

    selected_sentence = data['sentence']
    report_name = data['report_name']
    logger.info(f"Received request to explain: '{selected_sentence}' for report: '{report_name}'")

    # --- Find the selected report info ---
    selected_report_info = next((item for item in config.AVAILABLE_REPORTS if item['name'] == report_name), None)

    if not selected_report_info:
        logger.error(f"Report '{report_name}' not found in available reports.")
        return jsonify({"error": f"Report '{report_name}' not found."}), 404

    image_file = selected_report_info.get('image_file')
    report_file = selected_report_info.get('report_file')
    image_type = selected_report_info.get('image_type')

    if not image_file:
        logger.error(f"Image or report file path (relative to static) missing in config for report '{report_name}'.")
        return jsonify({"error": f"File configuration missing for report '{report_name}'."}), 500

    # Construct absolute server paths using BASE_DIR as image_file and report_file include "static/"
    server_image_path = config.BASE_DIR / image_file

    
    # --- Prepare Base64 Image for API ---
    if not server_image_path.is_file():
        logger.error(f"Image file not found at {server_image_path}")
        return jsonify({"error": f"Image file for report '{report_name}' not found on server."}), 500

    base64_image_data_url = utils.image_to_base64_data_url(str(server_image_path))
    if not base64_image_data_url:
        logger.error("Failed to encode image to base64.")
        return jsonify({"error": "Could not encode image for API request"}), 500

    logger.info("Image successfully encoded to base64 data URL for API.")

    full_report_text = ""
    if report_file: # Only attempt to read if a report file is configured
        server_report_path = config.BASE_DIR / report_file
        try:
            full_report_text = server_report_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            logger.error(f"Report file not found at {server_report_path}")
            return jsonify({"error": f"Report file for '{report_name}' not found on server."}), 500
        except Exception as e:
            logger.error(f"Error reading report file {server_report_path}: {e}", exc_info=True)
            return jsonify({"error": "Error reading report file."}), 500
    else: # If report_file is not configured (e.g. empty string from selected_report_info)
        logger.info(f"No report file configured for report '{report_name}'. Proceeding without full report text for system prompt.")

    system_prompt = (
        "You are a public-facing clinician. "
        f"A learning user has provided a sentence from a radiology report and is viewing the accompanying {image_type} image. "
        "Your task is to explain the meaning of ONLY the provided sentence in simple, clear terms. Explain terminology and abbriviations. Keep it concise. "
        "Directly address the meaning of the sentence. Do not use introductory phrases like 'Okay' or refer to the sentence itself or the report itself (e.g., 'This sentence means...'). " # noqa: E501
        f"{f'Crucially, since the user is looking at their {image_type} image, provide guidance on where to look on the image to understand your explanation, if applicable. ' if image_type != 'CT' else ''}"
        "Do not discuss any other part of the report or any sentences not explicitly provided by the user. Stick to facts in the text. Do not infer anything. \n"
        "===\n"
        f"For context, the full REPORT is:\n{full_report_text}"
    )
    user_prompt_text = f"Explain this sentence from the radiology report: '{selected_sentence}'"

    messages_for_api = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_text}
            ]
        }
    ]

    cache_key = f"explain::{report_name}::{selected_sentence}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info("Returning cached explanation.")
        return jsonify({"explanation": cached_result})

    try:
        logger.info("Sending request to LLM API (REST) with base64 image...")
        response = make_chat_completion_request(
            model="tgi",
            messages=messages_for_api,
            top_p=None,
            temperature=0,
            max_tokens=250,
            stream=True,
            seed=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None
        )
        logger.info("Received response stream from LLM API (REST).")

        explanation_parts = []
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_data_str = decoded_line[len('data: '):].strip()
                    if json_data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_data_str)
                        if chunk.get("choices") and chunk["choices"][0].get("delta") and chunk["choices"][0]["delta"].get("content"):
                            explanation_parts.append(chunk["choices"][0]["delta"]["content"])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from stream chunk: {json_data_str}")
                        # Depending on API, might need to handle partial JSON or other errors
                elif decoded_line.strip() == "[DONE]": # Some APIs might send [DONE] without "data: "
                    break

        explanation = _dedupe_repeated_sentences(
            "".join(explanation_parts).strip())
        if explanation:
            cache.set(cache_key, explanation, expire=None)

        logger.info("Explanation generated successfully." if explanation else "Empty explanation from API.")
        return jsonify({"explanation": explanation or "No explanation content received from the API."})
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during LLM API (REST) call: {e}", exc_info=True)
        user_error_message = ("Failed to generate explanation. The service might be temporarily unavailable "
                              "and is now likely starting up. Please try again in a few moments.")
        return jsonify({"error": user_error_message}), 500


@main_bp.route('/upload_explain', methods=['POST'])
def upload_explain():
    """Explains an uploaded X-ray / CT / MRI file or series via the LLM API.

    Accepts single 2D images (PNG/JPEG) as well as DICOM files (.dcm, one or
    several, or a .zip archive containing them). Volumetric series (CT / MRI)
    are windowed and down-sampled to a small stack of slice images before
    being sent to the multimodal MedGemma endpoint.
    """
    import shutil
    import tempfile
    import uuid

    if not llm_is_initialized():
        logger.error("LLM client (REST API) not initialized. Cannot process upload.")
        return jsonify({"error": "LLM client (REST API) not initialized. Check API key and base URL."}), 500

    uploaded_files = request.files.getlist('files')
    uploaded_files = [f for f in uploaded_files if f.filename]
    if not uploaded_files:
        return jsonify({"error": "No file uploaded."}), 400

    question = (request.form.get('question') or '').strip()
    modality_override = (request.form.get('modality') or '').strip()
    if modality_override not in ('CT', 'MRI', 'X-ray'):
        modality_override = None

    max_slices = 8
    try:
        max_slices = min(max(int(request.form.get('max_slices') or 8), 1), 30)
    except ValueError:
        max_slices = 8

    tmp_root = Path(tempfile.mkdtemp(prefix="radexplain-upload-"))
    try:
        saved_paths = []
        for f in uploaded_files:
            ext = Path(f.filename).suffix.lower()
            if ext not in {'.dcm', '.dicom', '.zip', '.png', '.jpg', '.jpeg',
                           '.webp', '.bmp', '.gif'}:
                return jsonify({
                    "error": f"Unsupported file type '{ext or '(none)'}'. "
                             "Use PNG/JPEG images or DICOM (.dcm) files."}), 400
            tmp_path = tmp_root / (uuid.uuid4().hex + (ext or '.bin'))
            f.stream.seek(0)
            f.save(tmp_path)
            saved_paths.append(tmp_path)

        # Auto-detect reads DICOM tags; for JPG/PNG-only stacks the tag is
        # missing so the default label is X-ray and the series count comes
        # from the number of images uploaded.
        result = imaging.process_upload(
            saved_paths, max_slices=max_slices, modality_override=modality_override,
            dest_dir=tmp_root)

        modality = result["modality"]
        total_slices = result["total_slices"]
        previews = result["previews"]
        warnings = list(result.get("skipped_files") or [])
        if not modality_override and total_slices > 1 \
                and all(p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp',
                                             '.bmp', '.gif'}
                        for p in saved_paths):
            warnings.append(
                "Auto-detect only reads DICOM tags. Multiple JPG/PNG images are "
                "assumed to be one study labeled X-ray — set Type above if this "
                "is a CT/MRI series.")

        messages = _build_messages(result, question)
        explanation = _stream_explanation(messages, max_tokens=1536)
        if not explanation:
            logger.warning("Empty explanation from API for uploaded image.")
        return jsonify({
            "modality": modality,
            "total_slices": total_slices,
            "prompt_slices": len(previews),
            "previews": previews,
            "warnings": warnings,
            "explanation": explanation or
                           "No explanation content received from the API."
        })
    except ValueError as e:
        logger.warning(f"Invalid upload: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400
    except LLMServiceError as e:
        logger.error(f"LLM service error for uploaded image: {e.message}")
        return jsonify({"error": e.message}), e.status
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during LLM API call for uploaded image: {e}", exc_info=True)
        return jsonify({"error": ("Failed to generate explanation. The service "
                                  "might be temporarily unavailable and is now "
                                  "likely starting up. Please try again in a few "
                                  "moments.")}), 500
    except Exception as e:  # noqa: BLE001
        logger.error(f"Unexpected error handling upload: {e}", exc_info=True)
        return jsonify({"error": f"Unexpected error processing upload: {e}"}), 500
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@main_bp.route('/idc_explain', methods=['POST'])
def idc_explain():
    """Explains a public X-ray / CT / MRI cancer sample from IDC.

    Prefers a random sample from the pre-fetched pool (see
    /idc_fetch_samples); if none is staged for the requested modality, one
    small public cancer series is fetched on the fly. Either way it runs
    through the DICOM -> slice previews -> MedGemma pipeline.
    """
    import shutil

    if not llm_is_initialized():
        logger.error("LLM client (REST API) not initialized. Cannot process IDC sample.")
        return jsonify({"error": "LLM client (REST API) not initialized. Check API key and base URL."}), 500

    data = request.get_json(silent=True) or {}
    modality = (data.get('modality') or '').strip()
    if modality not in ('X-ray', 'CT', 'MRI'):
        return jsonify({"error": "modality must be one of: X-ray, CT, MRI."}), 400

    question = (data.get('question') or '').strip()
    sample = _pick_pooled_sample(modality)

    max_slices = 8
    try:
        max_slices = min(max(int(data.get('max_slices') or 8), 1), 30)
    except ValueError:
        max_slices = 8

    tmp_root = Path(tempfile.mkdtemp(prefix="radexplain-idc-"))
    try:
        try:
            if sample is not None:
                logger.info("Using pooled IDC %s sample %s",
                            modality, sample["id"])
                info = sample["info"]
                files = sample["files"]
            else:
                info, files = imaging.fetch_idc_sample(modality, dest_dir=tmp_root)
        except requests.exceptions.RequestException as e:
            logger.error(f"IDC fetch failed: {e}", exc_info=True)
            return jsonify({
                "error": f"Could not reach the NCI Imaging Data Commons: {e}"
            }), 502
        except ValueError as e:
            logger.warning(f"Invalid IDC sample request: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 400

        result = imaging.process_upload_dicom(files, max_slices=max_slices)
        result["modality"] = modality
        previews = result["previews"]
        total_slices = result["total_slices"]

        messages = _build_messages(result, question)
        explanation = _stream_explanation(messages, max_tokens=1536)
        if not explanation:
            logger.warning("Empty explanation from API for IDC sample.")

        return jsonify({
            "modality": modality,
            "total_slices": total_slices,
            "prompt_slices": len(previews),
            "previews": previews,
            "explanation": explanation or
                           "No explanation content received from the API.",
            "collection": info["collection_id"],
            "body_part": info["body_part"],
            "series": info["series_description"],
            "source": f"IDC · {info['collection_id']}"
                       f" (n={info['instances']}, {info['size_mb']:.0f} MB)",
            "pool_counts": _idc_pool_summary(),
        })
    except LLMServiceError as e:
        logger.error(f"LLM service error for IDC sample: {e.message}")
        return jsonify({"error": e.message}), e.status
    except ValueError as e:
        logger.warning(f"Invalid IDC sample request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        logger.error(f"Unexpected error handling IDC sample: {e}", exc_info=True)
        return jsonify({"error": f"Unexpected error fetching IDC sample: {e}"}), 500
    finally:
        if sample is None:
            shutil.rmtree(tmp_root, ignore_errors=True)


@main_bp.route('/explain_sentence', methods=['POST'])
def explain_response_sentence():
    """Explain one sentence/bullet of a generated explanation in plain terms.

    Mirrors the demo's report-sentence click flow, but for the AI-generated
    Findings / Impression / Recommendations output.
    """
    if not llm_is_initialized():
        logger.error("LLM client (REST API) not initialized. Cannot explain sentence.")
        return jsonify({"error": "LLM client (REST API) not initialized. Check API key and base URL."}), 500

    data = request.get_json(silent=True) or {}
    sentence = (data.get('sentence') or '').strip()
    modality = (data.get('modality') or '').strip() or 'Medical Image'
    if not sentence:
        return jsonify({"error": "Missing 'sentence' in request."}), 400

    system_prompt = (
        "You are a public-facing clinician. A learning user clicked a "
        f"sentence from an AI-generated {modality} explanation and wants to "
        "understand what that specific sentence means. "
        "Explain ONLY the meaning of the provided sentence in simple, clear "
        "terms. Explain any terminology or abbreviations. Be concise but "
        "complete. Do not invent findings not implied by the sentence. "
        "This is for educational purposes only and is not a diagnosis."
    )
    user_prompt_text = (f"Explain this sentence in plain language: "
                        f"'{sentence}'")
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",
         "content": [{"type": "text", "text": user_prompt_text}]},
    ]

    try:
        explanation = _stream_explanation(messages, max_tokens=512)
        return jsonify({
            "explanation": explanation or
                           "No explanation content received from the API."
        })
    except LLMServiceError as e:
        logger.error(f"LLM service error explaining sentence: {e.message}")
        return jsonify({"error": e.message}), e.status
    except requests.exceptions.RequestException as e:
        logger.error(f"Error explaining sentence: {e}", exc_info=True)
        return jsonify({"error": ("Failed to generate explanation. The service "
                                  "might be temporarily unavailable and is now "
                                  "likely starting up. Please try again in a "
                                  "few moments.")}), 500
