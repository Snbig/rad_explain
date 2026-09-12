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
import requests
import config

logger = logging.getLogger(__name__)

_api_key = None
_endpoint_url = None
_initialized = False

def init_llm_client():
    """Initializes LLM client configuration by loading from config."""
    global _api_key, _endpoint_url, _initialized
    if config.HF_TOKEN and config.MEDGEMMA_ENDPOINT_URL:
        _api_key = config.HF_TOKEN
        _endpoint_url = config.MEDGEMMA_ENDPOINT_URL
        _initialized = True
        logger.info("LLM client configured successfully.")
    else:
        _api_key = None
        _endpoint_url = None
        logger.error("LLM client could not be configured due to missing API key or endpoint URL.")

def is_initialized():
    return _initialized

def make_chat_completion_request(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    stream: bool,
    top_p: float | None = None,
    seed: int | None = None,
    stop: list[str] | str | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None
):
    """
    Makes a chat completion request to the configured LLM API.
    """
    if not _initialized:
        logger.error("LLM client not initialized.")
        raise RuntimeError("LLM client not initialized.")

    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if top_p is not None: payload["top_p"] = top_p
    if seed is not None: payload["seed"] = seed
    if stop is not None: payload["stop"] = stop
    if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None: payload["presence_penalty"] = presence_penalty

    temp_url = _endpoint_url.rstrip('/')
    if temp_url.endswith("/v1/chat/completions"):
        full_url = temp_url
    elif temp_url.endswith("/v1"):
        full_url = temp_url + "/chat/completions"
    else:
        full_url = temp_url + "/v1/chat/completions"

    # Generous read timeout: the local Colab serve_medgemma server computes
    # the full answer before streaming, and a T4 generating multi-slice image
    # answers can take well over 60 s before the first bytes arrive.
    response = requests.post(full_url, headers=headers, json=payload, stream=stream, timeout=(10, 900))
    response.raise_for_status()
    return response