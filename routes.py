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

import os
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

        explanation = "".join(explanation_parts).strip()
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
        max_slices = min(max(int(request.form.get('max_slices') or 8), 1), 16)
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

        plain_image_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
        if all(p.suffix.lower() in plain_image_exts for p in saved_paths):
            previews = imaging.process_plain_images(saved_paths)
            result = {"modality": modality_override or "X-ray",
                      "total_slices": 1,
                      "prompt_slices": len(previews),
                      "previews": previews}
        else:
            result = imaging.process_upload_dicom(saved_paths, max_slices=max_slices)
            if modality_override:
                result["modality"] = modality_override

        modality = result["modality"]
        total_slices = result["total_slices"]
        previews = result["previews"]

        system_prompt = (
            "You are an expert radiologist explaining medical images to a "
            "non-specialist in simple, clear language. Be concise, describe what "
            "is visible (or not), and clearly flag any uncertainty. This is for "
            "educational purposes only and is not a diagnosis."
        )

        if total_slices > 1:
            instruction = (
                f"You are reviewing a {modality} series with {total_slices} "
                "slices. The evenly-sampled slices below represent the volume. "
                "Review them as a radiologist would a full study"
            )
            content = [{"type": "text", "text": instruction}]
            for prev in previews:
                content.append({"type": "image", "image": prev["data_url"]})
                content.append({"type": "text", "text": f"SLICE {prev['label']}"})
        else:
            instruction = (
                f"You are reviewing a {modality} image."
            )
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

        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

        logger.info("Sending uploaded image request to LLM API (REST)...")
        response = make_chat_completion_request(
            model="tgi",
            messages=messages,
            top_p=None,
            temperature=0,
            max_tokens=600,
            stream=True,
            seed=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None
        )

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

        explanation = "".join(explanation_parts).strip()
        if not explanation:
            logger.warning("Empty explanation from API for uploaded image.")
        return jsonify({
            "modality": modality,
            "total_slices": total_slices,
            "prompt_slices": len(previews),
            "previews": previews,
            "explanation": explanation or
                           "No explanation content received from the API."
        })
    except ValueError as e:
        logger.warning(f"Invalid upload: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400
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
