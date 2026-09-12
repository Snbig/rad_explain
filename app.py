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

import os
import logging
import sys
from flask import Flask

# Import configurations and initializers first
import config # Use relative import assuming app.py is in the rad_explain package
import llm_client
from routes import main_bp
from cache_store import cache

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__, static_folder=config.STATIC_DIR)

    # Cap upload size (a zip of a CT/MRI series can be large; 800 MB is ample).
    app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024

    # --- Configure Logging ---
    # Basic config should be done before creating the app or registering blueprints
    # if those modules rely on logging during import time.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    logger = logging.getLogger(__name__)

    # --- Check Configuration and Initialize Services ---
    if not config.HF_TOKEN:
        logger.error("HF_TOKEN environment variable not set.")
        sys.exit("Exiting: HF_TOKEN not set.")
    if not config.MEDGEMMA_ENDPOINT_URL:
        logger.error("MEDGEMMA_ENDPOINT_URL environment variable not set.")
        sys.exit("Exiting: MEDGEMMA_ENDPOINT_URL not set.")
    else:
        logger.info(f"Using LLM API Base URL: {config.MEDGEMMA_ENDPOINT_URL}")

    # Initialize LLM Client
    llm_client.init_llm_client()
    if not llm_client.is_initialized():
         logger.warning("LLM client failed to initialize. API calls will fail.")
         sys.exit("Exiting: LLM client initialization failed.")

    # Register Blueprints
    app.register_blueprint(main_bp)

    return app

# Create the application instance using the factory function
# This makes the 'app' instance available at the module level for WSGI servers
app = create_app()

if __name__ == '__main__':
    # This block now primarily focuses on running the app and pre-run checks/setup
    logger = logging.getLogger(__name__)

    # Run the Flask development server
    app.run(host='0.0.0.0', port=7860, debug=True)
