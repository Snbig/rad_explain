---
title: MedGemma - Radiology Explainer Demo
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Radiology Image & Report Explainer Demo. Built with MedGemma
models:
  - google/medgemma-1.5-4b-it
secrets:
  - HF_TOKEN
  - MEDGEMMA_ENDPOINT_URL
---

# Radiology Image & Report Explainer Demo - Built with MedGemma

Consider an educational scenario where interacting with a radiology image can
substantially improve learning. This demonstration shows how MedGemma might be built upon to provide a useful tool for exploring radiology images and associated reports by translating them into simple language, with visual cues to highlight the relevant areas of the image.

Powered by AI (MedGemma-4B Multimodel), this space analyzes both a sample radiology report and its corresponding Chest X-Ray/CT image. Click on any sentence in the report, and you'll receive an AI-generated explanation tailored to that specific text and visual context. When relevant, the explanation will also pinpoint the corresponding area on the X-ray/CT image.

This demonstration is for illustrative purposes only and does not represent a finished or approved product. It is not representative of compliance to any harmonized regulations or standards for quality, safety or efficacy. Any real-world application would require additional development, training, and adaptation. The experience highlighted in this demo shows MedGemma's baseline capability for the displayed task and is intended to help developers and users explore possible applications and inspire further development.

## Upload your own image

Use the main screen's dedicated **Upload** section (built with Flask + Pillow:
`PyDICOM` handles the DICOM files) to analyze your own files instead of the
bundled demo cases:

* **Chest X-Ray / 2D images**: upload a PNG/JPEG (or a single DICOM frame).
* **CT and MRI**: upload one or several DICOM files (`.dcm`) or a `.zip`
  archive containing the series. MedGemma 1.5 natively interprets volumetric
  CT *and* MRI (3D radiology); the series is windowed and down-sampled to an
  evenly sampled slice stack before being sent to the model.

The analysis prompt asks the model to describe the key findings and whether
the study appears normal or abnormal. An optional free-text question is added
to the prompt verbatim.

## Analyze a public cancer sample from IDC

The Upload tab also offers **X-Ray / CT / MRI** sample buttons that pull one
small *real* public cancer imaging series from the NCI [Imaging Data Commons
(IDC)](https://portal.imaging.datacommons.cancer.gov/) and run it through the
same pipeline — no files needed. Uses `idc-index` to pick a compact series
(≤ 80 MB, ≤ 60 instances, body-part aware) and download its DICOM files.

**Note:** This space uses a HuggingFace endpoint that may scale down to zero due to inactivity. If this occurs, please allow approximately 10 minutes for the endpoint to restart. As an alternative, the model can be deployed on ModelGarden (see the link below).

**Note for self-hosting this fork:** the app requires the `HF_TOKEN` and
`MEDGEMMA_ENDPOINT_URL` secrets to be configured on the Space (Settings >
Variables and Secrets). `MEDGEMMA_ENDPOINT_URL` should point at a MedGemma
chat-completions endpoint (e.g. a Hugging Face Inference Endpoint or an
OpenAI-compatible v1/chat/completions route).

## Run it on Colab (no Space / no endpoint needed)

`notebooks/rad_explain_colab.ipynb` runs the whole demo inside one Colab GPU
runtime. It starts `serve_medgemma.py` (a tiny local OpenAI-compatible
`/v1/chat/completions` server that runs MedGemma 1.5 in 4-bit on the T4) on
port 7861 and the Flask app itself on port 7860, then embeds the app as an
iframe. Nothing is uploaded anywhere — the images you upload are processed on
this runtime only.

To run it standalone: `pip install -r requirements.txt` then
`HF_TOKEN=<token> python serve_medgemma.py` (disable 4-bit with
`LOAD_IN_4BIT=0` on a 24 GB+ GPU).

# Links
* MedGemma HuggingFace - https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4
* MedGemma DevSite - https://developers.google.com/health-ai-developer-foundations/medgemma
* MedGemma ModelGarden - https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma
* HAI-DEF models - https://developers.google.com/health-ai-developer-foundations
