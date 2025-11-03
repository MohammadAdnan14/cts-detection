# 🦷 ToothGuard AI: Automatic Dental Crack Detection

## Overview
ToothGuard AI is an AI-powered web app for detecting cracks in dental X-ray images using classical computer vision techniques (edge detection + morphology). Built on a robust backend from a Jupyter notebook, it provides instant analysis for panoramic dental X-rays.

- **Input**: Upload dental X-ray images (JPG/PNG/JPEG only).
- **Output**: Annotated image with detected cracks, YES/NO crack presence, and severity level (NONE/LOW/MEDIUM/HIGH/SEVERE).
- **Tech**: Streamlit (UI) + OpenCV/SciKit-Image (Backend).

## Features
- Real-time processing (10-30s per image).
- Teeth-only segmentation (ignores skull/bone artifacts).
- Multi-method crack detection (Canny/Sobel/Laplacian).
- Clinical risk assessment.
- Responsive, professional UI with progress tracking.

## Setup & Run
1. Clone/Download this repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run: `streamlit run app.py`.
4. Open the local URL (e.g., http://localhost:8501) in your browser.
5. Upload a dental X-ray and analyze!

## Backend Details
- Based on edge detection & morphological ops.
- Optimized for panoramic X-rays (e.g., from Kaggle's Panoramic Dental Dataset).
- Authors: Mohammad Adnan Dalal (42), Mohammad Faqueem Khan (43), Sankalp Choubey (49).
- Institution: Ramdeobaba College, Nagpur.

## Limitations
- Best for grayscale X-rays; color images auto-converted.
- Assumes panoramic/full-mouth views.

## License
MIT License – Free to use/modify.
