# Developing Multi-Input Models for OCR

This project explores multi-modal deep learning for document understanding.
We combine image features, OCR text, and layout information to:

1. Classify document types (e.g., receipts, forms, invoices).
2. Extract key fields from scanned documents (similar to insurance codes).

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers & Datasets
- Tesseract / EasyOCR
- Google Colab (GPU) + VS Code (development)

## Project Structure

multi-input-ocr/
├─ data/           # Raw and processed datasets (not tracked in git)
├─ notebooks/      # Exploration and experiment notebooks
├─ src/            # Reusable python modules
├─ experiments/    # Logs, results, model checkpoints
├─ README.md
├─ requirements.txt
└─ .gitignore

## Goals

Build baselines:

- Image-only document classifier.
- Text-only classifier using OCR output.

Build multi-input models combining:

- Image encoder + text encoder.

Evaluate on public document datasets:

- RVL-CDIP (classification)
- SROIE, FUNSD (key-field extraction)
