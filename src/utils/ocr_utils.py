import pytesseract
from pathlib import Path

# Update this to your installation path
# Example: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def run_ocr(img_path):
    text = pytesseract.image_to_string(img_path)
    return text
