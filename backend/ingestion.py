# backend/ingestion.py
import io
from typing import Dict, Any, List

import pdfplumber
from PyPDF2 import PdfReader
from PIL import Image


def _process_pdf(file_bytes: bytes) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    num_images = 0

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            num_images += len(page.images or [])
            pages.append({"page_num": i, "text": text})

    num_pages = len(pages)
    legible = any(p["text"].strip() for p in pages)

    return {
        "num_pages": num_pages,
        "num_images": num_images,
        "pages": pages,
        "legible": legible,
    }


def _process_image(file_bytes: bytes) -> Dict[str, Any]:
    # For now we don't OCR, just mark 1 page & 1 image.
    # You can plug in pytesseract later.
    try:
        Image.open(io.BytesIO(file_bytes))
        legible = True
    except Exception:
        legible = False

    pages = [{"page_num": 1, "text": ""}]
    return {
        "num_pages": 1,
        "num_images": 1,
        "pages": pages,
        "legible": legible,
    }


def process_file(uploaded_file) -> Dict[str, Any]:
    """
    Accepts a Streamlit UploadedFile, returns a normalized doc_info dict.
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        info = _process_pdf(file_bytes)
    else:
        info = _process_image(file_bytes)

    info["filename"] = uploaded_file.name
    return info
