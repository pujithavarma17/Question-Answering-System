"""
Read text from PDFs and clean it up before question answering.
"""

import re
from functools import lru_cache
from io import BytesIO

from pypdf import PdfReader

# A page with fewer words than this is treated as a scanned / image page
MIN_WORDS_FOR_TEXT_PAGE = 15

PAGE_FOOTER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_ocr():
    """RapidOCR reads text from images. Returns None if it is not installed."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


def ocr_page_images(page):
    """Read the text inside the images of one PDF page (screenshots, scans)."""
    ocr = load_ocr()
    if ocr is None:
        return ""

    import numpy as np

    lines = []
    for image_file in page.images:
        try:
            image = image_file.image.convert("RGB")
        except Exception:
            continue  # unsupported image format
        if image.width < 100 or image.height < 50:
            continue  # icons and logos
        result, _ = ocr(np.array(image))
        lines += [text for _, text, _ in (result or [])]

    return "\n".join(lines)


def extract_pdf_text(file_bytes, use_ocr=True):
    """Read text from a PDF.

    "layout" mode keeps words of the same line together. The default mode
    puts every word on its own line for many PDFs (e.g. ones exported from
    Word), which breaks sentences and confuses the QA model.

    Pages that are only images (scanned pages, slide screenshots) have no
    text, so they are read with OCR instead.
    """
    pdf = PdfReader(BytesIO(file_bytes))
    pages_text = []

    for page in pdf.pages:
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            text = page.extract_text() or ""

        if use_ocr and len(text.split()) < MIN_WORDS_FOR_TEXT_PAGE:
            text = (text + "\n" + ocr_page_images(page)).strip()

        pages_text.append(text)

    return "\n\n".join(pages_text)


def clean_text(text):
    """Turn raw extracted text into clean paragraphs (one per line)."""

    # Characters that failed to decode (usually dashes like "–")
    text = text.replace("�", "-")

    # Join words split across lines with a hyphen: "synchroni-\nzation"
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)

    paragraphs = []
    current = []

    for line in text.splitlines():

        # Remove table-of-contents dot leaders: "INTRODUCTION ........ 5"
        line = re.sub(r"\.{4,}\s*\d*", " ", line)
        line = re.sub(r"\s+", " ", line).strip()

        # Remove page footers such as "Page 3", "Page 3 of 7" or a lone "3"
        if PAGE_FOOTER.match(line):
            line = ""

        if line:
            current.append(line)
        elif current:
            # A blank line ends the paragraph
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    # Join a short label such as "Submitted by:" with the text after it,
    # so the model sees "Submitted by: D.Kundan Sree ..." as one sentence.
    merged = []
    for paragraph in paragraphs:
        if merged and merged[-1].endswith(":") and len(merged[-1]) < 60:
            merged[-1] = merged[-1] + " " + paragraph
        else:
            merged.append(paragraph)

    return "\n".join(merged)
