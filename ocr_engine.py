import io
from pathlib import Path

import fitz
import pytesseract
from pdf2image import convert_from_bytes


def ocr_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """OCR a PDF and return structured JSON matching the app's schema."""
    doc_id = Path(filename).stem
    images = convert_from_bytes(pdf_bytes, dpi=300)
    pages = []

    for page_num, img in enumerate(images, start=1):
        page_text = pytesseract.image_to_string(img, lang="eng").strip()

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang="eng")

        line_map = {}
        for i, word in enumerate(data["text"]):
            if not word.strip() or int(data["conf"][i]) < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            line_map.setdefault(key, []).append(word)

        lines = [{"text": " ".join(words)} for words in line_map.values()]

        if page_text:
            pages.append({
                "doc_id": doc_id,
                "page": page_num,
                "text": page_text,
                "lines": lines,
            })

    return {"doc_id": doc_id, "file_name": filename, "pages": pages}


def create_searchable_pdf(pdf_bytes: bytes) -> bytes:
    """Convert a scanned PDF into a searchable PDF with an invisible OCR text layer."""
    images = convert_from_bytes(pdf_bytes, dpi=300)
    parts = [
        pytesseract.image_to_pdf_or_hocr(img, extension="pdf", lang="eng")
        for img in images
    ]

    if len(parts) == 1:
        return parts[0]

    merged = fitz.open()
    for part in parts:
        page_doc = fitz.open(stream=part, filetype="pdf")
        merged.insert_pdf(page_doc)
        page_doc.close()

    buf = io.BytesIO()
    merged.save(buf)
    merged.close()
    return buf.getvalue()


def ocr_json_to_txt(ocr_json: dict) -> str:
    """Convert OCR JSON to a readable plain-text string."""
    lines = [f"Document: {ocr_json.get('file_name', 'unknown')}\n"]
    for page in ocr_json.get("pages", []):
        lines.append(f"\n--- Page {page['page']} ---\n")
        lines.append(page.get("text", ""))
    return "\n".join(lines)
