import io
import os
from pathlib import Path

import fitz
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


# ---------------------------------------------------------------------------
# File-type sets (used by app.py too)
# ---------------------------------------------------------------------------
IMAGE_TYPES   = frozenset([".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"])
PDF_TYPES     = frozenset([".pdf"])
DOCX_TYPES    = frozenset([".docx"])
XLSX_TYPES    = frozenset([".xlsx"])          # .xls needs legacy xlrd — not supported
ALL_SUPPORTED = sorted(PDF_TYPES | IMAGE_TYPES | DOCX_TYPES | XLSX_TYPES)

# Types for which we can produce a searchable PDF
SEARCHABLE_PDF_TYPES = PDF_TYPES | IMAGE_TYPES

# OCR languages — Arabic + English by default. Override with OCR_LANG env var.
# Requires the matching Tesseract language packs (see packages.txt).
OCR_LANG = os.getenv("OCR_LANG", "eng+ara")
_OCR_DPI = int(os.getenv("OCR_DPI", "300"))

# Runtime language state — downgraded to "eng" automatically if a language
# pack (e.g. Arabic) is missing, so the app never hard-crashes.
_lang_state = {"lang": OCR_LANG}


def _current_lang() -> str:
    return _lang_state["lang"]


def _run_tess(func, img, **kwargs):
    """
    Call a pytesseract function with the configured language.
    If the language pack is missing, downgrade to English once and retry.
    """
    try:
        return func(img, lang=_current_lang(), **kwargs)
    except pytesseract.TesseractError as exc:
        if _current_lang() != "eng" and (
            "failed loading language" in str(exc).lower()
            or "could not initialize" in str(exc).lower()
            or "data file" in str(exc).lower()
        ):
            _lang_state["lang"] = "eng"
            return func(img, lang="eng", **kwargs)
        raise


# ---------------------------------------------------------------------------
# Internal OCR helper shared by PDF and image processors
# ---------------------------------------------------------------------------
def _ocr_pil_image(img: Image.Image, doc_id: str, page_num: int, file_name: str) -> dict | None:
    """Run Tesseract on a PIL image. Returns a page dict or None if no text found."""
    page_text = _run_tess(pytesseract.image_to_string, img).strip()
    data = _run_tess(pytesseract.image_to_data, img, output_type=pytesseract.Output.DICT)

    line_map = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or int(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_map.setdefault(key, []).append(word)

    lines = [{"text": " ".join(words)} for words in line_map.values()]

    if not page_text:
        return None
    return {"doc_id": doc_id, "page": page_num, "text": page_text, "lines": lines}


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def ocr_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """OCR every page of a PDF and return structured JSON."""
    doc_id = Path(filename).stem
    images = convert_from_bytes(pdf_bytes, dpi=_OCR_DPI)
    pages = [
        p for p in (
            _ocr_pil_image(img, doc_id, num, filename)
            for num, img in enumerate(images, start=1)
        )
        if p is not None
    ]
    return {"doc_id": doc_id, "file_name": filename, "pages": pages}


def create_searchable_pdf(pdf_bytes: bytes) -> bytes:
    """Embed an invisible OCR text layer into every page of a scanned PDF."""
    images = convert_from_bytes(pdf_bytes, dpi=_OCR_DPI)
    parts = [
        _run_tess(pytesseract.image_to_pdf_or_hocr, img, extension="pdf")
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


# ---------------------------------------------------------------------------
# Images  (JPG / PNG / BMP / TIFF / WEBP)
# ---------------------------------------------------------------------------
def ocr_image(file_bytes: bytes, filename: str) -> dict:
    """OCR a single raster image and return structured JSON (always page 1)."""
    doc_id = Path(filename).stem
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    page = _ocr_pil_image(img, doc_id, 1, filename)
    pages = [page] if page else []
    return {"doc_id": doc_id, "file_name": filename, "pages": pages}


def image_to_searchable_pdf(file_bytes: bytes) -> bytes:
    """Convert a single image to a one-page searchable PDF."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return _run_tess(pytesseract.image_to_pdf_or_hocr, img, extension="pdf")


# ---------------------------------------------------------------------------
# DOCX  (Word documents)
# ---------------------------------------------------------------------------
def extract_docx(file_bytes: bytes, filename: str) -> dict:
    """
    Extract text from a Word document.
    Headings are prefixed with # marks. Table cells are joined with | separators.
    Content is grouped into virtual pages of 40 blocks each.
    """
    from docx import Document as _DocxDoc

    doc_id = Path(filename).stem
    doc = _DocxDoc(io.BytesIO(file_bytes))
    blocks: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""
        if style.startswith("Heading"):
            level = "".join(c for c in style if c.isdigit()) or "1"
            text = "#" * int(level) + " " + text
        blocks.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    if not blocks:
        return {"doc_id": doc_id, "file_name": filename, "pages": []}

    PAGE_SIZE = 40
    pages = []
    for page_num, i in enumerate(range(0, len(blocks), PAGE_SIZE), start=1):
        chunk = blocks[i: i + PAGE_SIZE]
        page_text = "\n".join(chunk)
        lines = [{"text": b} for b in chunk]
        pages.append({
            "doc_id": doc_id,
            "page": page_num,
            "text": page_text,
            "lines": lines,
        })

    return {"doc_id": doc_id, "file_name": filename, "pages": pages}


# ---------------------------------------------------------------------------
# XLSX  (Excel workbooks)
# ---------------------------------------------------------------------------
def extract_xlsx(file_bytes: bytes, filename: str) -> dict:
    """
    Extract every sheet from an Excel workbook.
    Each sheet becomes one 'page'. Rows are pipe-separated strings.
    Requires openpyxl (already in requirements).
    """
    import pandas as pd

    doc_id = Path(filename).stem
    xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    pages = []

    for page_num, sheet_name in enumerate(xl.sheet_names, start=1):
        df = xl.parse(sheet_name, header=None, dtype=str).fillna("")
        df = df[df.apply(lambda r: r.str.strip().any(), axis=1)]

        if df.empty:
            continue

        rows_text = [
            " | ".join(str(v).strip() for v in row)
            for _, row in df.iterrows()
        ]
        page_text = f"Sheet: {sheet_name}\n" + "\n".join(rows_text)
        lines = [{"text": f"Sheet: {sheet_name}"}] + [{"text": r} for r in rows_text]

        pages.append({
            "doc_id": doc_id,
            "page": page_num,
            "text": page_text,
            "lines": lines,
        })

    return {"doc_id": doc_id, "file_name": filename, "pages": pages}


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
def process_document(file_bytes: bytes, filename: str) -> dict:
    """
    Route to the correct processor based on file extension.
    All processors return: {doc_id, file_name, pages: [{doc_id, page, text, lines}]}
    so the rest of the pipeline (ingest → ChromaDB → RAG) works unchanged.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return ocr_pdf(file_bytes, filename)
    if ext in IMAGE_TYPES:
        return ocr_image(file_bytes, filename)
    if ext in DOCX_TYPES:
        return extract_docx(file_bytes, filename)
    if ext in XLSX_TYPES:
        return extract_xlsx(file_bytes, filename)
    raise ValueError(
        f"Unsupported file type '{ext}'. "
        f"Supported: {', '.join(ALL_SUPPORTED)}"
    )


def can_make_searchable_pdf(filename: str) -> bool:
    """True only for PDF and image files (types that required OCR)."""
    return Path(filename).suffix.lower() in SEARCHABLE_PDF_TYPES


# ---------------------------------------------------------------------------
# Shared util
# ---------------------------------------------------------------------------
def ocr_json_to_txt(ocr_json: dict) -> str:
    """Convert the common OCR JSON schema to plain text."""
    lines = [f"Document: {ocr_json.get('file_name', 'unknown')}\n"]
    for page in ocr_json.get("pages", []):
        lines.append(f"\n--- Page {page['page']} ---\n")
        lines.append(page.get("text", ""))
    return "\n".join(lines)
