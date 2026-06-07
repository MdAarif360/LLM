import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"
OCR_TEXT_FOLDER = BASE_DIR / "data" / "ocr_text"
COLLECTION_NAME = "ocr_documents"
EMBED_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _make_local_collection():
    """Create a persistent ChromaDB collection for local development only."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def ingest_ocr_json_dict(data: dict, collection) -> int:
    """Ingest an OCR JSON dict into a ChromaDB collection. Returns chunk count."""
    if isinstance(data, list):
        pages = data
        file_doc_id = "unknown"
        file_name = "unknown"
    else:
        pages = data.get("pages", [])
        file_doc_id = data.get("doc_id", "unknown")
        file_name = data.get("file_name", "unknown")

    total_chunks = 0
    for page in pages:
        doc_id = page.get("doc_id", file_doc_id)
        page_no = page.get("page")
        page_text = page.get("text", "")
        lines = page.get("lines", [])
        line_text = "\n".join(ln.get("text", "") for ln in lines if ln.get("text"))

        if not page_text.strip():
            continue

        combined = f"File: {file_name}\nPage: {page_no}\n\nOCR Text:\n{page_text}"
        if line_text and line_text.strip() != page_text.strip():
            combined += f"\n\nLine Structure:\n{line_text}"

        for idx, chunk in enumerate(chunk_text(combined)):
            collection.upsert(
                ids=[f"{doc_id}_page_{page_no}_chunk_{idx}"],
                documents=[chunk],
                metadatas=[{
                    "doc_id": doc_id,
                    "source_file": file_name,
                    "page": page_no,
                    "chunk_index": idx,
                }],
            )
            total_chunks += 1

    return total_chunks


def ingest_ocr_json_file(json_path, collection) -> int:
    """Load an OCR JSON file from disk and ingest it into a collection."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ingest_ocr_json_dict(data, collection)


def ingest_all_ocr_json(folder_path=OCR_TEXT_FOLDER, collection=None):
    """Index all JSON files in a folder. Creates a local collection if none given."""
    if collection is None:
        collection = _make_local_collection()

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"OCR text folder not found: {folder}")

    json_files = list(folder.glob("*.json"))
    if not json_files:
        return 0

    for jf in json_files:
        ingest_ocr_json_file(jf, collection)

    return len(json_files)


if __name__ == "__main__":
    ingest_all_ocr_json()
