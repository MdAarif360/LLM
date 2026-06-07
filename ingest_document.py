import json
import uuid
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"
OCR_TEXT_FOLDER = BASE_DIR / "data" / "ocr_text"
COLLECTION_NAME = "ocr_documents"


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


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    return collection


def ingest_ocr_json(json_path):
    json_path = Path(json_path)

    collection = get_collection()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both document-level JSON and legacy list-of-pages JSON
    if isinstance(data, list):
        pages = data
        file_doc_id = json_path.stem
        file_name = json_path.name
    else:
        pages = data.get("pages", [])
        file_doc_id = data.get("doc_id", json_path.stem)
        file_name = data.get("file_name", json_path.name)

    total_chunks = 0

    for page in pages:
        doc_id = page.get("doc_id", file_doc_id)
        page_no = page.get("page")
        page_text = page.get("text", "")
        lines = page.get("lines", [])
        line_text = "\n".join(
            line.get("text", "") for line in lines if line.get("text")
        )

        if not page_text.strip():
            continue

        # Combine OCR text with reconstructed line structure for richer embeddings
        combined_text = (
            f"File: {file_name}\n"
            f"Page: {page_no}\n\n"
            f"OCR Text:\n{page_text}"
        )
        if line_text and line_text.strip() != page_text.strip():
            combined_text += f"\n\nLine Structure:\n{line_text}"

        chunks = chunk_text(combined_text)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_page_{page_no}_chunk_{idx}"

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{
                    "doc_id": doc_id,
                    "source_file": file_name,
                    "page": page_no,
                    "chunk_index": idx
                }]
            )

            total_chunks += 1

    print(f"Indexed: {json_path.name} | Chunks: {total_chunks}")


def ingest_all_ocr_json(folder_path=OCR_TEXT_FOLDER):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"OCR text folder not found: {folder}")

    json_files = list(folder.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {folder}")
        return 0

    for json_file in json_files:
        ingest_ocr_json(json_file)

    print("All OCR JSON files indexed successfully.")
    return len(json_files)


if __name__ == "__main__":
    ingest_all_ocr_json()