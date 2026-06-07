import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from llm_provider import ask_llm


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / os.getenv("CHROMA_PATH", "data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ocr_documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    return collection


def retrieve_context(question, top_k=5):
    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for doc, meta, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc,
            "doc_id": meta.get("doc_id"),
            "source_file": meta.get("source_file"),
            "page": meta.get("page"),
            "distance": distance
        })

    return retrieved


def answer_question(question):
    retrieved_chunks = retrieve_context(question)

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the source document.",
            "sources": []
        }

    context_text = ""

    for i, item in enumerate(retrieved_chunks, start=1):
        context_text += f"""
Source {i}
Document: {item['doc_id']}
File: {item['source_file']}
Page: {item['page']}
Text:
{item['text']}
"""

    prompt = f"""
You are a document question-answering assistant.

Rules:
1. Answer only using the provided source context.
2. Do not use outside knowledge.
3. If the answer is not available in the source context, say:
   "The answer is not available in the provided document."
4. Mention the page number used for the answer.
5. Keep the answer clear and concise.

Source Context:
{context_text}

User Question:
{question}

Answer:
"""

    answer = ask_llm(prompt)

    sources = [
        {
            "doc_id": item["doc_id"],
            "source_file": item["source_file"],
            "page": item["page"]
        }
        for item in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }