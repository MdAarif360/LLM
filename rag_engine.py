import os
from dotenv import load_dotenv

from llm_provider import ask_llm

load_dotenv()


def retrieve_context(question: str, collection, top_k: int = 5) -> list:
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, count),
        )
    except Exception:
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "doc_id": meta.get("doc_id"),
            "source_file": meta.get("source_file"),
            "page": meta.get("page"),
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def answer_question(question: str, collection) -> dict:
    chunks = retrieve_context(question, collection)

    if not chunks:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "sources": [],
        }

    context_text = ""
    for i, item in enumerate(chunks, start=1):
        context_text += (
            f"\nSource {i}\n"
            f"Document: {item['doc_id']}\n"
            f"File: {item['source_file']}\n"
            f"Page: {item['page']}\n"
            f"Text:\n{item['text']}\n"
        )

    prompt = f"""You are a document question-answering assistant.

Rules:
1. Answer only using the provided source context.
2. Do not use outside knowledge.
3. If the answer is not available in the source context, say: "The answer is not available in the provided document."
4. Mention the page number used for the answer.
5. Keep the answer clear and concise.

Source Context:
{context_text}

User Question:
{question}

Answer:"""

    answer = ask_llm(prompt)
    sources = [
        {"doc_id": c["doc_id"], "source_file": c["source_file"], "page": c["page"]}
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}
