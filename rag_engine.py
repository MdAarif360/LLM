import os
import json
import re
from dotenv import load_dotenv

from llm_provider import ask_llm

load_dotenv()

# Keywords that signal the user wants a chart / table / infographic
_VIZ_KEYWORDS = frozenset([
    "infographic", "chart", "graph", "plot", "visualize", "visualization",
    "bar chart", "pie chart", "pie", "breakdown", "price list", "item list",
    "items and price", "prices", "show items", "list items", "compare prices",
    "table of", "summarize prices", "price breakdown", "cost breakdown",
])


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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def is_visualization_request(text: str) -> bool:
    """Return True when the user's question is asking for a chart / infographic."""
    lower = text.lower()
    return any(kw in lower for kw in _VIZ_KEYWORDS)


def extract_items_prices(collection) -> dict | None:
    """
    Retrieve invoice/document context then ask the LLM to return structured
    JSON with items and prices.  Returns None if extraction fails.
    """
    # Broad retrieval terms to pull invoice line-item content
    chunks = retrieve_context(
        "items description quantity unit price total amount",
        collection,
        top_k=10,
    )
    if not chunks:
        return None

    context_text = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""From the document below, extract every line item with its price.
Return ONLY a JSON object in exactly this format — no explanation, no markdown:

{{
  "items": [
    {{"name": "item name", "quantity": 1, "unit_price": 0.0, "total": 0.0}}
  ],
  "currency": "KD",
  "subtotal": 0.0,
  "tax": 0.0,
  "grand_total": 0.0
}}

Rules:
- All price fields must be numbers (float), not strings.
- Use null for any field not found in the document.
- Include every distinct product / service listed.
- Infer the currency from the document if visible.

Document:
{context_text}

JSON:"""

    raw = ask_llm(prompt)

    # Pull the first JSON object out of the response (handles extra prose)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
