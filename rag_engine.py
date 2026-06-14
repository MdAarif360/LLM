import os
import json
import re
from dotenv import load_dotenv

from llm_provider import ask_llm
from extraction_engine import compute_data_profile, records_to_compact_json

load_dotenv()


# ---------------------------------------------------------------------------
# Query-intent gates (used only when structured records exist)
# ---------------------------------------------------------------------------
_ANALYTICAL_GATE = frozenset([
    "how many", "count", "number of", "total", "sum", "subtotal", "average",
    "avg", "mean", "list", "show all", "show me all", "show me the", "find",
    "which", "by vehicle", "by station", "by date", "by product", "per vehicle",
    "per station", "per km", "per litre", "per liter", "cost per", "duplicate",
    "duplicates", "missing", "incomplete", "audit", "suspicious", "verify",
    "verification", "reimbursement", "summary", "summarize", "highest", "lowest",
    "most", "least", "group", "breakdown", "compare", "difference", "odometer",
    "kilometer", "kilometre", "mileage", "liters", "litres", "amount", "expense",
    "page-wise", "page wise", "date-wise", "vehicle-wise", "unreadable", "unclear",
    "without signature", "manual verification", "highest amount", "consumed",
])

_EXPORT_GATE = frozenset([
    "csv", "excel", "xlsx", "spreadsheet", "export", "excel-ready", "excel ready",
    "csv output", "download", "csv file", "excel file", "excel table",
])


def is_analytical_query(text: str) -> bool:
    """True when the question is a count/total/list/audit/group-style query."""
    lower = text.lower()
    return any(kw in lower for kw in _ANALYTICAL_GATE)


def is_export_request(text: str) -> bool:
    """True when the user is asking for a CSV / Excel / downloadable table."""
    lower = text.lower()
    return any(kw in lower for kw in _EXPORT_GATE)


# ---------------------------------------------------------------------------
# Visualization gate
# ---------------------------------------------------------------------------
_VIZ_GATE = frozenset([
    "infographic", "chart", "graph", "plot", "visualize", "visualization",
    "dashboard", "diagram", "visual",
    "bar chart", "pie chart", "line chart", "scatter", "area chart",
    "gantt", "timeline", "time line", "heatmap", "pivot table", "pivot",
    "trend", "distribution", "breakdown", "percentage of",
    "proportion of", "ratio of", "comparison of",
])


def is_visualization_request(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _VIZ_GATE)


# ---------------------------------------------------------------------------
# Conversation-history helpers
# ---------------------------------------------------------------------------

# Pronouns / demonstratives that signal a question depends on prior context
_FOLLOWUP_RE = re.compile(
    r'\b(it|its|they|their|them|this|that|these|those|he|she|his|her|the same)\b',
    re.IGNORECASE,
)


def _is_followup(question: str) -> bool:
    """
    Heuristic: is this question likely dependent on prior conversation context?
    True when the question is short OR contains referential pronouns/demonstratives.
    No LLM call — pure regex + length check.
    """
    q = question.strip()
    if len(q.split()) <= 6:
        return True
    return bool(_FOLLOWUP_RE.search(q))


def _format_history(chat_history: list, max_turns: int = 4) -> str:
    """
    Return the last `max_turns` user/assistant pairs as a readable string.
    Trims long assistant answers to keep the prompt concise.
    """
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate very long assistant turns (e.g. chart descriptions)
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_query(question: str, chat_history: list) -> str:
    """
    Rewrite a context-dependent follow-up question into a fully self-contained
    search query for ChromaDB.

    Only fires an LLM call when the question looks like a follow-up (cheap
    heuristic check first).  Falls back to the original question on any error.
    """
    if not chat_history or not _is_followup(question):
        return question

    history_str = _format_history(chat_history, max_turns=3)

    prompt = f"""You are a search-query rewriter.

Given the conversation history and a follow-up question, rewrite the question \
into a single self-contained search query that can be understood without the history.

Conversation History:
{history_str}

Follow-up Question: {question}

Rules:
- Replace pronouns and vague references with the specific entities or terms \
they refer to from the conversation.
- If the question is already fully self-contained, return it unchanged.
- Return ONLY the rewritten question — no explanation, no prefix.

Rewritten Question:"""

    try:
        rewritten = ask_llm(prompt).strip().strip('"').strip("'")
        # Guard against runaway LLM responses
        if 0 < len(rewritten.split()) <= 40:
            return rewritten
        return question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# Core RAG helpers
# ---------------------------------------------------------------------------

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


def answer_question(question: str, collection, chat_history: list = None) -> dict:
    """
    Answer a question using retrieved document context.

    chat_history : list of {"role": "user"|"assistant", "content": str}
        Pass st.session_state.messages[:-1] (exclude the current turn).
        Used for:
          1. Rewriting ambiguous follow-up queries before ChromaDB retrieval.
          2. Giving the LLM conversation context so it can resolve references.
    """
    history = chat_history or []

    # ── Step 1: rewrite follow-up queries for better retrieval ────────────
    search_query = rewrite_query(question, history)

    # ── Step 2: retrieve relevant chunks ─────────────────────────────────
    chunks = retrieve_context(search_query, collection)

    if not chunks:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "sources": [],
        }

    # ── Step 3: build document context string ────────────────────────────
    context_text = ""
    for i, item in enumerate(chunks, start=1):
        context_text += (
            f"\nSource {i}\nDocument: {item['doc_id']}\n"
            f"File: {item['source_file']}\nPage: {item['page']}\n"
            f"Text:\n{item['text']}\n"
        )

    # ── Step 4: include conversation history in the prompt ────────────────
    history_section = ""
    if history:
        history_section = (
            "\nConversation History "
            "(use this to resolve references such as 'it', 'they', 'the Licensee', etc.):\n"
            + _format_history(history, max_turns=4)
            + "\n"
        )

    prompt = f"""You are a document question-answering assistant.

Rules:
1. Base your answer primarily on the Source Context below.
2. Use the Conversation History ONLY to resolve references (pronouns, "the same", \
entity names mentioned earlier) — do not introduce facts from history that are \
not also present in the Source Context.
3. Do not use outside knowledge.
4. If the answer is genuinely absent from both the Source Context and the \
Conversation History, say exactly: "The answer is not available in the provided document."
5. Cite the page number(s) used.
6. Keep the answer clear and concise.
{history_section}
Source Context:
{context_text}

Current Question: {question}

Answer:"""

    answer = ask_llm(prompt)
    sources = [
        {"doc_id": c["doc_id"], "source_file": c["source_file"], "page": c["page"]}
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Analytics over the COMPLETE structured dataset (no RAG truncation)
# ---------------------------------------------------------------------------

def answer_over_records(question: str, df, chat_history: list = None) -> str:
    """
    Answer an analytical / list / audit question using the full extracted table.

    Anti-hallucination design:
      • All aggregates are pre-computed by pandas (compute_data_profile) and given
        to the LLM as AUTHORITATIVE numbers it must not recompute.
      • The full record set is provided so the LLM can select/list/filter rows.
      • The LLM is instructed to answer ONLY from this data and to say so when a
        value is missing rather than inventing one.
    """
    if df is None or df.empty:
        return ("No structured records have been extracted yet. Open the **Upload & OCR** "
                "tab, process the document, then click **Extract Structured Records**.")

    profile = compute_data_profile(df)
    records_json, truncated = records_to_compact_json(df)

    history = chat_history or []
    history_section = ""
    if history:
        history_section = "\nConversation History:\n" + _format_history(history, 3) + "\n"

    truncation_note = ""
    if truncated:
        truncation_note = (
            "\nNOTE: Only a subset of raw rows is shown below, but the Computed "
            "Aggregates above cover ALL records and are complete and authoritative. "
            "For a full row-by-row listing, tell the user to use the CSV/Excel export.\n"
        )

    prompt = f"""You are a meticulous data analyst. Answer the user's question about a \
dataset that was extracted from a scanned document.

ABSOLUTE RULES:
1. Use ONLY the data provided below. Never invent, assume, or fill in values.
2. The "Computed Aggregates" are exact, pandas-computed, and AUTHORITATIVE. When the
   question needs a total, count, average, or group sum, take the number directly from
   there. Do NOT recompute or estimate it yourself.
3. Treat null / blank / missing values as missing — never guess them.
4. When listing or filtering records, include each record's _page (and _position) so
   the user can verify it in the original document.
5. If the answer is not derivable from this data, say exactly:
   "This information is not available in the extracted data."
6. Be concise. Use markdown tables for lists. Show the currency/units exactly as they
   appear in the data — do not assume a currency that isn't present.
{truncation_note}
Computed Aggregates (authoritative):
{profile}

Full Records (JSON):
{records_json}
{history_section}
Question: {question}

Answer:"""

    return ask_llm(prompt)


# ---------------------------------------------------------------------------
# LLM-driven universal chart extractor
# ---------------------------------------------------------------------------

def extract_chart_data(question: str, collection, chat_history: list = None,
                       records_df=None) -> dict | None:
    """
    Decide chart type and extract data from the document — entirely LLM-driven.

    If `records_df` is provided (structured records were extracted), the chart is
    built from the COMPLETE table + its exact pandas aggregates, so charts like
    "total amount by station" use every record — not just 10 retrieved chunks.
    Otherwise it falls back to RAG retrieval over the indexed text.
    """
    history = chat_history or []

    if records_df is not None and not records_df.empty:
        # Use the full structured dataset + authoritative aggregates as the source
        profile = compute_data_profile(records_df)
        records_json, _ = records_to_compact_json(records_df, char_budget=40000)
        context_text = (
            f"Computed Aggregates (exact, authoritative — use these numbers):\n{profile}\n\n"
            f"Full Records (JSON):\n{records_json}"
        )
    else:
        search_query = rewrite_query(question, history)
        chunks = retrieve_context(search_query, collection, top_k=10)
        if not chunks:
            return {"chart_type": "none", "description": "No indexed documents found."}
        context_text = "\n\n".join(c["text"] for c in chunks)

    history_section = ""
    if history:
        history_section = (
            "\nConversation History (for context on follow-up chart requests):\n"
            + _format_history(history, max_turns=3)
            + "\n"
        )

    prompt = f"""You are a data-visualisation expert. Given a user's request and \
the document content below, extract the relevant data and decide the best chart type.
{history_section}
User Request: {question}

Document Content:
{context_text}

Return ONLY a single JSON object — no explanation, no markdown, no code fences.

Schema:
{{
  "chart_type": "bar|horizontal_bar|line|pie|donut|area|scatter|gantt|table",
  "title": "Descriptive chart title",
  "description": "One sentence about what this chart shows",
  "x_label": "X-axis label (leave blank for pie/donut)",
  "y_label": "Y-axis label (leave blank for pie/donut/gantt)",
  "data": [
    {{"x": "category or date", "y": 0.0, "group": "series name — omit if single series"}}
  ],
  "gantt_data": [
    {{"name": "task", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD",
      "category": "Phase/Task/Milestone", "duration_weeks": 2}}
  ]
}}

Chart-type selection guide:
• bar            – comparing values across named categories
• horizontal_bar – same but >6 categories or long labels
• line           – trend over time / continuous sequence
• pie            – parts of a whole, ≤6 slices
• donut          – same as pie, cleaner look
• area           – cumulative or stacked trend
• scatter        – correlation between two numeric values
• gantt          – project phases / tasks with start+end or duration
• table          – pivot-style multi-dimensional data

Data rules:
• "x": category label, date string, or task name (string)
• "y": numeric value — MUST be a number, never a string
• "group": optional — series name for multi-series charts
• For gantt: fill "gantt_data", leave "data" as []
• Missing gantt dates: leave start/end null, provide duration_weeks
• No relevant data found: return {{"chart_type": "none", "description": "reason"}}

JSON:"""

    raw = ask_llm(prompt)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None
