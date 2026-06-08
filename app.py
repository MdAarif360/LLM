import json
from pathlib import Path

import streamlit as st
import chromadb
import plotly.express as px
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ingest_document import ingest_ocr_json_dict, ingest_ocr_json_file
from ocr_engine import ocr_pdf, create_searchable_pdf, ocr_json_to_txt
from rag_engine import answer_question, is_visualization_request, extract_items_prices

st.set_page_config(page_title="OCR PDF AI Reader", layout="wide")
st.title("OCR PDF AI Reader")
st.caption("Upload PDFs to extract text via OCR, then ask questions about your documents.")

BASE_DIR = Path(__file__).resolve().parent
OCR_TEXT_FOLDER = BASE_DIR / "data" / "ocr_text"


# ---------------------------------------------------------------------------
# Shared in-memory ChromaDB — built once per deployment, cached across users.
# Pre-bundled OCR JSON files in data/ocr_text/ are auto-indexed on first load.
# New uploads are added to the same collection during the app's lifetime.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading document index…")
def init_collection():
    client = chromadb.EphemeralClient()
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name="ocr_documents",
        embedding_function=embedding_fn,
    )

    bundled = []
    if OCR_TEXT_FOLDER.exists():
        for jf in sorted(OCR_TEXT_FOLDER.glob("*.json")):
            ingest_ocr_json_file(jf, collection)
            bundled.append(jf.stem)

    return collection, bundled


collection, bundled_docs = init_collection()

# Per-session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_docs" not in st.session_state:
    st.session_state.session_docs = []  # stems of docs uploaded this session


# ---------------------------------------------------------------------------
# Sidebar — document list
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Indexed Documents")

    all_docs = [(d, "bundled") for d in bundled_docs] + [
        (d, "uploaded") for d in st.session_state.session_docs
    ]

    if all_docs:
        for name, origin in all_docs:
            tag = "" if origin == "bundled" else " *(uploaded)*"
            st.markdown(f"- {name}{tag}")
    else:
        st.info("No documents indexed yet.")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------

def _render_invoice_chart(viz_data: dict):
    """
    Build a Plotly bar chart + summary table from extracted items/prices data.
    Returns (fig, df, summary_md) or (None, None, error_msg).
    """
    items = viz_data.get("items") or []
    if not items:
        return None, None, "No items could be extracted from the document."

    currency = viz_data.get("currency") or ""
    rows = []
    for item in items:
        name = str(item.get("name") or "Unknown")
        try:
            total = float(item.get("total") or item.get("unit_price") or 0)
        except (TypeError, ValueError):
            total = 0.0
        try:
            qty = float(item.get("quantity") or 1)
        except (TypeError, ValueError):
            qty = 1.0
        try:
            unit = float(item.get("unit_price") or 0)
        except (TypeError, ValueError):
            unit = 0.0
        rows.append({"Item": name, "Qty": qty, f"Unit Price ({currency})": unit, f"Total ({currency})": total})

    df = pd.DataFrame(rows)
    total_col = f"Total ({currency})"

    fig = px.bar(
        df,
        x="Item",
        y=total_col,
        text=total_col,
        title=f"Invoice Items & Prices ({currency})" if currency else "Invoice Items & Prices",
        color=total_col,
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
        margin=dict(t=60, b=100),
    )

    parts = []
    if viz_data.get("subtotal") is not None:
        parts.append(f"**Subtotal:** {currency} {viz_data['subtotal']:.3f}")
    if viz_data.get("tax") is not None:
        parts.append(f"**Tax:** {currency} {viz_data['tax']:.3f}")
    if viz_data.get("grand_total") is not None:
        parts.append(f"**Grand Total:** {currency} {viz_data['grand_total']:.3f}")
    summary_md = "  \n".join(parts) if parts else ""

    return fig, df, summary_md


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_chat = st.tabs(["Upload & OCR", "Chat with Documents"])


# ── Upload & OCR ────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a PDF for OCR Processing")
    st.markdown(
        "Uploaded files are processed **in-memory** — nothing is written to disk. "
        "Download your results before closing the tab."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Scanned or digital PDF — text will be extracted via Tesseract OCR",
    )

    if uploaded_file:
        # Cache raw bytes so repeated reruns don't lose the file position
        cache_key = f"bytes_{uploaded_file.name}_{uploaded_file.size}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = uploaded_file.read()
        pdf_bytes = st.session_state[cache_key]

        filename = uploaded_file.name
        doc_stem = Path(filename).stem
        ocr_key = f"ocr_{doc_stem}"

        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.info(f"**{filename}** — {len(pdf_bytes) / 1024:.1f} KB")
        with col_btn:
            run_ocr = st.button("Run OCR & Index", type="primary", use_container_width=True)

        if run_ocr:
            with st.spinner(f"Running OCR on {filename}… (may take a moment per page)"):
                ocr_json = ocr_pdf(pdf_bytes, filename)

            # Store result in session state
            st.session_state[ocr_key] = {"ocr_json": ocr_json, "pdf_bytes": pdf_bytes}

            # Add to shared ChromaDB collection
            chunk_count = ingest_ocr_json_dict(ocr_json, collection)

            if doc_stem not in st.session_state.session_docs:
                st.session_state.session_docs.append(doc_stem)

            page_count = len(ocr_json.get("pages", []))
            st.success(f"Done! {page_count} page(s) extracted · {chunk_count} chunks indexed.")

        # Show download section if OCR has been run for this file
        if ocr_key in st.session_state:
            result = st.session_state[ocr_key]
            ocr_json = result["ocr_json"]

            st.markdown("---")
            st.subheader("Download Results")

            json_bytes = json.dumps(ocr_json, ensure_ascii=False, indent=2).encode("utf-8")
            txt_bytes = ocr_json_to_txt(ocr_json).encode("utf-8")

            dl1, dl2, dl3 = st.columns(3)

            with dl1:
                st.download_button(
                    label="OCR JSON",
                    data=json_bytes,
                    file_name=f"{doc_stem}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Structured JSON with page and line data",
                )

            with dl2:
                st.download_button(
                    label="OCR Text (.txt)",
                    data=txt_bytes,
                    file_name=f"{doc_stem}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Plain text extracted from all pages",
                )

            with dl3:
                searchable_key = f"searchable_{doc_stem}"
                if searchable_key in st.session_state:
                    st.download_button(
                        label="Searchable PDF",
                        data=st.session_state[searchable_key],
                        file_name=f"{doc_stem}_searchable.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Original pages with invisible OCR text layer",
                    )
                else:
                    if st.button(
                        "Generate Searchable PDF",
                        use_container_width=True,
                        help="Embeds OCR text into the PDF so it becomes copy-paste and Ctrl+F searchable",
                    ):
                        with st.spinner("Embedding text layer into PDF…"):
                            st.session_state[searchable_key] = create_searchable_pdf(
                                result["pdf_bytes"]
                            )
                        st.rerun()

            # Text preview
            with st.expander("Preview extracted text"):
                for page in ocr_json.get("pages", []):
                    st.markdown(f"**Page {page['page']}**")
                    preview = page.get("text", "")
                    st.text(preview[:600] + ("…" if len(preview) > 600 else ""))
                    st.divider()


# ── Chat ────────────────────────────────────────────────────────────────────
with tab_chat:
    if not bundled_docs and not st.session_state.session_docs:
        st.warning(
            "No documents are indexed yet. "
            "Go to the **Upload & OCR** tab and process a PDF first."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question, or request a chart / infographic…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # ── Visualization branch ──────────────────────────────────────
            if is_visualization_request(question):
                with st.spinner("Extracting items and prices from the document…"):
                    viz_data = extract_items_prices(collection)

                if viz_data and viz_data.get("items"):
                    fig, df, summary_md = _render_invoice_chart(viz_data)

                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                    if df is not None:
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    if summary_md:
                        st.markdown(summary_md)

                    history_text = (
                        f"*Infographic generated.*\n\n{summary_md}"
                        if summary_md else "*Infographic generated.*"
                    )
                else:
                    # Extraction failed — fall back to regular Q&A
                    st.info(
                        "Could not extract structured item data. "
                        "Falling back to document Q&A."
                    )
                    result = answer_question(question, collection)
                    st.markdown(result["answer"])
                    history_text = result["answer"]

            # ── Regular Q&A branch ────────────────────────────────────────
            else:
                with st.spinner("Searching and generating answer…"):
                    result = answer_question(question, collection)

                st.markdown(result["answer"])

                if result["sources"]:
                    unique_sources = sorted(
                        {f"{s['doc_id']} — Page {s['page']}" for s in result["sources"]}
                    )
                    st.markdown("**Sources:**")
                    for src in unique_sources:
                        st.markdown(f"- {src}")

                history_text = result["answer"]

        st.session_state.messages.append({"role": "assistant", "content": history_text})
