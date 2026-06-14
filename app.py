import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import chromadb
import plotly.express as px
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ingest_document import ingest_ocr_json_dict, ingest_ocr_json_file
from ocr_engine import (
    process_document,
    create_searchable_pdf,
    image_to_searchable_pdf,
    can_make_searchable_pdf,
    ocr_json_to_txt,
    IMAGE_TYPES,
    XLSX_TYPES,
    ALL_SUPPORTED,
)
from extraction_engine import (
    extract_records_from_pages,
    records_to_dataframe,
    run_audit,
    df_to_csv_bytes,
    df_to_xlsx_bytes,
)
from rag_engine import (
    answer_question,
    is_visualization_request,
    is_analytical_query,
    is_export_request,
    answer_over_records,
    extract_chart_data,
    rewrite_query,
)

st.set_page_config(page_title="OCR PDF AI Reader", layout="wide", initial_sidebar_state="collapsed")
st.title("OCR PDF AI Reader")
st.caption(
    "Supported file types: **PDF · JPG · PNG · DOCX · XLSX** — "
    "upload a document, then ask questions or request charts."
)

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
if "structured" not in st.session_state:
    # doc_stem -> {"records": [...], "df": DataFrame}
    st.session_state.structured = {}


def combined_records_df():
    """Concatenate every extracted document's records into one DataFrame.

    Returns an empty DataFrame when nothing has been extracted yet. This is the
    complete dataset analytical chat queries run against.
    """
    frames = [
        entry["df"]
        for entry in st.session_state.structured.values()
        if entry.get("df") is not None and not entry["df"].empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
# Universal chart renderer — the LLM decides the type, we just draw it
# ---------------------------------------------------------------------------

def _build_gantt_fig(gantt_rows: list, title: str) -> "go.Figure | None":
    """Construct a Plotly Gantt/timeline figure from gantt_data records."""
    rows, cursor = [], datetime.today().replace(day=1)
    has_real_dates = False

    for i, task in enumerate(gantt_rows):
        name     = str(task.get("name") or f"Task {i+1}")
        category = str(task.get("category") or "Task")
        dur_w    = task.get("duration_weeks") or 2

        try:
            start = datetime.strptime(str(task.get("start", ""))[:10], "%Y-%m-%d")
            has_real_dates = True
        except (ValueError, TypeError):
            start = cursor
        try:
            end = datetime.strptime(str(task.get("end", ""))[:10], "%Y-%m-%d")
            has_real_dates = True
        except (ValueError, TypeError):
            try:
                end = start + timedelta(weeks=float(dur_w))
            except (ValueError, TypeError):
                end = start + timedelta(weeks=2)

        if end <= start:
            end = start + timedelta(days=1)
        cursor = end
        rows.append({"Task": name, "Start": start.strftime("%Y-%m-%d"),
                     "Finish": end.strftime("%Y-%m-%d"), "Category": category})

    if not rows:
        return None, False

    df_g = pd.DataFrame(rows)
    fig = px.timeline(df_g, x_start="Start", x_end="Finish", y="Task",
                      color="Category", title=title, template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=datetime.today().strftime("%Y-%m-%d"),
                  line_dash="dash", line_color="red",
                  annotation_text="Today", annotation_position="top right")
    fig.update_layout(margin=dict(t=60, l=240, b=80),
                      xaxis_title="Timeline", yaxis_title="", legend_title="Category")
    return fig, has_real_dates


def render_chart(chart_data: dict):
    """
    Dynamic chart renderer — handles every chart type the LLM may return.
    Returns (fig_or_None, df_or_None, note_str).
    """
    chart_type  = (chart_data.get("chart_type") or "none").lower().strip()
    title       = chart_data.get("title") or "Chart"
    description = chart_data.get("description") or ""
    x_label     = chart_data.get("x_label") or ""
    y_label     = chart_data.get("y_label") or ""
    raw_data    = chart_data.get("data") or []
    gantt_rows  = chart_data.get("gantt_data") or []
    colors      = px.colors.qualitative.Set2

    # ── Gantt / Timeline ──────────────────────────────────────────────────
    if chart_type == "gantt":
        source = gantt_rows or raw_data          # LLM sometimes puts data in wrong key
        fig, has_real = _build_gantt_fig(source, title)
        note = ("" if has_real
                else "ℹ️ *No explicit dates found — blocks are estimated placeholders.*")
        df_out = pd.DataFrame(source) if source else None
        return fig, df_out, (description + "\n\n" + note).strip()

    # ── No data returned ─────────────────────────────────────────────────
    if chart_type == "none" or not raw_data:
        return None, None, description or "No chart data could be extracted from the document."

    # ── Build normalised DataFrame ────────────────────────────────────────
    df = pd.DataFrame(raw_data)
    if "x" not in df.columns:
        return None, None, "The extracted data is missing the 'x' (category) field."
    if "y" not in df.columns:
        df["y"] = 0
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0)

    has_group = "group" in df.columns and df["group"].notna().any() and df["group"].nunique() > 1
    color_col = "group" if has_group else None
    shared    = dict(template="plotly_white", color_discrete_sequence=colors)

    # ── Table (no figure, just dataframe) ────────────────────────────────
    if chart_type == "table":
        if has_group:
            try:
                pivot = df.pivot_table(index="x", columns="group",
                                       values="y", aggfunc="sum")
                pivot.index.name = x_label or "Category"
                return None, pivot.reset_index(), description
            except Exception:
                pass
        return None, df.rename(columns={"x": x_label or "Category",
                                        "y": y_label or "Value"}), description

    # ── All Plotly chart types ────────────────────────────────────────────
    fig = None

    if chart_type == "bar":
        fig = px.bar(df, x="x", y="y", color=color_col, title=title,
                     text="y", **shared)
        fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-30)

    elif chart_type == "horizontal_bar":
        fig = px.bar(df, x="y", y="x", color=color_col, title=title,
                     text="y", orientation="h", **shared)
        fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"))

    elif chart_type == "line":
        fig = px.line(df, x="x", y="y", color=color_col, title=title,
                      markers=True, **shared)

    elif chart_type in ("pie", "donut"):
        fig = px.pie(df, names="x", values="y", title=title,
                     hole=0.4 if chart_type == "donut" else 0,
                     template="plotly_white",
                     color_discrete_sequence=colors)
        fig.update_traces(textposition="inside", textinfo="percent+label")

    elif chart_type == "area":
        fig = px.area(df, x="x", y="y", color=color_col, title=title, **shared)

    elif chart_type == "scatter":
        fig = px.scatter(df, x="x", y="y", color=color_col, title=title,
                         size_max=15, **shared)

    else:
        # Unknown type — fall back to bar
        fig = px.bar(df, x="x", y="y", color=color_col, title=title,
                     text="y", **shared)
        fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")

    if fig:
        fig.update_layout(xaxis_title=x_label or None,
                          yaxis_title=y_label or None,
                          margin=dict(t=60, b=100, l=80),
                          legend_title="")

    display_df = df.rename(columns={"x": x_label or "Category",
                                    "y": y_label or "Value"})
    return fig, display_df, description


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_chat = st.tabs(["Upload & OCR", "Chat with Documents"])


# ── Upload & Process ────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a Document")

    # Build accepted extension list from ocr_engine constants (no leading dots for Streamlit)
    accepted_exts = [e.lstrip(".") for e in ALL_SUPPORTED]

    # Human-readable type labels for the UI
    _TYPE_LABELS = {
        ".pdf": ("📄", "PDF"),
        ".jpg": ("🖼️", "Image"),  ".jpeg": ("🖼️", "Image"),
        ".png": ("🖼️", "Image"),  ".bmp":  ("🖼️", "Image"),
        ".tiff":("🖼️", "Image"),  ".webp": ("🖼️", "Image"),
        ".docx":("📝", "Word"),
        ".xlsx":("📊", "Excel"),
    }

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=accepted_exts,
        help="PDF · Images (JPG/PNG/BMP/TIFF) · Word (.docx) · Excel (.xlsx)",
    )

    if uploaded_file:
        # Cache raw bytes — prevents losing file position on Streamlit reruns
        cache_key = f"bytes_{uploaded_file.name}_{uploaded_file.size}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = uploaded_file.read()
        file_bytes = st.session_state[cache_key]

        filename  = uploaded_file.name
        doc_stem  = Path(filename).stem
        file_ext  = Path(filename).suffix.lower()
        ocr_key   = f"ocr_{doc_stem}"
        icon, type_label = _TYPE_LABELS.get(file_ext, ("📁", "Document"))

        # ── Info row ──
        c_info, c_btn = st.columns([3, 1])
        with c_info:
            st.info(f"{icon} **{filename}** · {type_label} · {len(file_bytes) / 1024:.1f} KB")
        with c_btn:
            btn_label = (
                "Run OCR & Index"
                if file_ext in IMAGE_TYPES | {".pdf"}
                else "Extract & Index"
            )
            run_btn = st.button(btn_label, type="primary", use_container_width=True)

        # ── Processing ──
        if run_btn:
            spinner_msg = (
                f"Running OCR on {filename}… (may take a moment per page)"
                if file_ext in IMAGE_TYPES | {".pdf"}
                else f"Extracting text from {filename}…"
            )
            with st.spinner(spinner_msg):
                try:
                    ocr_json = process_document(file_bytes, filename)
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                    ocr_json = None

            if ocr_json is not None:
                st.session_state[ocr_key] = {
                    "ocr_json":   ocr_json,
                    "file_bytes": file_bytes,
                    "file_ext":   file_ext,
                }
                chunk_count = ingest_ocr_json_dict(ocr_json, collection)
                if doc_stem not in st.session_state.session_docs:
                    st.session_state.session_docs.append(doc_stem)

                page_count = len(ocr_json.get("pages", []))
                unit = "sheet(s)" if file_ext in XLSX_TYPES else "page(s)"
                st.success(f"Done! {page_count} {unit} extracted · {chunk_count} chunks indexed.")

        # ── Results & Downloads ──
        if ocr_key in st.session_state:
            res       = st.session_state[ocr_key]
            ocr_json  = res["ocr_json"]
            fb        = res["file_bytes"]
            fext      = res["file_ext"]

            st.markdown("---")
            st.subheader("Downloads")

            json_bytes = json.dumps(ocr_json, ensure_ascii=False, indent=2).encode("utf-8")
            txt_bytes  = ocr_json_to_txt(ocr_json).encode("utf-8")

            dl1, dl2, dl3 = st.columns(3)

            with dl1:
                st.download_button(
                    "Extracted JSON",
                    data=json_bytes,
                    file_name=f"{doc_stem}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Structured JSON with page/sheet and line data",
                )
            with dl2:
                st.download_button(
                    "Extracted Text (.txt)",
                    data=txt_bytes,
                    file_name=f"{doc_stem}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Full plain-text extraction across all pages/sheets",
                )
            with dl3:
                if can_make_searchable_pdf(filename):
                    s_key = f"searchable_{doc_stem}"
                    if s_key in st.session_state:
                        st.download_button(
                            "Searchable PDF",
                            data=st.session_state[s_key],
                            file_name=f"{doc_stem}_searchable.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            help="Original image with invisible OCR text layer — Ctrl+F searchable",
                        )
                    else:
                        if st.button(
                            "Generate Searchable PDF",
                            use_container_width=True,
                            help="Only for PDF / image inputs — embeds OCR text so the file is searchable",
                        ):
                            with st.spinner("Embedding OCR text layer…"):
                                if fext == ".pdf":
                                    st.session_state[s_key] = create_searchable_pdf(fb)
                                else:
                                    st.session_state[s_key] = image_to_searchable_pdf(fb)
                            st.rerun()
                else:
                    st.caption("ℹ️ Searchable PDF not applicable — source is already native text.")

            # ── Content preview ──
            with st.expander("Preview extracted content"):
                # For images: show the actual image first
                if fext in IMAGE_TYPES:
                    st.image(
                        io.BytesIO(fb),
                        caption=filename,
                        use_container_width=True,
                    )
                    st.divider()

                # For XLSX: also show the raw dataframe of the first sheet
                if fext in XLSX_TYPES:
                    try:
                        df_preview = pd.read_excel(
                            io.BytesIO(fb), sheet_name=0, engine="openpyxl"
                        )
                        st.markdown("**Spreadsheet preview (Sheet 1):**")
                        st.dataframe(df_preview.head(20), use_container_width=True)
                        st.divider()
                    except Exception:
                        pass

                # Text preview for all types
                page_label = "Sheet" if fext in XLSX_TYPES else "Page"
                for page in ocr_json.get("pages", []):
                    st.markdown(f"**{page_label} {page['page']}**")
                    preview = page.get("text", "")
                    st.text(preview[:600] + ("…" if len(preview) > 600 else ""))
                    st.divider()

            # ── Structured extraction (for analytics / audit / export) ──
            st.markdown("---")
            st.subheader("Structured Records  ·  Analytics / Audit / Export")
            st.caption(
                "Convert this document into a verifiable table of records. Each page is "
                "read individually; unreadable values are left blank (never guessed). "
                "Required before analytical chat questions (totals, duplicates, audits, exports)."
            )

            already = doc_stem in st.session_state.structured
            extract_label = (
                "Re-extract Structured Records" if already else "Extract Structured Records"
            )

            if st.button(extract_label, type="secondary", use_container_width=True):
                pages = ocr_json.get("pages", [])
                if not pages:
                    st.warning("No text pages to extract from.")
                else:
                    prog = st.progress(0.0, text="Starting extraction…")

                    def _cb(done, total, page_no):
                        frac = done / total if total else 1.0
                        label = (
                            f"Extracting page {page_no}… ({done}/{total})"
                            if page_no else f"Finalising… ({done}/{total})"
                        )
                        prog.progress(min(frac, 1.0), text=label)

                    records = extract_records_from_pages(pages, doc_stem, progress_cb=_cb)
                    df_records = records_to_dataframe(records)
                    st.session_state.structured[doc_stem] = {
                        "records": records, "df": df_records,
                    }
                    prog.empty()
                    st.success(f"Extracted {len(records)} record(s) from {len(pages)} page(s).")

            # Show extracted table + downloads + audit if available
            if doc_stem in st.session_state.structured:
                df_records = st.session_state.structured[doc_stem]["df"]

                if df_records.empty:
                    st.info("No structured records were found in this document.")
                else:
                    st.markdown(f"**{len(df_records)} record(s) extracted:**")
                    st.dataframe(df_records, use_container_width=True, hide_index=True)

                    ex1, ex2 = st.columns(2)
                    with ex1:
                        st.download_button(
                            "Download CSV",
                            data=df_to_csv_bytes(df_records),
                            file_name=f"{doc_stem}_records.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with ex2:
                        st.download_button(
                            "Download Excel",
                            data=df_to_xlsx_bytes(df_records),
                            file_name=f"{doc_stem}_records.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                    with st.expander("Automated audit findings"):
                        for finding in run_audit(df_records):
                            st.markdown(f"- {finding}")


# ── Chat ────────────────────────────────────────────────────────────────────
with tab_chat:
    if not bundled_docs and not st.session_state.session_docs:
        st.warning(
            "No documents are indexed yet. "
            "Go to the **Upload & OCR** tab and process a PDF first."
        )

    # Status banner: is the structured dataset available for analytics?
    _records_df = combined_records_df()
    _has_records = not _records_df.empty
    if _has_records:
        st.caption(
            f"✅ {len(_records_df)} structured record(s) available — "
            "you can ask for totals, group-bys, duplicates, audits, charts, or CSV/Excel exports."
        )
    elif bundled_docs or st.session_state.session_docs:
        st.caption(
            "ℹ️ For totals, duplicate detection, audits or exports, first click "
            "**Extract Structured Records** on the Upload tab. Plain Q&A works without it."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question, request a chart, totals, an audit, or a CSV/Excel export…")

    if question:
        # Snapshot history BEFORE appending the current turn.
        prior_history = list(st.session_state.messages)

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            # ── 1. Export request (CSV / Excel) — needs structured records ──
            if _has_records and is_export_request(question):
                st.markdown("Here is your downloadable export of all extracted records:")
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.download_button(
                        "Download CSV",
                        data=df_to_csv_bytes(_records_df),
                        file_name="extracted_records.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with ec2:
                    st.download_button(
                        "Download Excel",
                        data=df_to_xlsx_bytes(_records_df),
                        file_name="extracted_records.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                st.dataframe(_records_df, use_container_width=True, hide_index=True)
                history_text = (
                    f"*Prepared CSV and Excel export of {len(_records_df)} record(s). "
                    "Downloads are also available on the Upload tab.*"
                )

            # ── 2. Visualization path ─────────────────────────────────────
            elif is_visualization_request(question):
                with st.spinner("Analysing data and deciding the best chart type…"):
                    chart_data = extract_chart_data(
                        question, collection, chat_history=prior_history,
                        records_df=_records_df if _has_records else None,
                    )

                if chart_data is None:
                    st.error("Chart extraction failed. Try rephrasing your request.")
                    history_text = "Chart extraction failed."

                elif chart_data.get("chart_type") == "none":
                    st.info(
                        f"No chart data found: {chart_data.get('description', '')}  \n"
                        "Answering as a question instead."
                    )
                    if _has_records and is_analytical_query(question):
                        answer = answer_over_records(question, _records_df, prior_history)
                    else:
                        answer = answer_question(
                            question, collection, chat_history=prior_history
                        )["answer"]
                    st.markdown(answer)
                    history_text = answer

                else:
                    fig, df, note = render_chart(chart_data)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    if df is not None and not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    if note:
                        st.markdown(note)

                    chart_label = (chart_data.get("chart_type") or "chart").title()
                    history_text = (
                        f"*{chart_label} generated: {chart_data.get('title', '')}*"
                        + (f"\n\n{note}" if note else "")
                    ).strip()

            # ── 3. Analytical query over the full structured dataset ──────
            elif _has_records and is_analytical_query(question):
                with st.spinner("Analysing the full extracted dataset…"):
                    answer = answer_over_records(question, _records_df, prior_history)
                st.markdown(answer)
                history_text = answer

            # ── 4. Plain semantic Q&A (RAG) ───────────────────────────────
            else:
                with st.spinner("Searching and generating answer…"):
                    result = answer_question(
                        question, collection, chat_history=prior_history
                    )
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
