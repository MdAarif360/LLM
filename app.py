import streamlit as st
from ingest_document import ingest_all_ocr_json
from rag_engine import answer_question

st.set_page_config(
    page_title="OCR PDF AI Reader",
    layout="wide"
)

st.title("OCR PDF AI Reader")
st.caption("Ask questions from your OCR-processed PDF documents.")

with st.sidebar:
    st.header("Document Indexing")
    if st.button("Index OCR Documents", use_container_width=True):
        with st.spinner("Indexing…"):
            try:
                indexed_count = ingest_all_ocr_json()
                st.success(f"Indexed {indexed_count} OCR JSON file(s).")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Indexing failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question from the document...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            result = answer_question(question)

            st.markdown(result["answer"])

            if result["sources"]:
                st.markdown("### Sources")
                unique_sources = set()

                for src in result["sources"]:
                    key = f"{src['doc_id']} - Page {src['page']}"
                    unique_sources.add(key)

                for source in sorted(unique_sources):
                    st.markdown(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })