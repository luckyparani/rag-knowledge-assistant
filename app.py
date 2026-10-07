"""
app.py
Streamlit UI for the Company Knowledge Assistant (final RAG project).
All RAG logic lives in rag_core.py - this file only handles the interface.
"""

import os
import tempfile
import streamlit as st

from rag_core import (
    process_document,
    retrieve_relevant_chunks,
    generate_answer,
    get_all_sources,
    delete_source,
    reset_knowledge_base,
    GROQ_API_KEY,
)

st.set_page_config(page_title="Company Knowledge Assistant", page_icon="\U0001F4C4", layout="wide")

st.title("\U0001F4C4 Company Knowledge Assistant")
st.caption("Upload documents, then ask questions. Answers are grounded only in what you upload.")

# ---- Fail fast with a clear message if the API key is missing ----
if not GROQ_API_KEY:
    st.error(
        "⚠️ GROQ_API_KEY not found.\n\n"
        "- Running locally: add it to a `.env` file in this folder.\n"
        "- Deployed on Streamlit Cloud: add it under **App settings → Secrets**."
    )
    st.stop()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar: document upload & knowledge base management ----
with st.sidebar:
    st.header("\U0001F4C1 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents", use_container_width=True):
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    count, status = process_document(tmp_path, source_filename=uploaded_file.name)
                    if status == "processed":
                        st.success(f"✅ {uploaded_file.name}: {count} chunks added")
                    else:
                        st.info(f"ℹ️ {uploaded_file.name} already processed, skipped")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        st.rerun()

    st.divider()
    st.subheader("Knowledge base")
    sources = get_all_sources()
    if sources:
        for s in sources:
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"• {s}")
            if col_b.button("🗑️", key=f"del_{s}", help=f"Remove {s}"):
                delete_source(s)
                st.rerun()
    else:
        st.write("No documents processed yet.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001F5D1️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("⚠️ Reset KB", use_container_width=True):
            reset_knowledge_base()
            st.session_state.messages = []
            st.rerun()

    with st.expander("⚙️ Retrieval settings"):
        top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=3)

# ---- Chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("\U0001F4DA Sources: " + ", ".join(msg["sources"]))

# ---- Chat input ----
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    current_sources = get_all_sources()

    if not current_sources:
        st.warning("⚠️ Please upload and process at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            retrieved_texts, retrieved_metadata, unique_sources = [], [], []
            with st.spinner("Thinking..."):
                try:
                    results = retrieve_relevant_chunks(user_query, top_k=top_k)
                    retrieved_texts = results["documents"][0] if results["documents"] else []
                    retrieved_metadata = results["metadatas"][0] if results["metadatas"] else []
                    unique_sources = sorted(set(m["source"] for m in retrieved_metadata))

                    history_for_llm = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]

                    answer = generate_answer(user_query, retrieved_texts, chat_history=history_for_llm)
                except Exception as e:
                    answer = f"⚠️ Something went wrong while answering: {e}"

            st.write(answer)
            if unique_sources:
                st.caption("\U0001F4DA Sources: " + ", ".join(unique_sources))

            if retrieved_texts:
                with st.expander("\U0001F50D Retrieved chunks (debug view)"):
                    for i, chunk in enumerate(retrieved_texts):
                        src = retrieved_metadata[i]["source"] if i < len(retrieved_metadata) else "unknown"
                        st.markdown(f"**Chunk {i + 1}** (from `{src}`)")
                        st.write(chunk)
                        st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": unique_sources}
        )
