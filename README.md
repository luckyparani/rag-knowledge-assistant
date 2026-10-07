# Company Knowledge Assistant — RAG Final Project

A Retrieval-Augmented Generation (RAG) app: upload PDF/TXT documents, ask
questions in a chat interface, get answers grounded in those documents with
source citations.

**Stack:** Streamlit (UI) · sentence-transformers (embeddings) · ChromaDB
(vector store) · Groq (LLM)

## Architecture

```
User
  |
Streamlit UI (app.py)
  |
Document Upload -> load_document() -> chunk_text() -> store_chunks()
  |                                                        |
  |                                              ChromaDB (persistent, on disk)
  |
User Question -> retrieve_relevant_chunks() -> Top-K chunks (+ source metadata)
  |
generate_answer() -> System Prompt + Context + Chat History + Question -> Groq LLM
  |
Answer + Sources -> shown in chat, saved to session history
```

`rag_core.py` holds all RAG logic (loading, chunking, embedding, storage,
retrieval, generation). `app.py` is only the Streamlit interface — it imports
and calls functions from `rag_core.py`. Keeping these separate means the core
logic can be tested, reused, or swapped to a different UI without touching it.

## Features

- Multi-document upload (PDF + TXT), processed on demand from the sidebar
- Sentence-based chunking with overlap (no sentence ever gets cut mid-word)
- Persistent vector storage (ChromaDB) — survives app restarts, unlike an
  in-memory store
- Source citations under every answer, plus an expandable "retrieved chunks"
  debug view
- Conversation history (follow-up questions work, e.g. "what about leave for IT staff?")
- Duplicate-upload protection (re-uploading the same file is a no-op)
- Graceful error handling: missing API key, empty knowledge base, and LLM
  API failures all show a readable message instead of crashing
- Reset / clear-chat controls in the sidebar

## Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
# Create a file named .env in this folder with:
# GROQ_API_KEY=gsk_your_key_here

# 4. Run
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`. Upload the files in
`sample_docs/` (or your own PDFs/TXTs) from the sidebar, click **Process
Documents**, then start asking questions.

> Note: the Groq model name used (`openai/gpt-oss-20b`) was confirmed available
> via `client.models.list()` on the developer's account at the time this was
> built. If you get a `model_not_found` error, run that same check against
> your own account and swap the model name in `rag_core.py` (`GROQ_MODEL`).

## Deploying to Streamlit Community Cloud (free)

Streamlit Cloud needs your code on GitHub — these steps are done from your
own GitHub/Streamlit accounts, not from inside this chat.

1. **Push this project to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial RAG project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   (`.env` and `chroma_db/` are already in `.gitignore` — your API key and
   local vector data will NOT be pushed.)

2. **Create the app**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
   - Click **New app**
   - Select your repository, branch (`main`), and main file path (`app.py`)

3. **Add your secret API key**
   - In the app's settings, open **Secrets**
   - Paste:
     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     ```
   - Save

4. **Deploy** — Streamlit installs `requirements.txt` and starts the app.
   First load is slower (downloading the embedding model), then it's fast.

### Known limitation (worth knowing for an interview)

Streamlit Community Cloud's filesystem is **ephemeral** — the `chroma_db/`
folder (and any uploaded documents) are wiped whenever the app restarts or
redeploys. That's fine for a demo/portfolio project. A production version
would point `chromadb.PersistentClient` at a managed vector DB (e.g.
Pinecone, or Chroma's own hosted offering) instead of local disk — this is
exactly the "works on my laptop" vs. "production-ready" distinction covered
in the Production RAG module of this course.

## Project Structure

```
final_rag_project/
├── app.py                  # Streamlit UI
├── rag_core.py              # All RAG logic (load, chunk, embed, store, retrieve, generate)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example # Template for API key (copy -> secrets.toml locally, or paste into Streamlit Cloud)
├── sample_docs/
│   ├── company_policy.txt
│   └── it_support_policy.pdf
└── README.md
```

## Interview-Ready Summary

"I built a RAG application that lets users upload PDF and TXT documents,
chunks them using sentence-aware splitting with overlap, embeds the chunks
with a sentence-transformers model, and stores them in a persistent Chroma
vector database with source metadata. User questions are embedded the same
way and matched via cosine similarity to retrieve the top-K most relevant
chunks, which are passed to a Groq-hosted LLM alongside a system prompt that
restricts it to answering only from the provided context — reducing
hallucination. The app maintains conversation history for natural follow-up
questions, shows source citations for every answer, and handles failure
cases (missing API key, empty knowledge base, API errors) gracefully instead
of crashing."
