"""
rag_core.py
Core RAG logic: document loading, chunking, embedding, vector storage,
retrieval, and generation. Kept separate from app.py (the UI) on purpose -
logic and interface should not be tangled together.
"""

import os
import re
import shutil
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---- Sentence tokenizer, with a fallback ----
# NLTK's sentence splitter is more accurate (handles "Mr. Sharma" correctly),
# but it needs a one-time data download on first run. If that download can't
# happen (offline machine, restricted network, blocked proxy), we fall back
# to a simpler regex-based splitter instead of crashing the whole app.
try:
    import nltk
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # Confirm it actually works (download may have silently failed)
    _nltk_sent_tokenize("Test sentence.")
    sent_tokenize = _nltk_sent_tokenize
except Exception:
    def sent_tokenize(text):
        """Regex fallback: splits on '.', '!', '?' followed by whitespace+capital letter.
        Less accurate than NLTK (e.g. doesn't know 'Mr.' isn't a sentence end),
        but keeps the app functional when NLTK's data can't be downloaded."""
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
        return [s.strip() for s in sentences if s.strip()]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
CHROMA_DB_PATH = "./chroma_db"

# ---- Lazy singletons: heavy objects are created once and reused ----
_embedding_model = None
_groq_client = None
_chroma_client = None
_collection = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in a .env file (local) "
                "or in Streamlit secrets (deployed)."
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_collection():
    """Persistent Chroma collection - survives app restarts (unlike an in-memory client)."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _chroma_client.get_or_create_collection(name="knowledge_base")
    return _collection


# ==== DOCUMENT LOADING ====
def load_document(file_path):
    """Extracts raw text from a .pdf or .txt file."""
    lower_path = file_path.lower()

    if lower_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    elif lower_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {file_path}. Only .pdf and .txt are supported.")


# ==== CHUNKING (sentence-based, with overlap) ====
def chunk_text(text, max_chunk_size=400, overlap_sentences=1):
    """
    Splits text into chunks without ever cutting a sentence in half.
    Groups sentences until max_chunk_size is reached, then starts a new
    chunk that overlaps the previous one by `overlap_sentences` sentences.
    """
    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ==== STORAGE ====
def is_source_already_processed(source_filename):
    """Avoids re-processing the same file twice (duplicate chunks)."""
    collection = get_collection()
    if collection.count() == 0:
        return False
    existing = collection.get(where={"source": source_filename}, limit=1)
    return len(existing["ids"]) > 0


def store_chunks(chunks, source_filename):
    """Embeds chunks and stores them with source metadata. IDs are kept unique
    across multiple documents by offsetting from the current collection size."""
    if not chunks:
        return 0

    collection = get_collection()
    model = get_embedding_model()
    existing_count = collection.count()

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{existing_count + i}")
        embeddings.append(model.encode(chunk).tolist())
        documents.append(chunk)
        metadatas.append({"source": source_filename})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(chunks)


def process_document(file_path, source_filename=None):
    """Full pipeline for one file: load -> chunk -> store. Skips files already processed."""
    source_filename = source_filename or os.path.basename(file_path)

    if is_source_already_processed(source_filename):
        return 0, "already_processed"

    text = load_document(file_path)
    chunks = chunk_text(text)
    count = store_chunks(chunks, source_filename)
    return count, "processed"


def get_all_sources():
    """Returns the list of distinct document names currently in the knowledge base."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    all_data = collection.get()
    sources = sorted(set(m["source"] for m in all_data["metadatas"]))
    return sources


def reset_knowledge_base():
    """Wipes all stored documents/chunks by deleting the collection through
    Chroma's own API (not the filesystem), avoiding Windows file-lock issues."""
    global _collection
    client_instance = _chroma_client if _chroma_client else chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client_instance.delete_collection(name="knowledge_base")
    except Exception:
        pass  # collection didn't exist yet, nothing to delete
    _collection = None

def delete_source(source_filename):
    """Deletes only the chunks belonging to one specific document."""
    collection = get_collection()
    collection.delete(where={"source": source_filename})


# ==== RETRIEVAL ====
def retrieve_relevant_chunks(query, top_k=3):
    collection = get_collection()
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    return results


# ==== GENERATION ====
SYSTEM_PROMPT = """You are a helpful company knowledge assistant. Answer the \
user's question ONLY using the context provided below. If the answer is not \
found in the context, say clearly that you don't have that information in the \
uploaded documents. Do not use any outside/general knowledge. Be concise."""


def generate_answer(query, retrieved_chunks, chat_history=None):
    """
    Builds the final prompt (System + Context + recent chat history + Question)
    and calls the LLM. Wrapped in try/except so API failures surface as a
    readable message instead of crashing the app.
    """
    try:
        client = get_groq_client()
    except ValueError as e:
        return f"⚠️ {e}"

    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context was found."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Conversational RAG: include a little recent history so follow-up
    # questions ("what about the IT one?") still make sense.
    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Sorry, I ran into an error while generating the answer: {e}"
