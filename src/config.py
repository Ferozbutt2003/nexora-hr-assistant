"""Central configuration: API keys, model names and pipeline settings."""
import os

from dotenv import load_dotenv

load_dotenv()


def _setting(name, default=None):
    """Read a setting from the environment (.env locally) or from Streamlit secrets (cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


# LLM (MiniMax, OpenAI-compatible API)
LLM_API_KEY = _setting("LLM_API_KEY")
LLM_BASE_URL = _setting("LLM_BASE_URL", "https://api.minimax.io/v1")
LLM_MODEL = _setting("LLM_MODEL", "MiniMax-M3")

# Vector database (Pinecone)
PINECONE_API_KEY = _setting("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _setting("PINECONE_INDEX_NAME", "rag-course")

# Embeddings: EMBEDDING_DIM must match the Pinecone index dimension
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4

DATA_DIR = "data"

# Save chats to chat_history.json. Turn off on a public deployment, where one file
# would be shared by every visitor.
SAVE_CHAT_HISTORY = str(_setting("SAVE_CHAT_HISTORY", "true")).lower() == "true"
