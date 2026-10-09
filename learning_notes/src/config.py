"""
config.py - ALL project settings live here.
Other files import from this one, so you change a setting in ONE place only.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file

# --- MiniMax (the LLM that writes the answer) ---
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.minimax.io/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "MiniMax-M3")

# --- Pinecone (the cloud database that stores vectors) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-course")

# --- Embeddings (turn text into numbers; runs free on your laptop) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # this model outputs 384 numbers per text; Pinecone must match

# --- Chunking (Lesson 3) ---
CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

# --- Retrieval (Lesson 6) ---
TOP_K = 4  # how many chunks to fetch for each question

# --- Folders ---
DATA_DIR = "data"
