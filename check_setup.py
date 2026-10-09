"""Setup check: verifies API keys, MiniMax, Pinecone (creating the index) and the data folder.

Run with:  uv run check_setup.py
"""
import os
import re

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from src import config

print("=" * 50)
print("1) Checking .env keys...")
missing = [k for k in ["LLM_API_KEY", "PINECONE_API_KEY"] if not getattr(config, k)]
if missing:
    print("[FAIL] Missing in .env:", missing)
    raise SystemExit
print("[OK]   Keys found")

print("=" * 50)
print("2) Checking MiniMax...")
try:
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, timeout=60)
    r = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": "Reply with just: API is alive"}],
        temperature=1.0,
    )
    clean = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content, flags=re.DOTALL).strip()
    print(f"[OK]   {config.LLM_MODEL} says: {clean}")
except Exception as e:
    print("[FAIL]", e)

print("=" * 50)
print("3) Checking Pinecone...")
try:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    name = config.PINECONE_INDEX_NAME
    if not pc.has_index(name):
        print(f"       Index '{name}' not found, creating it (takes ~30 sec)...")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    info = pc.describe_index(name)
    print(f"[OK]   Index '{name}' | dimension={info.dimension} | metric={info.metric}")
except Exception as e:
    print("[FAIL]", e)

print("=" * 50)
print("4) Checking data folder...")
pdfs = [f for f in os.listdir(config.DATA_DIR) if f.lower().endswith(".pdf")]
print(f"[OK]   PDFs found: {pdfs}" if pdfs else "[FAIL] No PDF in data/ folder")
