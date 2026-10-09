"""Indexing pipeline: load PDFs, chunk them, embed them and upload to Pinecone.

Run with:  uv run ingest.py
Re-run whenever PDFs in the data folder are added, removed or changed.
"""
import time

from pinecone import Pinecone

from src import config
from src.loader import load_documents
from src.splitter import split_documents
from src.vectorstore import clear_index, index_chunks


def main():
    print("STEP 1/4: Loading PDFs...")
    docs = load_documents()

    print("\nSTEP 2/4: Chunking...")
    chunks = split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    print("\nSTEP 3/4: Clearing old vectors...")
    clear_index()

    print("\nSTEP 4/4: Embedding and uploading to Pinecone...")
    count = index_chunks(chunks)
    print(f"Uploaded {count} chunks.")

    # Pinecone needs a few seconds before new vectors appear in its stats.
    time.sleep(10)
    stats = (
        Pinecone(api_key=config.PINECONE_API_KEY)
        .Index(config.PINECONE_INDEX_NAME)
        .describe_index_stats()
    )
    print(f"\nDONE. Pinecone now holds {stats.total_vector_count} vectors.")


if __name__ == "__main__":
    main()
