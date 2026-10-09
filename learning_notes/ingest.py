"""
ingest.py - Pipeline 1 (INDEXING): PDF -> load -> chunk -> embed -> Pinecone.
Run with:  uv run ingest.py
Re-run it whenever you add, remove or change PDFs in the data folder.
"""
import time
# Built-in module; we use time.sleep to wait a few seconds.

from pinecone import Pinecone
# Pinecone client, used at the end to count records.

from src import config
# Our settings.

from src.loader import load_documents
# Step 2.

from src.splitter import split_documents
# Step 3.

from src.vectorstore import clear_index, index_chunks
# Step 5.


def main():
    # All the work happens inside this function.

    print("STEP 1/4: Loading PDFs...")
    docs = load_documents()
    # PDF -> page Documents.

    print("\nSTEP 2/4: Chunking...")
    chunks = split_documents(docs)
    # Pages -> chunks.
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    print("\nSTEP 3/4: Clearing old vectors...")
    clear_index()
    # Remove old records so we don't keep ghosts.

    print("\nSTEP 4/4: Embedding and uploading to Pinecone...")
    count = index_chunks(chunks)
    # Embed + upload every chunk.
    print(f"Uploaded {count} chunks.")

    time.sleep(10)
    # Wait 10 seconds: Pinecone needs a moment before new records show up in its counts.

    stats = Pinecone(api_key=config.PINECONE_API_KEY).Index(
        config.PINECONE_INDEX_NAME
    ).describe_index_stats()
    # Ask Pinecone for statistics about the index.

    print(f"\nDONE. Pinecone now holds {stats.total_vector_count} vectors.")
    # Should equal the number of chunks (16). If it's 32, we have duplicates!


if __name__ == "__main__":
    # This line means: "only run main() when this file is run directly",
    # NOT when another file imports something from it. Standard Python habit.
    main()