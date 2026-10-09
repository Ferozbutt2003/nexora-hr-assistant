"""Split page Documents into smaller, overlapping chunks for retrieval."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    """Split Documents into chunks and give each chunk a sequential chunk_id."""
    # Prefer paragraph and line breaks as cut points; split mid-word only as a last resort.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
