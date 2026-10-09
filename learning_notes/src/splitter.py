"""
splitter.py - Step 3: cut page Documents into smaller chunk Documents.
"""
# Docstring: describes what this file does.

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import LangChain's smart splitter (tries nice cut points first).

from src import config
# Import our settings, to read CHUNK_SIZE and CHUNK_OVERLAP.


def split_documents(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    # Define the function. docs = list of page Documents from the loader.
    # chunk_size and chunk_overlap default to the values in config.py,
    # but we can pass different numbers to experiment.

    """Split Documents into overlapping chunks. Metadata is copied to every chunk."""
    # Docstring for the function.

    splitter = RecursiveCharacterTextSplitter(
        # Create the splitter object with our rules:
        chunk_size=chunk_size,          # max characters per chunk
        chunk_overlap=chunk_overlap,    # characters shared between neighbouring chunks
        separators=["\n\n", "\n", ". ", " ", ""],
        # Cut points in order of preference:
        # blank line -> line break -> sentence end -> space -> anywhere (last resort)
        add_start_index=True,
        # Saves the character position where each chunk starts inside its page
        # (stored in metadata as "start_index"; useful for debugging).
    )

    chunks = splitter.split_documents(docs)
    # Do the actual cutting. Each chunk is a new Document that keeps
    # its page's metadata (source, page, page_label...).

    for i, chunk in enumerate(chunks):
        # Loop through chunks. enumerate gives us a counter i (0, 1, 2...) plus the chunk.
        chunk.metadata["chunk_id"] = i
        # Give every chunk a unique number. Pinecone will need IDs in Step 5.

    return chunks
    # Send the list of chunks back.