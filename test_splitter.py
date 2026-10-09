"""Inspect chunking: chunk sizes, the leave table, and the overlap between chunks."""
from src.loader import load_documents
from src.splitter import split_documents

docs = load_documents()
chunks = split_documents(docs)
print(f"\n{len(docs)} pages  ->  {len(chunks)} chunks\n")

for c in chunks:
    print(f"Chunk {c.metadata['chunk_id']:>2} | page {c.metadata['page_label']} | "
          f"{len(c.page_content):>3} chars | starts: {c.page_content[:50]!r}")

# Does "Sick Leave" stay next to "12 working days"?
print("\n" + "=" * 60)
print("Which chunks mention 'Sick Leave'?")
print("=" * 60)
for c in chunks:
    if "Sick Leave" in c.page_content:
        print(f"\n--- Chunk {c.metadata['chunk_id']} (page {c.metadata['page_label']}) ---")
        print(c.page_content)

# The end of one chunk should repeat at the start of the next.
print("\n" + "=" * 60)
print("OVERLAP CHECK: end of chunk 1 vs start of chunk 2")
print("=" * 60)
print("END of chunk 1:  ", repr(chunks[1].page_content[-150:]))
print("START of chunk 2:", repr(chunks[2].page_content[:150]))
