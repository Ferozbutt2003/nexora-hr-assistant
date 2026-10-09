"""Inspect what the PDF loader extracts: page counts, metadata and raw text."""
from src.loader import load_documents

docs = load_documents()
print(f"\nTotal Documents: {len(docs)}\n")

for d in docs:
    print(f"Page {d.metadata['page']} | {len(d.page_content)} characters | starts with: "
          f"{d.page_content[:60]!r}")

# Full view of page 1, which contains the start of the leave table.
print("\n" + "=" * 60)
print("METADATA of page 1:", docs[1].metadata)
print("=" * 60)
print(docs[1].page_content)
