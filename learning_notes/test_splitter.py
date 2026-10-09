from src.loader import load_documents
# Import our Step 2 loader.

from src.splitter import split_documents
# Import the splitter we just wrote.

docs = load_documents()
# Load the PDF pages.

chunks = split_documents(docs)
# Cut them into chunks.

print(f"\n{len(docs)} pages  ->  {len(chunks)} chunks\n")
# Show how many pages became how many chunks.

for c in chunks:
    # Loop through every chunk.
    print(f"Chunk {c.metadata['chunk_id']:>2} | page {c.metadata['page_label']} | "
          f"{len(c.page_content):>3} chars | starts: {c.page_content[:50]!r}")
    # :>2 and :>3 just right-align the numbers so the columns line up neatly.
    # page_label = the human page number (from Step 2).

print("\n" + "=" * 60)
print("Which chunks mention 'Sick Leave'?")
print("=" * 60)
# Divider + title.

for c in chunks:
    # Check every chunk...
    if "Sick Leave" in c.page_content:
        # ...and keep only the ones containing the text "Sick Leave".
        print(f"\n--- Chunk {c.metadata['chunk_id']} (page {c.metadata['page_label']}) ---")
        # Header showing which chunk it is.
        print(c.page_content)
        # Print the full chunk so we can see if "12 working days" is next to it.

print("\n" + "=" * 60)
print("OVERLAP CHECK: end of chunk 1 vs start of chunk 2")
print("=" * 60)
# Divider + title.

print("END of chunk 1:  ", repr(chunks[1].page_content[-150:]))
# [-150:] = the LAST 150 characters of chunk 1.

print("START of chunk 2:", repr(chunks[2].page_content[:150]))
# [:150] = the FIRST 150 characters of chunk 2. Compare: do they share text?