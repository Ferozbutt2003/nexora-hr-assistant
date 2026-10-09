from src.loader import load_documents
# Import the function we just wrote from src/loader.py.

docs = load_documents()
# Run it. docs is now a list of Document objects (one per PDF page).

print(f"\nTotal Documents: {len(docs)}\n")
# len(docs) counts how many pages were loaded.
# "\n" means "new line", just to add empty lines for readability.

# Quick overview of every page
for d in docs:
    # Loop through each page; d is one Document.

    print(f"Page {d.metadata['page']} | {len(d.page_content)} characters | starts with: "
          f"{d.page_content[:60]!r}")
    # d.metadata['page']   -> the page number stored in the metadata (starts at 0!)
    # len(d.page_content)  -> how many characters of text this page has
    # d.page_content[:60]  -> only the first 60 characters, as a preview
    # !r                   -> shows the text in its "raw" form, so hidden line breaks appear as \n
    # The two f"..." pieces on separate lines are joined automatically into one string.

# Look closely at one page (it contains the Leave table)
print("\n" + "=" * 60)
# "=" * 60 repeats "=" sixty times, making a divider line.

print("METADATA of page 1:", docs[1].metadata)
# docs[1] is the SECOND item in the list (lists start at 0), i.e. the page after the cover.
# .metadata shows the "envelope": source file, page number, etc.

print("=" * 60)
# Another divider line.

print(docs[1].page_content)
# Print the FULL text of that page, so we can inspect how the Leave table came out.