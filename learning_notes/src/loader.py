"""
loader.py - Step 2: read PDFs from the data folder into LangChain Documents.
"""
# The text above in triple quotes is a "docstring": a description of what this file does.

from pathlib import Path
# Path is a built-in Python tool for working with folders and file paths
# (it handles Windows "\" vs Linux "/" automatically).

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader is LangChain's ready-made tool that opens a PDF and extracts its text.

from src import config
# Import our settings file, so we can use config.DATA_DIR (the "data" folder name).


def load_documents(data_dir: str = config.DATA_DIR):
    # Define a function named load_documents.
    # data_dir is the folder to read from; if we don't pass one, it uses "data" from config.
    # ": str" is a hint saying data_dir should be text (a string).

    """Load every PDF in data_dir. Returns one Document per page."""
    # Docstring for the function: explains what it does.

    all_pages = []
    # Create an empty list. We'll fill it with pages from all PDFs.

    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        # Path(data_dir)    -> turns "data" into a folder object
        # .glob("*.pdf")    -> finds every file ending in .pdf inside that folder
        # sorted(...)       -> puts them in alphabetical order, so results are predictable
        # for pdf_path in   -> loop: handle one PDF file at a time

        pages = PyPDFLoader(str(pdf_path)).load()   # 1 Document per page
        # PyPDFLoader(...)  -> create a loader for this one PDF (it needs the path as text, hence str())
        # .load()           -> actually read the PDF; returns a list with one Document per page

        all_pages.extend(pages)
        # Add all pages of this PDF to our big list.
        # (extend adds each item; append would add the whole list as ONE item, which we don't want)

        print(f"Loaded {pdf_path.name}: {len(pages)} pages")
        # Show progress, e.g. "Loaded Nexora_Employee_Handbook.pdf: 5 pages"
        # f"..." lets us put variables inside {} in a string.
        # pdf_path.name gives just the file name, without the folder.

    return all_pages
    # Send the full list of pages back to whoever called this function.
