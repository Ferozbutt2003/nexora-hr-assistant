"""Load PDF files into LangChain Documents (one Document per page)."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from src import config


def load_documents(data_dir: str = config.DATA_DIR):
    """Load every PDF in data_dir and return all pages as Documents."""
    all_pages = []
    for pdf_path in sorted(Path(data_dir).glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()
        all_pages.extend(pages)
        print(f"Loaded {pdf_path.name}: {len(pages)} pages")
    return all_pages
