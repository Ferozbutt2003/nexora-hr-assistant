# Nexora HR Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers employee questions from a
company handbook, cites the page it used, and says so honestly when the handbook has
no answer.

## How it works

1. **Indexing** (`ingest.py`): PDF -> pages -> 800-character chunks -> embeddings
   (all-MiniLM-L6-v2) -> Pinecone.
2. **Answering** (`src/rag_chain.py`): follow-up questions are rewritten into standalone
   questions, the 4 closest chunks are retrieved from Pinecone, and MiniMax-M3 answers
   from those chunks only.
3. **Interface** (`app.py`): Streamlit chat with saved conversations and page sources.

## Tech stack

LangChain · Pinecone · HuggingFace sentence-transformers · MiniMax-M3 · Streamlit · uv

## Run locally

```bash
uv sync
cp .env.example .env        # then add your MiniMax and Pinecone keys
uv run check_setup.py       # verifies keys and creates the Pinecone index
uv run ingest.py            # indexes the PDFs in data/
uv run streamlit run app.py
```

## Evaluation

`uv run test_eval.py` runs 16 graded questions: single facts, multi-fact and
calculation questions, honest refusals, off-topic and prompt-injection attempts,
follow-ups, and regression tests for bugs found during development.

The handbook in `data/` is fictional and was created for learning purposes.
