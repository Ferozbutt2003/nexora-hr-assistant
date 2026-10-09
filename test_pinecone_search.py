"""Search Pinecone directly and show the retrieved chunks with their scores."""
from src import config
from src.vectorstore import get_vectorstore

vectorstore = get_vectorstore()

# The last question is not covered by the handbook: compare its scores.
questions = [
    "How many sick days do I get?",
    "Can I work from home?",
    "What is the company's pension policy?",
]

for q in questions:
    print("\n" + "=" * 70)
    print(f"QUESTION: {q}")
    print("=" * 70)
    for doc, score in vectorstore.similarity_search_with_score(q, k=config.TOP_K):
        print(f"\n  Score {score:.3f} | Chunk {int(doc.metadata['chunk_id'])} | page {doc.metadata['page']}")
        print("  " + doc.page_content[:150].replace("\n", " ") + "...")
