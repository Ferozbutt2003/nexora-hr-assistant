from src.vectorstore import get_vectorstore
# Our Pinecone connection.

from src import config
# To read TOP_K (4).

vectorstore = get_vectorstore()
# Connect to Pinecone (loads the embedding model for the questions).

questions = [
    "How many sick days do I get?",
    "Can I work from home?",
    "What is the company's pension policy?",
]
# Three test questions. The last one is NOT in the document: watch its scores.

for q in questions:
    # Handle one question at a time.
    print("\n" + "=" * 70)
    print(f"QUESTION: {q}")
    print("=" * 70)

    results = vectorstore.similarity_search_with_score(q, k=config.TOP_K)
    # Pinecone: embed the question, find the TOP_K most similar chunks,
    # and return each chunk together with its similarity score.

    for doc, score in results:
        # Each result is a (Document, score) pair.
        print(f"\n  Score {score:.3f} | Chunk {doc.metadata['chunk_id']} | page {doc.metadata['page']}")
        print("  " + doc.page_content[:150].replace("\n", " ") + "...")
        # Preview of the chunk text that Pinecone sent back.