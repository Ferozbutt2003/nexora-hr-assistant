"""Explore embeddings: vector size, meaning-based similarity, and a manual search."""
import numpy as np

from src.loader import load_documents
from src.splitter import split_documents
from src.vectorstore import get_embeddings


def similarity(a, b):
    """Cosine similarity (vectors are normalized, so a dot product is enough)."""
    return float(np.dot(a, b))


emb = get_embeddings()
question = "How many sick days do I get?"
q_vec = emb.embed_query(question)

# Part 1: what an embedding looks like
print("\nPART 1: The question as numbers")
print(f"Length: {len(q_vec)} numbers")
print(f"First 5: {[round(x, 4) for x in q_vec[:5]]}")

# Part 2: similar meaning scores high even without shared words
sentences = [
    "Staff members are allowed twelve paid absences annually due to illness.",
    "You can take time off when you are not feeling well.",
    "Salaries are paid on the 25th of each month.",
    "The company was founded in 2015 in Dubai.",
]
s_vecs = emb.embed_documents(sentences)

print("\nPART 2: Similarity to the question (higher = closer meaning)")
for sentence, vec in zip(sentences, s_vecs):
    print(f"  {similarity(q_vec, vec):.3f}  |  {sentence}")

# Part 3: retrieval by hand, scoring every chunk against the question
chunks = split_documents(load_documents())
chunk_vecs = emb.embed_documents([c.page_content for c in chunks])
results = sorted(
    ((similarity(q_vec, vec), chunk) for chunk, vec in zip(chunks, chunk_vecs)),
    key=lambda r: r[0],
    reverse=True,
)

print(f"\nPART 3: Top 3 chunks for: '{question}'")
for score, chunk in results[:3]:
    print(f"\n  Score {score:.3f} | Chunk {chunk.metadata['chunk_id']} | page {chunk.metadata['page_label']}")
    print("  " + chunk.page_content[:200].replace("\n", " ") + "...")
