import numpy as np
# numpy = Python's math library for lists of numbers (installed with sentence-transformers).

from src.vectorstore import get_embeddings
# Our embedding function from Step 4.

from src.loader import load_documents
# Step 2 loader.

from src.splitter import split_documents
# Step 3 splitter.


def similarity(a, b):
    # Function to score how similar two embeddings are.
    return float(np.dot(a, b))
    # Because vectors are normalized, the dot product IS the cosine similarity.
    # float(...) converts numpy's number type into a normal Python number.


emb = get_embeddings()
# Load the model (first run downloads it, please be patient).

# ---------------- PART 1: What does an embedding look like? ----------------
question = "How many sick days do I get?"
# The question we'll use throughout this test.

q_vec = emb.embed_query(question)
# Turn the question into its 384 numbers.
# embed_query is used for QUESTIONS (single text).

print("\nPART 1: The question as numbers")
print(f"Length: {len(q_vec)} numbers")
# Should print 384, matching our Pinecone index dimension.
print(f"First 5: {[round(x, 4) for x in q_vec[:5]]}")
# Show only the first 5 numbers, rounded to 4 decimals.

# ---------------- PART 2: Compare meanings ----------------
sentences = [
    "Staff members are allowed twelve paid absences annually due to illness.",
    # Same meaning as the question, almost NO shared words!
    "You can take time off when you are not feeling well.",
    # Related meaning.
    "Salaries are paid on the 25th of each month.",
    # Different HR topic.
    "The company was founded in 2015 in Dubai.",
    # Unrelated.
]
# A list of test sentences, from most to least related.

s_vecs = emb.embed_documents(sentences)
# Turn all sentences into numbers at once.
# embed_documents is used for STORED TEXTS (a list).

print("\nPART 2: Similarity to the question (higher = closer meaning)")
for sentence, vec in zip(sentences, s_vecs):
    # zip pairs each sentence with its own vector, so we loop through both together.
    print(f"  {similarity(q_vec, vec):.3f}  |  {sentence}")
    # :.3f = show the score with 3 decimals.

# ---------------- PART 3: Mini search over our 16 chunks ----------------
chunks = split_documents(load_documents())
# Load the PDF and chunk it (Steps 2 + 3).

chunk_vecs = emb.embed_documents([c.page_content for c in chunks])
# Embed the TEXT of every chunk.
# [c.page_content for c in chunks] builds a list of just the chunk texts.

results = []
# Empty list to collect (score, chunk) pairs.
for chunk, vec in zip(chunks, chunk_vecs):
    # Pair each chunk with its vector.
    results.append((similarity(q_vec, vec), chunk))
    # Store the score together with the chunk.

results.sort(key=lambda r: r[0], reverse=True)
# Sort by score (r[0]), highest first.
# lambda r: r[0] is a tiny inline function meaning "use the first item of each pair".

print(f"\nPART 3: Top 3 chunks for: '{question}'")
for score, chunk in results[:3]:
    # Only the best 3 results.
    print(f"\n  Score {score:.3f} | Chunk {chunk.metadata['chunk_id']} | page {chunk.metadata['page_label']}")
    print("  " + chunk.page_content[:200].replace("\n", " ") + "...")
    # First 200 characters, with line breaks replaced by spaces so it prints on one line.