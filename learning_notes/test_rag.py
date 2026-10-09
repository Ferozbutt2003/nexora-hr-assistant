from src.rag_chain import RAGChain
# Our RAG pipeline class.

rag = RAGChain()
# "Open the shop": connect to Pinecone, load embeddings, connect MiniMax (once).

# ---- Part 1: see EXACTLY what MiniMax receives, for one question ----
result = rag.ask("How many sick days do I get?", show_prompt=True)
# show_prompt=True prints the full prompt before asking.
print("ANSWER:", result["answer"])

# ---- Part 2: run all test questions ----
tests = [
    ("How many annual leave days do I get?", "24 working days"),
    ("Which days must I come to the office?", "Tuesday and Wednesday"),
    ("When is salary paid?", "25th of each month"),
    ("What's the probation period for a manager?", "6 months"),
    ("What happens if I'm rated 2?", "3-month IDP"),
    ("What's the max gift value I can accept?", "50 USD"),
    ("What is the company's pension policy?", "NOT in document -> should say it couldn't find it"),
]
# Each test = (question, the answer we expect). We'll compare by eye.

for question, expected in tests:
    # Loop through each test pair.
    result = rag.ask(question)
    # Run the full RAG pipeline.

    pages = sorted({d.metadata["page"] for d in result["sources"]})
    # {... for d in ...} builds a SET (no duplicates) of pages the chunks came from;
    # sorted() puts them in order.

    print("\n" + "=" * 70)
    print(f"Q:        {question}")
    print(f"EXPECTED: {expected}")
    print(f"ANSWER:   {result['answer']}")
    print(f"Retrieved from pages: {pages}")