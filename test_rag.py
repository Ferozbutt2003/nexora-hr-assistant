"""End-to-end check: show the full prompt once, then run the basic test questions."""
from src.rag_chain import RAGChain

rag = RAGChain()

result = rag.ask("How many sick days do I get?", show_prompt=True)
print("ANSWER:", result["answer"])

tests = [
    ("How many annual leave days do I get?", "24 working days"),
    ("Which days must I come to the office?", "Tuesday and Wednesday"),
    ("When is salary paid?", "25th of each month"),
    ("What's the probation period for a manager?", "6 months"),
    ("What happens if I'm rated 2?", "3-month IDP"),
    ("What's the max gift value I can accept?", "50 USD"),
    ("What is the company's pension policy?", "NOT in document -> should say it couldn't find it"),
]

for question, expected in tests:
    result = rag.ask(question)
    pages = sorted({d.metadata["page"] for d in result["sources"]})
    print("\n" + "=" * 70)
    print(f"Q:        {question}")
    print(f"EXPECTED: {expected}")
    print(f"ANSWER:   {result['answer']}")
    print(f"Retrieved from pages: {pages}")
