"""
app.py - Step 8: Streamlit chat interface for the RAG chatbot.
Run with:  uv run streamlit run app.py
"""
import streamlit as st
# Streamlit: turns this Python script into a web app. "st" is the standard short name.

from src.rag_chain import RAGChain
# Our complete RAG pipeline from Steps 6 & 7.


st.set_page_config(page_title="Nexora HR Assistant", page_icon="💼")
# Browser tab title and icon. Must be the FIRST Streamlit command in the script.


@st.cache_resource
# Decorator: run the function below only ONCE, then reuse its result on every rerun.
def load_rag():
    """Create the RAG pipeline once (loads embeddings, connects Pinecone + MiniMax)."""
    return RAGChain()


rag = load_rag()
# First run: slow (setup). Every later rerun: instant (cached).


st.title("💼 Nexora HR Assistant")
# Big heading at the top of the page.
st.caption("Ask me anything about the Nexora Employee Handbook.")
# Small grey text under the title.


with st.sidebar:
    # Everything indented here appears in the left sidebar.
    st.header("About")
    st.write("A RAG chatbot: answers come only from the handbook, "
             "retrieved from Pinecone and written by MiniMax-M3.")
    if st.button("🗑️ Clear chat"):
        # st.button returns True on the rerun right after it's clicked.
        st.session_state.messages = []
        # Empty the chat history.
        st.rerun()
        # Restart the script immediately so the page shows an empty chat.


if "messages" not in st.session_state:
    # On the very first run, session_state has no "messages" yet...
    st.session_state.messages = []
    # ...so create an empty list. Each message will be a dictionary:
    # {"role": "user" or "assistant", "content": text, "sources": chunks}


def show_sources(sources):
    # Shows the retrieved chunks in a collapsible box under an answer.
    with st.expander(f"📄 Sources ({len(sources)} chunks)"):
        # st.expander = a box that's closed by default; click to open.
        for doc in sources:
            # One entry per retrieved chunk.
            st.markdown(f"**Page {doc.metadata['page']}** · chunk {int(doc.metadata['chunk_id'])}")
            # Bold page number + chunk number (int() fixes the 6.0 display issue).
            st.caption(doc.page_content[:300] + "...")
            # First 300 characters of the chunk, in small grey text.


for msg in st.session_state.messages:
    # Redraw the whole chat history on every rerun (otherwise it would vanish).
    with st.chat_message(msg["role"]):
        # A chat bubble with a user or assistant avatar.
        st.markdown(msg["content"])
        # st.markdown renders **bold**, lists, etc. properly.
        if msg.get("sources"):
            # .get returns None if "sources" is missing (user messages have none).
            show_sources(msg["sources"])


if question := st.chat_input("e.g. How many sick days do I get?"):

    history = st.session_state.messages.copy()
    # NEW: grab the history BEFORE adding the new question.
    # .copy() makes a separate list, so adding the new message below doesn't change it.

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook..."):
            try:
                result = rag.ask(question, history=history)
                # NEW: pass the history so follow-ups can be rewritten.
                answer, sources = result["answer"], result["sources"]
                searched = result["search_question"]
                # NEW: the question that was actually sent to Pinecone.
            except Exception as e:
                answer, sources, searched = f"⚠️ Something went wrong: {e}", [], None

        if searched and searched != question:
            # Only when a rewrite actually changed something...
            st.caption(f"🔎 Searched for: {searched}")
            # ...show it in small grey text. Great for learning AND for debugging.

        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    # Save the assistant's reply to history, so it's redrawn on the next rerun.
