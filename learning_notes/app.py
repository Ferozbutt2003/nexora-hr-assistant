"""
app.py - Streamlit chat interface for the Nexora HR RAG chatbot.
Run with:  uv run streamlit run app.py
"""
import json
import uuid
from pathlib import Path

import streamlit as st
from src.rag_chain import RAGChain


st.set_page_config(page_title="Nexora HR Assistant", page_icon="📘", layout="centered")


# ------------------------------------------------------------------
# STYLE
# Fonts: Bricolage Grotesque (headings) + Figtree (text), from Google Fonts.
# The memorable detail: sources appear as amber "binder tabs" (p. 2, p. 3),
# like the index tabs on a printed handbook.
# ------------------------------------------------------------------
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Figtree:wght@400;500;600&display=swap');

html, body, .stApp, .stMarkdown, .stCaption, button, input, textarea {
  font-family: 'Figtree', sans-serif;
}
.block-container { padding-top: 3rem; max-width: 760px; }

/* Header */
.masthead { border-bottom: 2px solid #1F3A68; padding-bottom: 18px; margin-bottom: 8px; }
.masthead .org { color: #5A6578; font-size: 0.95rem; margin: 0 0 4px; }
.masthead h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: 2.9rem; line-height: 1.05; letter-spacing: -0.02em;
  color: #1B2433; margin: 0; padding: 0;
}
.masthead p { color: #3A4556; font-size: 1.02rem; margin: 10px 0 0; max-width: 34em; }

/* Starter questions: quiet rows, not cards */
.st-key-starters button {
  justify-content: flex-start; text-align: left; width: 100%;
  background: #FFFFFF; border: 1px solid #D9DEE6; border-radius: 8px;
  padding: 10px 14px; color: #1B2433;
}
.st-key-starters button:hover { border-color: #1F3A68; color: #1F3A68; }
.starter-label { color: #5A6578; font-size: 0.92rem; margin: 18px 0 6px; }

/* Source tabs */
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 2px; }
.tab {
  background: #F2B544; color: #3B2A06; font-weight: 600; font-size: 0.78rem;
  padding: 3px 10px 3px 9px; border-radius: 0 6px 6px 0;
  border-left: 3px solid #B97D0F;
}
.passage {
  border-left: 3px solid #F2B544; background: #FFFFFF; border-radius: 0 6px 6px 0;
  padding: 8px 12px; margin-bottom: 8px; font-size: 0.86rem; color: #3A4556;
}
.passage b { color: #1F3A68; }

/* Sidebar: chat history list */
.st-key-history button {
  justify-content: flex-start; text-align: left; width: 100%;
  background: transparent; border: none; padding: 6px 10px; color: #3A4556;
}
.st-key-history button:hover { background: #D9DEE6; color: #1B2433; }
.st-key-current_chat button { background: #FFFFFF; color: #1F3A68; font-weight: 600; }
.side-label { color: #5A6578; font-size: 0.85rem; margin: 14px 0 4px; }
</style>
""")


# ------------------------------------------------------------------
# RAG: loaded lazily (only when the first question arrives), no spinner text.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag():
    return RAGChain()


# ------------------------------------------------------------------
# SAVED CHAT HISTORY
# All chats are saved to chat_history.json in the project folder,
# so they survive a page refresh or an app restart.
# ------------------------------------------------------------------
HISTORY_FILE = Path("chat_history.json")


def load_chats():
    """Read saved chats from disk (empty dict if there's no file yet)."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}          # damaged file: start fresh instead of crashing
    return {}


def save_chats():
    """Write all chats to disk."""
    HISTORY_FILE.write_text(json.dumps(st.session_state.chats, indent=2), encoding="utf-8")


def new_chat():
    """Create an empty chat and make it the current one."""
    chat_id = uuid.uuid4().hex[:8]            # short random id, e.g. "a3f9c1d2"
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.current = chat_id


if "chats" not in st.session_state:
    # First run in this browser tab: load saved chats from disk.
    st.session_state.chats = load_chats()
    # Open the most recent chat, or create one if none exist.
    if st.session_state.chats:
        st.session_state.current = list(st.session_state.chats)[-1]
    else:
        new_chat()

chat = st.session_state.chats[st.session_state.current]
# "chat" = the conversation currently on screen: {"title": ..., "messages": [...]}


# ------------------------------------------------------------------
# SIDEBAR: new chat, saved chats, clear options
# ------------------------------------------------------------------
with st.sidebar:
    if st.button("➕ New chat", key="new_chat_btn"):
        new_chat()
        st.rerun()

    st.markdown("<p class='side-label'>Your chats</p>", unsafe_allow_html=True)
    with st.container(key="history"):
        # Newest first. Chats that were never used are not listed.
        for chat_id, c in reversed(list(st.session_state.chats.items())):
            if not c["messages"] and chat_id != st.session_state.current:
                continue
            is_current = chat_id == st.session_state.current
            if st.button(c["title"], key="current_chat" if is_current else f"chat_{chat_id}"):
                st.session_state.current = chat_id
                st.rerun()

    st.divider()
    if st.button("🧹 Clear this chat", key="clear_btn"):
        chat["messages"] = []
        chat["title"] = "New chat"
        save_chats()
        st.rerun()
    if st.button("🗑️ Delete all chats", key="delete_all_btn"):
        st.session_state.chats = {}
        new_chat()
        save_chats()
        st.rerun()


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.html("""
<div class="masthead">
  <p class="org">People &amp; Culture, Nexora Technologies</p>
  <h1>Ask the handbook.</h1>
  <p>Leave, office days, pay, benefits and conduct rules, answered from the
     Employee Handbook 2026, with the page so you can check it.</p>
</div>
""")


def show_sources(sources):
    """Amber page tabs, plus the passages behind a click.
    sources = list of {"page", "text"} dictionaries (saved form, so they fit in JSON)."""
    pages = sorted({s["page"] for s in sources}, key=int)
    tabs = "".join(f"<span class='tab'>p. {p}</span>" for p in pages)
    st.markdown(f"<div class='tabs'>{tabs}</div>", unsafe_allow_html=True)
    with st.expander("Show the passages used"):
        for s in sources:
            snippet = s["text"][:280].replace("\n", " ")
            st.markdown(f"<div class='passage'><b>Page {s['page']}</b><br>{snippet}…</div>",
                        unsafe_allow_html=True)


# ------------------------------------------------------------------
# EMPTY STATE: starter questions
# ------------------------------------------------------------------
STARTERS = [
    "How many sick days do I get?",
    "Which days do I have to be in the office?",
    "When is salary paid each month?",
    "What happens if my performance is rated 2?",
]

if not chat["messages"]:
    st.markdown("<p class='starter-label'>Not sure where to start? Try one:</p>",
                unsafe_allow_html=True)
    with st.container(key="starters"):
        for i, text in enumerate(STARTERS):
            if st.button(text, key=f"starter_{i}"):
                st.session_state.pending = text
                st.rerun()


# ------------------------------------------------------------------
# CHAT ON SCREEN
# ------------------------------------------------------------------
AVATARS = {"user": "🙂", "assistant": "📘"}

for msg in chat["messages"]:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("searched"):
            st.caption(f"Searched the handbook for: {msg['searched']}")
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])


# ------------------------------------------------------------------
# NEW QUESTION (typed, or a clicked starter)
# ------------------------------------------------------------------
typed = st.chat_input("Ask about leave, pay, office days…")
question = typed or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
    # Only role + content are needed for follow-up rewriting.

    if not chat["messages"]:
        chat["title"] = question[:40] + ("…" if len(question) > 40 else "")
        # The first question becomes the chat's name in the sidebar.

    chat["messages"].append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        first_time = "rag_ready" not in st.session_state
        label = "Opening the handbook (first question takes a few extra seconds)…" \
            if first_time else "Reading the handbook…"
        with st.spinner(label):
            try:
                rag = load_rag()
                st.session_state.rag_ready = True
                result = rag.ask(question, history=history)
                answer = result["answer"]
                sources = [{"page": d.metadata["page"], "text": d.page_content}
                           for d in result["sources"]]
                # Turn LangChain Documents into plain dictionaries so they can be saved to JSON.
                searched = result["search_question"]
            except Exception as e:
                answer = ("The assistant couldn't reach the handbook service. "
                          f"Check your internet connection and API keys, then try again.\n\n`{e}`")
                sources, searched = [], None

        searched = searched if searched and searched != question else None
        if searched:
            st.caption(f"Searched the handbook for: {searched}")
        st.markdown(answer)
        if sources:
            show_sources(sources)

    chat["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources, "searched": searched}
    )
    save_chats()
    # Save after every answer, so nothing is lost if the app closes.
    st.rerun()
