import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DevMind",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🧠 DevMind")
st.caption("Agentic code intelligence — ask anything about your codebase")

# Agent badge colours
AGENT_COLORS = {
    "qa":       ("🔵", "How it works"),
    "reviewer": ("🔴", "Security review"),
    "planner":  ("🟢", "Refactor plan"),
}

# ─────────────────────────────────────────────
# SESSION STATE — keeps chat history
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            agent = msg.get("agent", "qa")
            icon, label = AGENT_COLORS.get(agent, ("🔵", "QA"))
            st.caption(f"{icon} Handled by **{agent.upper()} AGENT** — {label}")
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask about your codebase..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

                agent  = data.get("agent", "qa")
                answer = data.get("answer", "No answer returned.")

                icon, label = AGENT_COLORS.get(agent, ("🔵", "QA"))
                st.caption(f"{icon} Handled by **{agent.upper()} AGENT** — {label}")
                st.markdown(answer)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "agent":   agent
                })

            except requests.exceptions.ConnectionError:
                err = "❌ Cannot reach DevMind API. Is `python run.py` running?"
                st.error(err)
            except requests.exceptions.Timeout:
                err = "❌ Request timed out. The model may be slow — try again."
                st.error(err)
            except Exception as e:
                err = f"❌ Unexpected error: {str(e)}"
                st.error(err)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Agents")
    st.markdown("""
🔵 **QA Agent**
Explains how code works — flow, functions, logic

🔴 **Reviewer Agent**
Finds bugs, security issues, missing error handling

🟢 **Planner Agent**
Plans refactors and architectural changes step by step
""")

    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("API status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API online ✅")
        else:
            st.error("API error")
    except:
        st.error("API offline ❌")