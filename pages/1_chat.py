import streamlit as st
from dotenv import load_dotenv
from agent.graph import run_agent

load_dotenv()

st.set_page_config(page_title="Chat — Stock Agent", page_icon="💬", layout="wide")
st.title("💬 Stock Market Agent Chat")

EXAMPLE_QUERIES = [
    "What is AAPL's current stock price?",
    "Why did NVIDIA stock drop recently?",
    "What is my ROI if I bought TSLA at $150 and it's now at $250?",
    "Should I buy MSFT given recent news?",
    "Analyze my portfolio: AAPL, TSLA, MSFT, NVDA",
    "What's 20% of $3,500?",
    "Compare AAPL and MSFT performance today",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_history" not in st.session_state:
    st.session_state.tool_history = []

with st.sidebar:
    st.header("Quick Examples")
    for q in EXAMPLE_QUERIES:
        if st.button(q, use_container_width=True):
            st.session_state.pending_query = q

    st.divider()
    st.header("Session Tool Usage")
    if st.session_state.tool_history:
        for entry in reversed(st.session_state.tool_history[-5:]):
            st.markdown(f"**Q:** {entry['query'][:40]}...")
            for t in entry["tools"]:
                st.markdown(f"- `{t}`")
    else:
        st.caption("No tools used yet.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_history = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            with st.expander(f"🔧 Tools Used ({len(msg['tools_used'])})"):
                for tc in msg["tools_used"]:
                    st.markdown(f"**`{tc['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(tc.get("input", {}))
                    with col2:
                        st.caption("Output")
                        st.json(tc.get("output", "—"))

query = st.chat_input("Ask a financial question...")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... calling tools..."):
            result = run_agent(query)

        response = result["response"]
        tools_used = result["tools_used"]

        st.markdown(response)

        if tools_used:
            with st.expander(f"🔧 Tools Used ({len(tools_used)})"):
                for tc in tools_used:
                    st.markdown(f"**`{tc['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Input")
                        st.json(tc.get("input", {}))
                    with col2:
                        st.caption("Output")
                        st.json(tc.get("output", "—"))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tools_used": tools_used,
    })

    st.session_state.tool_history.append({
        "query": query,
        "tools": [tc["tool"] for tc in tools_used],
    })
