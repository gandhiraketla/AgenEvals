import streamlit as st

st.set_page_config(
    page_title="Stock Market Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Market AI Agent")
st.markdown("""
Welcome to the **Stock Market Agent** — an AI-powered financial assistant built with
**LangGraph**, **LangSmith**, and **OpenAI GPT-4o**.

Use the sidebar to navigate between pages:
- **Chat** — Ask financial questions and watch the agent use tools in real time
- **Evals** — Run evaluations and see LLM-as-judge scores

> 📊 **Traces** are viewed directly in [LangSmith](https://smith.langchain.com) under the `stock-market-agent` project.
""")

st.info("Navigate using the sidebar to get started.")
