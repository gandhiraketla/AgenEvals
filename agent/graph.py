import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from agent.tools import stock_price_fetcher, news_fetcher, calculator, portfolio_analyzer

load_dotenv()

AGENT_NAME = "stock-market-agent"

TOOLS = [stock_price_fetcher, news_fetcher, calculator, portfolio_analyzer]

SYSTEM_PROMPT = """You are a professional stock market and financial analysis assistant.

You have access to the following tools:
- stock_price_fetcher: Get current price, change, 52-week range and market cap for a stock ticker
- news_fetcher: Get recent news headlines and summaries for a stock or market topic
- calculator: Perform financial calculations like ROI, profit/loss, percentage change
- portfolio_analyzer: Analyze a portfolio of multiple stocks at once

Guidelines:
- Always use the most appropriate tool(s) for the user's question
- For price questions: use stock_price_fetcher
- For news/sentiment/why questions: use news_fetcher
- For math/return/profit questions: use calculator
- For multi-stock portfolio questions: use portfolio_analyzer
- For complex questions, chain multiple tools logically
- Be concise and factual in your final response
- Always cite the data returned by the tools in your answer
"""


def build_agent():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


@traceable(name=AGENT_NAME, tags=["stock-agent"], run_type="chain")
def run_agent(query: str) -> dict:
    """Run the stock market agent and return response + tool trace."""
    agent = build_agent()

    config = {
        "run_name": AGENT_NAME,
        "tags": ["stock-agent"],
        "metadata": {"query": query},
    }

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    messages = result["messages"]
    final_response = messages[-1].content

    tool_calls_used = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_used.append({
                    "tool": tc["name"],
                    "input": tc["args"],
                })
        if hasattr(msg, "name") and msg.name:
            for existing in tool_calls_used:
                if existing["tool"] == msg.name and "output" not in existing:
                    existing["output"] = msg.content
                    break

    return {
        "response": final_response,
        "tools_used": tool_calls_used,
    }
