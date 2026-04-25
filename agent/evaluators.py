import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith import Client

load_dotenv()

AGENT_NAME = "stock-market-agent"

GOLDEN_DATASET = [
    {"query": "What is AAPL's current stock price?", "expected_tools": ["stock_price_fetcher"]},
    {"query": "Why did TSLA drop today?", "expected_tools": ["news_fetcher"]},
    {"query": "What is 20% of $3500?", "expected_tools": ["calculator"], "expected_answer": "700"},
    {"query": "Is MSFT a good buy right now?", "expected_tools": ["stock_price_fetcher", "news_fetcher"]},
    {"query": "What is my ROI if I bought NVDA at $400 and it is now at $900?", "expected_tools": ["stock_price_fetcher", "calculator"]},
    {"query": "Analyze my portfolio: AAPL, TSLA, MSFT", "expected_tools": ["portfolio_analyzer"]},
    {"query": "Should I sell AAPL?", "expected_tools": ["stock_price_fetcher", "news_fetcher"]},
    {"query": "What is the best performing stock today: AAPL or MSFT?", "expected_tools": ["stock_price_fetcher", "calculator"]},
    {"query": "Summarize recent NVIDIA news", "expected_tools": ["news_fetcher"]},
    {"query": "Rebalance my portfolio of AAPL, TSLA, MSFT, NVDA", "expected_tools": ["portfolio_analyzer", "stock_price_fetcher"]},
    {"query": "What is 15% of $12000?", "expected_tools": ["calculator"], "expected_answer": "1800"},
    {"query": "Give me the 52 week high for TSLA", "expected_tools": ["stock_price_fetcher"]},
    {"query": "What news is moving the market today?", "expected_tools": ["news_fetcher"]},
    {"query": "Compare AAPL and NVDA prices", "expected_tools": ["stock_price_fetcher"]},
    {"query": "Analyze and give buy/sell recommendation for my portfolio: MSFT, AAPL", "expected_tools": ["portfolio_analyzer", "stock_price_fetcher", "news_fetcher"]},
]


def trajectory_evaluator(run_result: dict, expected_tools: list[str]) -> dict:
    """Check if agent used the expected tools (order-insensitive, partial credit)."""
    actual_tools = [tc["tool"] for tc in run_result.get("tools_used", [])]
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    if expected_set == actual_set:
        score = 1.0
        label = "exact_match"
    elif expected_set.issubset(actual_set):
        score = 0.8
        label = "superset"
    elif actual_set.issubset(expected_set):
        score = 0.6
        label = "subset"
    elif expected_set & actual_set:
        score = 0.4
        label = "partial"
    else:
        score = 0.0
        label = "no_match"

    return {
        "score": score,
        "label": label,
        "expected": list(expected_set),
        "actual": actual_tools,
    }


def output_evaluator(response: str, expected_answer: str) -> dict:
    """Exact numeric match evaluator for calculator-type queries."""
    try:
        actual_nums = [float(s.replace(",", "")) for s in response.split() if s.replace(",", "").replace(".", "").isdigit()]
        expected_num = float(expected_answer.replace(",", ""))
        match = any(abs(n - expected_num) < 0.5 for n in actual_nums)
        return {"score": 1.0 if match else 0.0, "expected": expected_answer, "found_in_response": actual_nums}
    except Exception:
        return {"score": 0.0, "expected": expected_answer, "error": "parse_error"}


def llm_as_judge(query: str, response: str) -> dict:
    """Use GPT-4o to evaluate response quality on a 1-5 rubric."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are an expert financial analyst evaluating an AI assistant's response.

User question: {query}

AI response: {response}

Rate the response on each dimension from 1 (poor) to 5 (excellent):

1. Relevance: Does the response directly answer the question?
2. Grounding: Is the answer backed by real data (prices, news, calculations)?
3. Conciseness: Is it clear and to the point without unnecessary filler?
4. Accuracy: Does the financial reasoning appear sound?

Respond in this exact format:
Relevance: <score>
Grounding: <score>
Conciseness: <score>
Accuracy: <score>
Overall: <average score>
Reasoning: <one sentence>
"""
    result = llm.invoke(prompt)
    text = result.content

    scores = {}
    for line in text.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            try:
                scores[key] = float(val.strip().split()[0])
            except (ValueError, IndexError):
                scores[key] = val.strip()

    return scores


def run_full_eval(progress_callback=None) -> list[dict]:
    """Run all evals over the golden dataset and return results."""
    from agent.graph import run_agent

    results = []
    for i, example in enumerate(GOLDEN_DATASET):
        query = example["query"]
        expected_tools = example["expected_tools"]
        expected_answer = example.get("expected_answer")

        if progress_callback:
            progress_callback(i, len(GOLDEN_DATASET), query)

        try:
            run_result = run_agent(query)
            response = run_result["response"]

            traj = trajectory_evaluator(run_result, expected_tools)
            out = output_evaluator(response, expected_answer) if expected_answer else None
            judge = llm_as_judge(query, response)

            results.append({
                "query": query,
                "expected_tools": expected_tools,
                "actual_tools": traj["actual"],
                "trajectory_score": traj["score"],
                "trajectory_label": traj["label"],
                "output_score": out["score"] if out else "N/A",
                "relevance": judge.get("relevance", "—"),
                "grounding": judge.get("grounding", "—"),
                "conciseness": judge.get("conciseness", "—"),
                "accuracy": judge.get("accuracy", "—"),
                "overall_judge": judge.get("overall", "—"),
                "judge_reasoning": judge.get("reasoning", "—"),
                "response": response,
            })
        except Exception as e:
            results.append({
                "query": query,
                "expected_tools": expected_tools,
                "actual_tools": [],
                "trajectory_score": 0.0,
                "trajectory_label": "error",
                "output_score": "N/A",
                "relevance": "—",
                "grounding": "—",
                "conciseness": "—",
                "accuracy": "—",
                "overall_judge": "—",
                "judge_reasoning": str(e),
                "response": "",
            })

    return results
