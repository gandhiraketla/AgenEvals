# Stock Market AI Agent — Agent Evals Course

A fully instrumented **AI agent** built with **LangGraph**, **LangSmith**, and **OpenAI GPT-4o** for a hands-on course on building and evaluating AI agents.

The agent answers financial questions by orchestrating 4 specialized tools, with full tracing in LangSmith and a Streamlit UI to explore tool calls, traces, and evaluation results.

---

## What You Will Learn

- How to build a multi-tool AI agent using **LangGraph**
- How to trace every agent run with **LangSmith** (named runs, tags, metadata)
- How to identify common agent failure modes (wrong tool, missing tool, over-tooling)
- How to write and run **agent evaluations**:
  - Trajectory evals (did the agent pick the right tools?)
  - Output evals (is the numeric answer correct?)
  - LLM-as-Judge (GPT-4o grades response quality)
- How to build an interactive **Streamlit UI** to demo all of the above

---

## Agent Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│     GPT-4o (LangGraph)      │
│   stock-market-agent        │
└──────────┬──────────────────┘
           │ decides which tools to call
    ┌──────┴──────────────────────────┐
    │                                 │
    ▼                                 ▼
stock_price_fetcher            news_fetcher
    │                                 │
    ▼                                 ▼
calculator                    portfolio_analyzer
    │
    ▼
Final Response → LangSmith Trace
```

---

## The 4 Tools

| Tool | Purpose | Fires When |
|------|---------|-----------|
| `stock_price_fetcher` | Current price, change, 52w high/low, market cap | User asks about price, performance, valuation |
| `news_fetcher` | Recent headlines and summaries from NewsAPI | User asks why a stock moved, sentiment, news |
| `calculator` | ROI, percentage change, profit/loss, arithmetic | User asks for a numerical computation |
| `portfolio_analyzer` | Multi-stock analysis, best/worst performer | User asks about a portfolio of stocks |

### Tool Chain Examples

| Query | Tools Used |
|-------|-----------|
| "What is AAPL's price?" | `stock_price_fetcher` |
| "Why did TSLA drop?" | `news_fetcher` |
| "What's 20% of $3500?" | `calculator` |
| "Is MSFT a good buy?" | `stock_price_fetcher` + `news_fetcher` |
| "ROI on NVDA bought at $400?" | `stock_price_fetcher` + `calculator` |
| "Should I sell AAPL?" | `stock_price_fetcher` + `news_fetcher` + `calculator` |
| "Analyze AAPL, TSLA, MSFT portfolio" | All 4 tools |

---

## Failure Modes Demonstrated

| Failure | Description | Teaching Point |
|---------|-------------|---------------|
| **Wrong tool** | Agent calls `calculator` for a news question | Tool descriptions matter |
| **Missing tool** | Agent skips `stock_price_fetcher` for a buy/sell question | Incomplete reasoning |
| **Wrong order** | `calculator` fires before price is fetched | Tool dependencies matter |
| **Over-tooling** | 4 tools fire for a simple math question | LLMs over-retrieve |
| **Inconsistent** | Same query produces different tool chains | Non-determinism in agents |

---

## Project Structure

```
AgentEvals/
├── app.py                    # Streamlit entry point (home page)
├── pages/
│   ├── 1_chat.py             # Chat UI — ask questions, see tool cards
│   └── 3_evals.py            # Eval runner + LLM-as-judge results
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph agent definition + run_agent()
│   ├── tools.py              # 4 tool definitions
│   └── evaluators.py         # Trajectory, output, and LLM-as-judge evals
├── data/                     # (optional) local data files
├── .env                      # API keys (not committed)
├── .env.example              # Template for .env
├── requirements.txt
└── README.md
```

> **Traces** are viewed directly in LangSmith at https://smith.langchain.com under the `stock-market-agent` project — no separate UI needed.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/gandhiraketla/AgenEvals.git
cd AgenEvals
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# OpenAI
OPENAI_API_KEY=sk-...

# LangSmith
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=stock-market-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# News API (https://newsapi.org — free tier)
NEWS_API_KEY=...
```

| Key | Where to get it |
|-----|----------------|
| `OPENAI_API_KEY` | https://platform.openai.com |
| `LANGCHAIN_API_KEY` | https://smith.langchain.com → Settings → API Keys |
| `NEWS_API_KEY` | https://newsapi.org (free tier, 100 req/day) |

### 5. Run the app

```bash
streamlit run app.py
```

---

## LangSmith Setup

1. Sign up at https://smith.langchain.com
2. Create a new project named `stock-market-agent`
3. Copy your API key to `.env`
4. Run the app — every agent query will appear as a named trace in LangSmith

Every run is tagged with:
- **Project:** `stock-market-agent`
- **Run name:** `stock-market-agent`
- **Tags:** `["stock-agent"]`
- **Metadata:** the original user query

---

## Evaluations

The eval suite covers 15 golden examples across all tool combinations.

### Evaluator Types

**1. Trajectory Eval** — Did the agent call the right tools?

```
Score 1.0 → exact match
Score 0.8 → agent used all expected tools + extras (over-tooling)
Score 0.6 → agent used a subset of expected tools (missing tools)
Score 0.4 → partial overlap
Score 0.0 → completely wrong tools
```

**2. Output Eval** — Is the numeric answer correct? (calculator queries)

```
Checks if the expected number appears in the response within ±0.5 tolerance
```

**3. LLM-as-Judge** — GPT-4o scores responses on a 1–5 rubric

```
Relevance   — Does the response answer the question?
Grounding   — Is it backed by real tool data?
Conciseness — Is it clear and to the point?
Accuracy    — Is the financial reasoning sound?
```

Run evals from the **Evals** page in the Streamlit UI, or directly:

```python
from agent.evaluators import run_full_eval
results = run_full_eval()
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o |
| Agent framework | LangGraph |
| Tracing & evals | LangSmith |
| Stock data | yfinance (free, no API key) |
| News data | NewsAPI (free tier) |
| UI | Streamlit |
| Environment | python-dotenv |

---

## Course Outline

1. **Build the tools** — Define 4 LangChain tools with clear descriptions
2. **Build the agent** — LangGraph ReAct loop with conditional tool routing
3. **Add LangSmith tracing** — Named runs, tags, metadata
4. **Demo the UI** — Chat page, tool cards, then open LangSmith directly for traces
5. **Introduce failure modes** — Wrong tool, missing tool, over-tooling
6. **Write evals** — Trajectory, output, and LLM-as-judge
7. **Run evals in LangSmith** — Dashboard, scoring, iteration

---

## License

MIT
