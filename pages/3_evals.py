import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Evals — Stock Agent", page_icon="🧪", layout="wide")
st.title("🧪 Agent Evaluations")
st.caption("Run evaluations against the golden dataset and see LLM-as-judge scores.")

from agent.evaluators import GOLDEN_DATASET

st.subheader("Golden Dataset")
st.markdown(f"**{len(GOLDEN_DATASET)} examples** covering 1-tool, 2-tool, 3-tool, and 4-tool scenarios.")

with st.expander("View Golden Dataset"):
    df_golden = pd.DataFrame([
        {
            "Query": ex["query"],
            "Expected Tools": ", ".join(ex["expected_tools"]),
            "Tool Count": len(ex["expected_tools"]),
            "Has Expected Answer": "expected_answer" in ex,
        }
        for ex in GOLDEN_DATASET
    ])
    st.dataframe(df_golden, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Run Evaluations")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    Runs 3 evaluators on every example:
    - **Trajectory** — Did the agent pick the right tools?
    - **Output** — Is the numeric answer correct? (calculator queries only)
    - **LLM-as-Judge** — GPT-4o scores response on Relevance, Grounding, Conciseness, Accuracy
    """)
with col2:
    run_all = st.button("▶ Run All Evals", use_container_width=True, type="primary")
    run_single = st.button("▶ Run Single Example", use_container_width=True)

if run_all or run_single:
    from agent.evaluators import run_full_eval, GOLDEN_DATASET, run_agent, trajectory_evaluator, output_evaluator, llm_as_judge

    if run_single:
        idx = st.number_input("Example index (0–14)", min_value=0, max_value=len(GOLDEN_DATASET) - 1, value=0)
        examples_to_run = [GOLDEN_DATASET[int(idx)]]
    else:
        examples_to_run = GOLDEN_DATASET

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    for i, example in enumerate(examples_to_run):
        query = example["query"]
        status_text.text(f"Running ({i+1}/{len(examples_to_run)}): {query[:60]}...")
        progress_bar.progress((i + 1) / len(examples_to_run))

        try:
            from agent.graph import run_agent
            from agent.evaluators import trajectory_evaluator, output_evaluator, llm_as_judge

            run_result = run_agent(query)
            response = run_result["response"]
            expected_tools = example["expected_tools"]
            expected_answer = example.get("expected_answer")

            traj = trajectory_evaluator(run_result, expected_tools)
            out = output_evaluator(response, expected_answer) if expected_answer else None
            judge = llm_as_judge(query, response)

            results.append({
                "Query": query,
                "Expected Tools": ", ".join(expected_tools),
                "Actual Tools": ", ".join(traj["actual"]),
                "Trajectory": traj["label"],
                "Traj Score": traj["score"],
                "Output Score": out["score"] if out else "—",
                "Relevance": judge.get("relevance", "—"),
                "Grounding": judge.get("grounding", "—"),
                "Conciseness": judge.get("conciseness", "—"),
                "Accuracy": judge.get("accuracy", "—"),
                "Overall": judge.get("overall", "—"),
                "Judge Reasoning": judge.get("reasoning", "—"),
            })
        except Exception as e:
            results.append({
                "Query": query,
                "Expected Tools": ", ".join(example["expected_tools"]),
                "Actual Tools": "ERROR",
                "Trajectory": "error",
                "Traj Score": 0.0,
                "Output Score": "—",
                "Relevance": "—", "Grounding": "—", "Conciseness": "—", "Accuracy": "—",
                "Overall": "—",
                "Judge Reasoning": str(e),
            })

    status_text.text("Done!")
    st.session_state["eval_results"] = results

if "eval_results" in st.session_state:
    results = st.session_state["eval_results"]
    df = pd.DataFrame(results)

    st.divider()
    st.subheader("Results Summary")

    numeric_traj = [r["Traj Score"] for r in results if isinstance(r["Traj Score"], float)]
    numeric_overall = [r["Overall"] for r in results if isinstance(r["Overall"], float)]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Trajectory Score", f"{sum(numeric_traj)/len(numeric_traj):.2f}" if numeric_traj else "—")
    col_b.metric("Avg LLM Judge Score", f"{sum(numeric_overall)/len(numeric_overall):.2f}" if numeric_overall else "—")
    col_c.metric("Examples Run", len(results))

    st.subheader("Detailed Results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Failure Analysis")
    failures = [r for r in results if isinstance(r["Traj Score"], float) and r["Traj Score"] < 1.0]
    if failures:
        st.markdown(f"**{len(failures)} trajectory mismatches:**")
        for f in failures:
            with st.expander(f"❌ {f['Query'][:60]}..."):
                st.markdown(f"- **Expected:** `{f['Expected Tools']}`")
                st.markdown(f"- **Actual:** `{f['Actual Tools']}`")
                st.markdown(f"- **Label:** `{f['Trajectory']}`")
                st.markdown(f"- **Judge Reasoning:** {f['Judge Reasoning']}")
    else:
        st.success("All trajectory evals passed!")
