"""
Evaluation script for the study assistant agent.
Measures hit_rate@k, MRR, answer correctness, refusal rate, and latency.

Run (from src/):
  Tuning set:   python3 -m evaluation.eval
  Held-out set: python3 -m evaluation.eval --held-out

Web search is disabled for every agent in this file, so the eval measures
answers from the lecture notes only. The real app is unchanged.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated

# must be set before any chromadb import
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

if os.getenv("OPENAI_API_KEY") and not os.getenv("CHROMA_OPENAI_API_KEY"):
    os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print(f"CHROMA key loaded: {bool(os.getenv('CHROMA_OPENAI_API_KEY'))}")

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.study_agent import ask
from evaluation.questions import EVAL_QUESTIONS
from ingestion.vector_store import query_chroma
from retrieval.tools import semantic_search


# ── helpers ────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase and normalize hyphens/underscores/slashes for matching."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[-_/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_match(keyword: str, text: str) -> bool:
    """
    Whole-word match (simple plurals/inflections allowed).
    Trailing * means prefix match: 'speciali*' matches 'specialized', 'specialise'.
    """
    text_norm = normalize(text)
    kw = normalize(keyword)
    if kw.endswith("*"):
        return re.search(r"\b" + re.escape(kw[:-1]), text_norm) is not None
    return re.search(r"\b" + re.escape(kw) + r"(?:s|es|d|ed|ing)?\b", text_norm) is not None


def check_retrieval(
    question: str,
    expected_keywords: list[str],
    source_pages: list[int],
    k: int = 5,
) -> tuple[bool, int]:
    results = query_chroma(question, n_results=k)

    if source_pages:
        retrieved_pages = [r["page_num"] for r in results]
        print(f"    gold={source_pages} retrieved={retrieved_pages}")
        for rank, result in enumerate(results, 1):
            if int(result["page_num"]) in [int(p) for p in source_pages]:
                return True, rank
        return False, 0

    for rank, result in enumerate(results, 1):
        if any(keyword_match(kw, result["text"]) for kw in expected_keywords):
            return True, rank

    return False, 0


def check_answer(
    answer: str,
    expected_keywords: list[str],
    min_required: int,
) -> bool:
    """Check if answer contains at least min_required keywords."""
    found = sum(1 for kw in expected_keywords if keyword_match(kw, answer))
    return found >= min_required


NEGATORS = ["not ", "no ", "isn't", "aren't", "doesn't", "never",
            "rather than", "instead of", "incorrect", "there is no"]


def check_false_premise(answer: str, q: dict) -> bool:
    """PASS if the answer corrects the premise and doesn't accept it.
    A hallucination phrase only counts if no negation appears just before it."""
    a = normalize(answer)
    has_correction = any(normalize(p) in a for p in q.get("correction_phrases", []))

    accepted = False
    for p in q.get("hallucination_phrases", []):
        for m in re.finditer(re.escape(normalize(p)), a):
            window = a[max(0, m.start() - 40): m.start()]
            if not any(n in window for n in NEGATORS):
                accepted = True
    return has_correction and not accepted
'''
def check_false_premise(answer: str, q: dict) -> bool:
    """
    PASS if the answer contains a correction phrase AND does NOT accept the premise
    (e.g. list a sixth force). Negated mentions like "not six forces" are ignored.
    """
    answer_norm = normalize(answer)
    has_correction = any(
        normalize(p) in answer_norm for p in q.get("correction_phrases", [])
    )

    cleaned = answer_norm
    for neg in ("not six forces", "not six", "rather than six",
                "no sixth force", "no sixth", "not a sixth"):
        cleaned = cleaned.replace(neg, "")
    
    for neg in ("not six forces", "not six", "rather than six",
                "no sixth force", "no sixth", "not a sixth",
                "not seven forces", "not seven", "rather than seven",
                "no seventh force", "not four questions", "not four",
                "rather than four"):
        cleaned = cleaned.replace(neg, "")

    has_hallucination = any(
        normalize(p) in cleaned for p in q.get("hallucination_phrases", [])
    )
    return has_correction and not has_hallucination
'''

def check_refusal(answer: str) -> bool:
    """Check if the agent correctly refused an unanswerable question."""
    refusal_signals = [
        "not in the notes",
        "not covered",
        "doesn't cover",
        "does not cover",
        "no information",
        "couldn't find",
        "could not find",
        "isn't covered",
        "not mentioned",
        "does not contain",
        "doesn't contain",
        "not available",
        "cannot find",
        "can't find",
        "not provided",
        "outside the scope",
        "not discussed",
        "no mention",
        "not addressed",
        "not included",
        "not found",
    ]
    answer_norm = normalize(answer)
    return any(normalize(s) in answer_norm for s in refusal_signals)


# ── no-web agent (used for ALL eval questions) ─────────────────────────────────

def build_no_web_agent():
    """Agent with web search disabled, so answers come from the notes only."""

    NO_WEB_TOOLS = [semantic_search]

    NO_WEB_PROMPT = """You are a study assistant for a university course on
Markets and Competitive Strategy (JRE410). ALWAYS call semantic_search first.
Answer ONLY from the retrieved lecture notes and cite the page number.
- If the first search doesn't fully answer the question, search again with
  different wording (for example the specific terms or slide topic) before
  giving up. Combine what you find across searches.
- If the retrieved text contains the answer, give it. Do not refuse.
- If the specific fact asked for is not in the notes (even if the topic is),
  say 'This information is not in the notes.'
- Questions may contain wrong assumptions (a wrong number, or a model or
  force that doesn't exist). Compare every number, name and claim in the
  question with the retrieved notes. If the notes disagree, do NOT just
  refuse: start with "Correction:" and state what the notes actually say.
- Use only facts that appear in the retrieved notes. Do not add reasoning or
  examples from general knowledge.
Do not search the web."""


    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: AgentState) -> AgentState:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        llm_with_tools = llm.bind_tools(NO_WEB_TOOLS)
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=NO_WEB_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(NO_WEB_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


# ── main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(held_out: bool = False) -> dict:
    """
    held_out=False -> tuning set (questions without the held_out flag)
    held_out=True  -> held-out set (questions with held_out=True)
    Report both separately. Never tune on the held-out questions.
    """
    if held_out:
        questions = [q for q in EVAL_QUESTIONS if q.get("held_out")]
        label = "HELD-OUT SET"
    else:
        questions = [q for q in EVAL_QUESTIONS if not q.get("held_out")]
        label = "TUNING SET"

    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    false_premise_qs = [q for q in answerable if q.get("correction_check")]
    regular_qs = [q for q in answerable if not q.get("correction_check")]

    print("Building agent (web search disabled)...")
    agent = build_no_web_agent()

    print(f"\nRunning {label}")
    print(f"  Regular answerable: {len(regular_qs)}")
    print(f"  False premise:      {len(false_premise_qs)}")
    print(f"  Unanswerable:       {len(unanswerable)}")
    print("=" * 60)

    hits = 0
    reciprocal_ranks = []
    correct = 0
    latencies = []
    fp_correct = 0
    refusals = 0
    failed_ids = []

    # ── regular answerable questions ───────────────────────────────────
    for q in regular_qs:
        print(f"\nQ{q['id']} [{q['category']}]: {q['question'][:55]}...")

        hit, rank = check_retrieval(
            q["question"], q["expected_keywords"], q["source_pages"], k=5
        )
        if hit:
            hits += 1
            reciprocal_ranks.append(1 / rank)
            retrieval_str = f"hit@{rank}"
        else:
            reciprocal_ranks.append(0)
            retrieval_str = "miss"

        start = time.time()
        answer = ask(q["question"], agent, thread_id=f"eval_{q['id']}")
        latency = time.time() - start
        latencies.append(latency)

        passed = check_answer(answer, q["expected_keywords"], q["min_keywords_required"])
        correct += int(passed)
        if not passed:
            failed_ids.append(q["id"])

        status = "✅" if passed else "❌"
        print(f"  Retrieval: {retrieval_str} | Answer: {status} | {latency:.1f}s")
        if not passed:
            print(f"  Answer: {answer[:300]}...")

    # ── false premise questions ────────────────────────────────────────
    for q in false_premise_qs:
        print(f"\nQ{q['id']} [false_premise]: {q['question'][:55]}...")

        start = time.time()
        answer = ask(q["question"], agent, thread_id=f"eval_{q['id']}")
        latency = time.time() - start

        passed = check_false_premise(answer, q)
        fp_correct += int(passed)
        if not passed:
            failed_ids.append(q["id"])

        status = "✅ Corrected" if passed else "❌ Accepted premise"
        print(f"  {status} | {latency:.1f}s")
        if not passed:
            print(f"  Answer: {answer[:300]}...")

    # ── unanswerable questions ─────────────────────────────────────────
    print("\n--- Unanswerable questions (web search disabled) ---")
    for q in unanswerable:
        print(f"\nQ{q['id']}: {q['question'][:55]}...")
        answer = ask(q["question"], agent, thread_id=f"eval_u_{q['id']}")
        refused = check_refusal(answer)
        refusals += int(refused)
        if not refused:
            failed_ids.append(q["id"])
        print(f"  {'✅ Refused' if refused else '❌ Hallucinated'}")
        if not refused:
            print(f"  Answer: {answer[:200]}...")

    # ── final metrics ──────────────────────────────────────────────────
    n_regular = len(regular_qs)
    n_fp = len(false_premise_qs)
    n_unans = len(unanswerable)

    hit_rate = hits / n_regular if n_regular else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    acc = correct / n_regular if n_regular else 0
    fp_acc = fp_correct / n_fp if n_fp else 0
    refusal_rate = refusals / n_unans if n_unans else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 60)
    print(f"RESULTS — {label}")
    print("=" * 60)
    print(f"Hit Rate@5:          {hit_rate:.1%}  ({hits}/{n_regular})")
    print(f"MRR:                 {mrr:.3f}")
    print(f"Answer Accuracy:     {acc:.1%}  ({correct}/{n_regular})")
    print(f"False Premise:       {fp_acc:.1%}  ({fp_correct}/{n_fp})")
    print(f"Refusal Rate:        {refusal_rate:.1%}  ({refusals}/{n_unans})")
    print(f"Avg Latency:         {avg_latency:.1f}s")
    print(f"Failed ids:          {failed_ids}")
    print("=" * 60)
    print("\nResume line:")
    print(
        f'  "Evaluated on {len(questions)} questions; '
        f"hit rate@5 {hit_rate:.0%}, "
        f"answer accuracy {acc:.0%}, "
        f'refusal rate {refusal_rate:.0%}"'
    )

    return {
        "hit_rate_at_5": hit_rate,
        "mrr": mrr,
        "answer_accuracy": acc,
        "false_premise_accuracy": fp_acc,
        "refusal_rate": refusal_rate,
        "avg_latency_seconds": avg_latency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Run held-out set instead of tuning set",
    )
    args = parser.parse_args()
    run_evaluation(held_out=args.held_out)