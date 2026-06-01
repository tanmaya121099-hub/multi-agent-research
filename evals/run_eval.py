"""
Evaluation harness for the multi-agent research pipeline.

Runs a set of reference queries through the full agent pipeline and scores
each output against expected criteria: source count, critic iterations, latency.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --queries evals/eval_queries.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_QUERIES = [
    "What are the trade-offs between RAG and fine-tuning for production LLMs?",
    "How does LangGraph differ from vanilla LangChain for agent orchestration?",
    "What metrics matter most when evaluating a RAG pipeline?",
]


@dataclass
class EvalResult:
    query: str
    latency_ms: float
    source_count: int
    critic_iterations: int
    has_report: bool
    passed: bool


def score_result(query: str, result: dict, latency_ms: float) -> EvalResult:
    has_report = bool(result.get("report", "").strip())
    source_count = len(result.get("sources", []))
    critic_iter = result.get("critic_iterations", 0)
    passed = has_report and source_count > 0

    return EvalResult(
        query=query,
        latency_ms=round(latency_ms, 1),
        source_count=source_count,
        critic_iterations=critic_iter,
        has_report=has_report,
        passed=passed,
    )


def run(queries: list[str], output_path: str = "evals/eval_results.json") -> None:
    from src.orchestration.graph import ResearchGraph

    graph = ResearchGraph()
    results = []

    for query in queries:
        start = time.time()
        result = graph.run(query)
        latency_ms = (time.time() - start) * 1000
        eval_result = score_result(query, result, latency_ms)
        results.append(asdict(eval_result))
        status = "PASS" if eval_result.passed else "FAIL"
        print(f"[{status}] {query[:60]} — {eval_result.latency_ms}ms, {eval_result.source_count} sources")

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    summary = {"pass_rate": round(pass_rate, 3), "n": len(results), "results": results}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(summary, indent=2))
    print(f"\nPass rate: {pass_rate:.1%} | Report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=None, help="Path to JSON array of query strings")
    parser.add_argument("--out", default="evals/eval_results.json")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries:
        queries = json.loads(Path(args.queries).read_text())

    run(queries, args.out)
