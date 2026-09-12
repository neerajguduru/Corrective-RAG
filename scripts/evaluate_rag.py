"""Run RAGAS evaluation of the CRAG pipeline against a golden QA set.

Usage:
    python scripts/evaluate_rag.py                      # all questions
    python scripts/evaluate_rag.py --limit 5            # first 5 questions
    python scripts/evaluate_rag.py --output results.csv # custom CSV path

Reads data/eval/golden_qa.json (see that file for the format), runs each
question through the real CRAG graph (Ollama + ChromaDB), then scores
faithfulness / answer_relevancy / context_precision / context_recall with
RAGAS, using the local Ollama LLM as judge.

Requires: Ollama running with the configured model, documents indexed
(scripts/ingest_documents.py), and golden_qa.json filled with real pairs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_DIR

GOLDEN_PATH = BASE_DIR / "data" / "eval" / "golden_qa.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "eval" / "results.csv"


def load_golden() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text())
    samples = [
        s for s in data.get("samples", [])
        if not s["question"].startswith("SAMPLE")
    ]
    if not samples:
        print(
            "ERROR: golden_qa.json contains only SAMPLE entries.\n"
            "Fill it with real question/ground_truth pairs from your PDFs "
            "(see the _comment in the file), then re-run."
        )
        sys.exit(1)
    return samples


def run_pipeline(samples: list[dict]) -> list[dict]:
    """Run each question through the real CRAG graph; collect RAGAS rows."""

    from app.graph.workflow import graph

    rows = []
    for i, sample in enumerate(samples, 1):
        q = sample["question"]
        print(f"\n[{i}/{len(samples)}] {q}")

        try:
            result = graph.invoke(
                {
                    "question": q,
                    "selected_docs": [],
                    "history": [],
                    "rewritten_question": None,
                    "attempts": 0,
                    "web_used": False,
                    "chunk_grades": [],
                }
            )
        except Exception as e:
            print(f"  pipeline failed: {e}")
            rows.append({
                "question": q,
                "ground_truth": sample["ground_truth"],
                "answer": "",
                "contexts": [],
                "web_used": "false",
            })
            continue

        answer = result.get("generation", "")
        contexts = [
            doc.page_content
            for doc in result.get("documents", [])
        ]
        web = result.get("web_used", False)
        print(f"  web_used={web} chunks={len(contexts)} answer={answer[:80]!r}")

        rows.append({
            "question": q,
            "ground_truth": sample["ground_truth"],
            "answer": answer,
            "contexts": contexts,
            "web_used": "true" if web else "false",
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="evaluate first N questions")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples = load_golden()
    if args.limit:
        samples = samples[: args.limit]

    rows = run_pipeline(samples)

    # ---- Score with RAGAS ----
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from app.llm.model import llm as judge_llm
    from app.retrieval.embeddings import embeddings as judge_embeddings

    dataset = Dataset.from_list(rows)

    print("\nScoring with RAGAS (Ollama as judge — this runs multiple "
          "LLM calls per row and may take a while)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        raise_exceptions=False,
    )

    print("\n================ RAGAS scores ================")
    df = result.to_pandas()
    print(df.to_string(max_colwidth=60))

    summary = {
        metric: round(float(score), 3)
        for metric, score in result.items()
        if isinstance(score, (int, float))
    }
    print("\nSummary:", summary)

    targets = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.75,
        "context_precision": 0.75,
        "context_recall": 0.70,
    }
    print("\nTargets:", targets)
    for metric, target in targets.items():
        actual = summary.get(metric)
        if actual is None:
            state = "N/A"
        elif actual >= target:
            state = "PASS"
        else:
            state = "NEEDS WORK"
        print(f"  {metric:20s} {actual if actual is not None else '—'} vs {target}  [{state}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nPer-row results written to {args.output}")


if __name__ == "__main__":
    main()
