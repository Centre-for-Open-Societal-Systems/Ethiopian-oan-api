"""
RAG evaluation with DeepEval — answer quality + retrieval, in ONE run.

Sibling to `tests/rag_ragas_eval.py`. We evaluated RAG two ways:
  - `rag_ragas_eval.py`  — RAGAS metrics, must run in an isolated venv (ragas pins openai<2).
  - `rag_deepeval_eval.py` (this file) — DeepEval metrics, runs in the **app environment** (no venv;
    DeepEval coexists with the app's openai 2.x). DeepEval covers BOTH layers, so one run replaces the
    two-tool setup:
      * Answer quality  — Answer Relevancy, Answer Correctness (meaning-based, LLM-judged).
      * Retrieval       — Contextual Recall, Contextual Precision, Faithfulness.

Retrieval metrics need the chunks the app actually retrieved; those are surfaced by the
EVAL_EXPOSE_CONTEXTS flag (see app changes) and read back from the chat response.

Prereq: DeepEval installed in the container (`docker exec oan_app pip install "deepeval>=4.0.5"` —
it's dev/eval-only and gets wiped on a container rebuild). RAG must actually return chunks (cosdata
client patched). App up with EVAL_EXPOSE_CONTEXTS=true and vLLM up.

Run (app env, no venv):
    docker exec oan_app python3 /app/tests/rag_deepeval_eval.py

Options (env vars):
    EVAL_INTENTS=unknown   Intent filter (default: unknown — the LLM/RAG path)
    EVAL_LIMIT=0           First N rows only (0 = all)
    RAGEVAL_JUDGE=google/gemini-2.5-flash   single OpenRouter judge (cheap, multilingual)
    BENCH_CSV=...          Override benchmark CSV
    OPENROUTER_API_KEY=... Required for the judge (env or /app/.env)

CLI:
    (no args)              collect from app, then score
    --collect-only         collect + cache, no scoring
    --score-from FILE.csv  skip the app; score a previously collected CSV
    --selftest             score one built-in row (validates judge + metrics; no app)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow importing the sibling harness (reuse its app-call + judge wrapper).
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from eval_prompt_quality import call_app, _make_openrouter_judge  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────────
EVAL_INTENTS_FILTER = [i.strip() for i in os.getenv("EVAL_INTENTS", "unknown").split(",") if i.strip()]
EVAL_LIMIT          = int(os.getenv("EVAL_LIMIT", "0"))
JUDGE_MODEL         = os.getenv("RAGEVAL_JUDGE", "google/gemini-2.5-flash")

_here       = Path(__file__).parent
CSV_PATH    = Path(os.getenv("BENCH_CSV", str(_here.parent / "results" / "benchmark_samples.csv")))
RESULTS_DIR = _here.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Metrics ─────────────────────────────────────────────────────────────────────
# Returns specs: (display_name, metric, needs_context). Contextual/Faithfulness metrics
# require retrieval_context and are skipped on rows that retrieved nothing.

def build_metrics(judge):
    from deepeval.metrics import (
        GEval, ContextualRecallMetric, ContextualPrecisionMetric, FaithfulnessMetric,
    )
    try:
        from deepeval.test_case import SingleTurnParams as EvalParams
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams as EvalParams  # older versions

    kw = {"model": judge} if judge else {}
    # async_mode=False → fully synchronous measure() (avoids DeepEval's asyncio batch
    # machinery, which mis-handles our blocking OpenRouter judge in a tight loop).
    common = {"async_mode": False, **kw}

    answer = [
        ("Answer Relevancy", GEval(
            name="Answer Relevancy",
            criteria=(
                "Evaluate whether the response directly and completely answers the question asked. "
                "High score (close to 1): addresses the exact question with relevant information. "
                "Low score (close to 0): off-topic, ignores the question, or only partially answers it."
            ),
            evaluation_steps=[
                "Identify the specific information requested in the input.",
                "Check if the actual output directly provides that information.",
                "If it gives the exact requested information, assign a high score (0.8-1.0).",
                "If partially relevant or incomplete, assign a medium score (0.3-0.7).",
                "If off-topic, an error message, or failing to address the question, assign 0.0-0.2.",
            ],
            evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT],
            threshold=0.5, **common,
        ), False),
        ("Answer Correctness", GEval(
            name="Answer Correctness",
            criteria=(
                "Evaluate whether the actual response conveys the same key facts as the expected "
                "answer. High score: same core information (values, dates, names, agronomic advice). "
                "Low score: contradicts or omits critical facts."
            ),
            evaluation_steps=[
                "Read the expected answer and identify its key facts.",
                "Check whether the actual output contains those same key facts.",
                "If all key facts are present and consistent, assign a high score (0.8-1.0).",
                "If some facts are missing or only partially correct, assign 0.3-0.7.",
                "If it contradicts or entirely omits the expected facts, assign 0.0-0.2.",
            ],
            evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT, EvalParams.EXPECTED_OUTPUT],
            threshold=0.5, **common,
        ), False),
    ]
    retrieval = [
        ("Contextual Recall",    ContextualRecallMetric(threshold=0.5, **common), True),
        ("Contextual Precision", ContextualPrecisionMetric(threshold=0.5, **common), True),
        ("Faithfulness",         FaithfulnessMetric(threshold=0.5, **common), True),
    ]
    return answer + retrieval


METRIC_NAMES = [name for name, _, _ in build_metrics(None)]


# ── OpenRouter key (reused pattern) ───────────────────────────────────────────────

def _load_openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_path = _here.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


# ── Phase 1: collect from the live app ────────────────────────────────────────────

def collect(rows: list[dict]) -> list[dict]:
    run_id = os.urandom(3).hex()
    records = []
    print(f"\n  Collecting from app — {len(rows)} queries ...")
    for i, row in enumerate(rows, 1):
        query    = row["query"]
        lang     = row["lang"]
        expected = row.get("expected_answer", "").strip()
        try:
            answer, _path, _ms, captures = call_app(query, lang, f"deval-{run_id}-{i}")
        except Exception as e:
            answer, captures = f"ERROR: {e}", []
        contexts = [c for cap in (captures or []) for c in cap.get("contexts", [])]
        records.append({
            "lang": lang, "query": query, "reference": expected,
            "answer": answer, "contexts": contexts,
        })
        print(f"  [{i:3d}/{len(rows)}] 📄{len(contexts):<2} {query[:60]}")
    return records


def save_collected(records: list[dict], ts: str) -> Path:
    out = RESULTS_DIR / f"rag_deepeval_collected_{ts}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "query", "reference", "answer", "n_contexts", "contexts_json"])
        w.writeheader()
        for r in records:
            w.writerow({
                "lang": r["lang"], "query": r["query"], "reference": r["reference"],
                "answer": r["answer"], "n_contexts": len(r["contexts"]),
                "contexts_json": json.dumps(r["contexts"], ensure_ascii=False),
            })
    print(f"  Collected data → {out}")
    return out


def load_collected(path: str) -> list[dict]:
    records = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        records.append({
            "lang": r["lang"], "query": r["query"], "reference": r["reference"],
            "answer": r["answer"], "contexts": json.loads(r.get("contexts_json") or "[]"),
        })
    return records


# ── Phase 2: score with DeepEval ──────────────────────────────────────────────────

def score(records: list[dict]) -> None:
    from deepeval.test_case import LLMTestCase

    if not _load_openrouter_key():
        print("ERROR: OPENROUTER_API_KEY not set (env or /app/.env).")
        sys.exit(1)

    judge = _make_openrouter_judge(JUDGE_MODEL)
    specs = build_metrics(judge)

    n_ctx = sum(1 for r in records if r["contexts"])
    print(f"\n  Scoring {len(records)} rows with DeepEval (judge: {JUDGE_MODEL}) ...")
    print(f"  ({len(records) - n_ctx} rows had no retrieved contexts — retrieval metrics skipped for them.)\n")

    per_row: list[dict] = []
    for i, r in enumerate(records, 1):
        ctx = r["contexts"]
        tc = LLMTestCase(
            input=r["query"],
            actual_output=r["answer"],
            expected_output=r["reference"] or None,
            retrieval_context=ctx or None,
        )
        scores: dict = {"lang": r["lang"], "query": r["query"], "n_contexts": len(ctx)}
        for name, metric, needs_ctx in specs:
            if needs_ctx and not ctx:
                scores[name] = None
                continue
            try:
                metric.measure(tc)
                scores[name] = round(metric.score, 3) if metric.score is not None else None
            except Exception as e:
                scores[name] = None
                scores[f"{name}__err"] = str(e)[:120]
        per_row.append(scores)
        line = "  ".join(f"{n[:12]}:{('%.2f' % scores[n]) if scores.get(n) is not None else '  -'}"
                          for n in METRIC_NAMES)
        print(f"  [{i:3d}/{len(records)}] 📄{len(ctx):<2} {line}")

    _report(per_row)


def _report(per_row: list[dict]) -> None:
    def _mean(rows, name):
        vals = [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        return round(statistics.mean(vals), 3) if vals else None

    fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    en = [r for r in per_row if r["lang"] == "en"]
    am = [r for r in per_row if r["lang"] == "am"]

    print("\n" + "=" * 80)
    print("  DeepEval RAG BASELINE — current Cosdata retriever (no reranker)")
    print("=" * 80)
    print(f"  {'metric':<24}{'overall':>10}{'en':>10}{'am':>10}")
    print("  " + "─" * 54)
    for name in METRIC_NAMES:
        print(f"  {name:<24}{fmt(_mean(per_row, name)):>10}{fmt(_mean(en, name)):>10}{fmt(_mean(am, name)):>10}")
    print("=" * 80)
    no_ctx = sum(1 for r in per_row if r["n_contexts"] == 0)
    print(f"  Rows: {len(per_row)} (en {len(en)} / am {len(am)})  |  no-retrieval: {no_ctx}")

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"rag_deepeval_baseline_{ts}.csv"
    fields = ["lang", "query", "n_contexts"] + METRIC_NAMES
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_row:
            w.writerow(r)
    print(f"  Saved → {out}")


def selftest() -> None:
    print("  Self-test: scoring one built-in row (validates judge + metrics, no app) ...")
    score([{
        "lang": "en",
        "query": "How do I control fall armyworm in maize?",
        "reference": "Control fall armyworm by scouting early and applying recommended insecticides "
                     "(e.g. emamectin benzoate); handpicking egg masses helps.",
        "answer": "Scout early and apply recommended insecticides like emamectin benzoate; "
                  "handpicking egg masses also helps.",
        "contexts": [
            "Fall armyworm in maize is controlled by early scouting and applying recommended "
            "insecticides such as emamectin benzoate. Handpicking egg masses also helps.",
            "Maize requires weeding at the 3-leaf stage.",
        ],
    }])


# ── Entry point ───────────────────────────────────────────────────────────────────

def _read_rows() -> list[dict]:
    if not CSV_PATH.exists():
        print(f"ERROR: benchmark CSV not found at {CSV_PATH}")
        sys.exit(1)
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if EVAL_INTENTS_FILTER:
        rows = [r for r in rows if r.get("expected_intent", "") in EVAL_INTENTS_FILTER]
    if EVAL_LIMIT:
        rows = rows[:EVAL_LIMIT]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG evaluation with DeepEval (answer + retrieval).")
    ap.add_argument("--collect-only", action="store_true", help="collect + cache, no scoring")
    ap.add_argument("--score-from", metavar="CSV", help="score a previously collected CSV (skip app)")
    ap.add_argument("--selftest", action="store_true", help="score one built-in row (no app)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.score_from:
        score(load_collected(args.score_from))
        return

    rows = _read_rows()
    print(f"  Benchmark: {CSV_PATH.name} | intents={EVAL_INTENTS_FILTER or 'all'} "
          f"| limit={EVAL_LIMIT or 'all'} | rows={len(rows)} | judge={JUDGE_MODEL}")
    ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = collect(rows)
    save_collected(records, ts)
    if not args.collect_only:
        score(records)


if __name__ == "__main__":
    main()
