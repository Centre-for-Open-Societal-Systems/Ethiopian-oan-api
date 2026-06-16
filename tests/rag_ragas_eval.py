"""
RAGAS retrieval evaluation — baseline the CURRENT Cosdata retriever.

This is the "test current RAG" step of the RAG-correctness epic (.doc/rag_evaluation.md).
v1 answer correctness is retrieval-bound; RAGAS gives the retrieval-level metric the
answer-only deepeval harness lacks: **Context Recall** (did we retrieve the facts the
reference answer needs?).

Two phases, deliberately decoupled:

  1. COLLECT — POST each benchmark query to the live app (`POST /api/chat/`). The app runs
     its real pipeline (intent router → LLM → search_documents → answer); with
     EVAL_EXPOSE_CONTEXTS=true it returns the retrieved chunks in `metrics.contexts`.
     Needs the app + vLLM up. Cached to results/ragas_collected_<ts>.csv.
  2. SCORE — feed (query, retrieved_contexts, answer, reference) to RAGAS. Uses a single
     OpenRouter judge; needs only OpenRouter (NOT vLLM), so it can re-run offline.

ISOLATION: run from the dedicated venv (see tests/setup_ragas_venv.sh) — ragas pins
openai<2 and would break the app if installed into the app env:
    docker exec oan_app /opt/ragas-venv/bin/python /app/tests/rag_ragas_eval.py

Options (env vars):
    EVAL_INTENTS=unknown     Intent filter (default: unknown — the LLM/RAG path)
    EVAL_LIMIT=0             Evaluate only first N rows (0 = all)
    RAGAS_JUDGE=google/gemini-2.5-flash   OpenRouter judge model id
    BENCH_CSV=...            Override benchmark CSV
    OPENROUTER_API_KEY=...   Required for the judge (read from env or /app/.env)

CLI:
    (no args)                collect from app, then score
    --collect-only           collect + cache, no scoring (works without OpenRouter)
    --score-from FILE.csv    skip the app; score a previously collected CSV
    --selftest               score one built-in sample (validates venv + judge; no app)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────────
BASE                = os.getenv("RAGAS_APP_BASE", "http://localhost:8000/api")
EVAL_INTENTS_FILTER = [i.strip() for i in os.getenv("EVAL_INTENTS", "unknown").split(",") if i.strip()]
EVAL_LIMIT          = int(os.getenv("EVAL_LIMIT", "0"))
JUDGE_MODEL         = os.getenv("RAGAS_JUDGE", "google/gemini-2.5-flash")

_here       = Path(__file__).parent
CSV_PATH    = Path(os.getenv("BENCH_CSV", str(_here.parent / "results" / "benchmark_samples.csv")))
RESULTS_DIR = _here.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_openrouter_key() -> str:
    """OPENROUTER_API_KEY from env, falling back to a light parse of /app/.env."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_path = _here.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


# ── Phase 1: collect from the live app ──────────────────────────────────────────

def call_app(query: str, lang: str, session_id: str) -> tuple[str, list[str]]:
    """POST to /api/chat/; return (answer, flattened retrieved_contexts)."""
    payload = json.dumps({
        "query": query, "session_id": session_id,
        "source_lang": lang, "target_lang": lang,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/", data=payload, headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    lines = [ln for ln in resp.read().decode().splitlines() if ln.strip()]
    r = json.loads(lines[-1]) if lines else {}
    metrics  = r.get("metrics", {})
    captures = metrics.get("contexts", []) or []
    contexts = [c for cap in captures for c in cap.get("contexts", [])]
    return r.get("response", ""), contexts


def collect(rows: list[dict]) -> list[dict]:
    run_id = os.urandom(3).hex()
    records = []
    print(f"\n  Collecting from app ({BASE}) — {len(rows)} queries ...")
    for i, row in enumerate(rows, 1):
        query    = row["query"]
        lang     = row["lang"]
        expected = row.get("expected_answer", "").strip()
        try:
            answer, contexts = call_app(query, lang, f"ragas-{run_id}-{i}")
        except Exception as e:
            answer, contexts = f"ERROR: {e}", []
        records.append({
            "lang": lang, "query": query, "reference": expected,
            "answer": answer, "contexts": contexts,
        })
        print(f"  [{i:3d}/{len(rows)}] 📄{len(contexts):<2} {query[:60]}")
    return records


def save_collected(records: list[dict], ts: str) -> Path:
    out = RESULTS_DIR / f"ragas_collected_{ts}.csv"
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


# ── Phase 2: score with RAGAS ────────────────────────────────────────────────────

def _build_judge():
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    key = _load_openrouter_key()
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set (env or /app/.env).")
        sys.exit(1)
    return LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        temperature=0.0,
        default_headers={"HTTP-Referer": "https://oan.app"},
    ))


def score(records: list[dict]) -> None:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        LLMContextRecall, LLMContextPrecisionWithReference, Faithfulness, FactualCorrectness,
    )

    # RAGAS retrieval/faithfulness metrics need non-empty contexts. Rows where the LLM
    # answered without calling search_documents are reported separately (no retrieval).
    scorable   = [r for r in records if r["contexts"]]
    no_context = [r for r in records if not r["contexts"]]
    if not scorable:
        print("  No rows with retrieved contexts to score.")
        return

    judge   = _build_judge()
    metrics = [
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        Faithfulness(),
        FactualCorrectness(),
    ]
    samples = [
        SingleTurnSample(
            user_input=r["query"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["reference"],
        )
        for r in scorable
    ]

    print(f"\n  Scoring {len(scorable)} rows with RAGAS (judge: {JUDGE_MODEL}) ...")
    print(f"  ({len(no_context)} rows had no retrieved contexts — excluded from retrieval metrics.)")
    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics, llm=judge)
    df = result.to_pandas()
    df.insert(0, "lang", [r["lang"] for r in scorable])

    metric_cols = [c for c in df.columns if c not in ("lang", "user_input", "retrieved_contexts",
                                                       "response", "reference")]

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"ragas_baseline_{ts}.csv"
    df.to_csv(out, index=False)

    def _mean(vals):
        nums = [v for v in vals if isinstance(v, (int, float)) and v == v]  # drop NaN
        return round(statistics.mean(nums), 3) if nums else None

    print("\n" + "=" * 78)
    print("  RAGAS BASELINE — current Cosdata retriever (no reranker)")
    print("=" * 78)
    header = f"  {'metric':<40}{'overall':>9}{'en':>9}{'am':>9}"
    print(header)
    print("  " + "─" * 67)
    for col in metric_cols:
        overall = _mean(df[col].tolist())
        en = _mean(df[df["lang"] == "en"][col].tolist())
        am = _mean(df[df["lang"] == "am"][col].tolist())
        fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
        print(f"  {col:<40}{fmt(overall):>9}{fmt(en):>9}{fmt(am):>9}")
    print("=" * 78)
    print(f"  Scored {len(scorable)} rows ({len(no_context)} no-retrieval excluded).")
    print(f"  Saved → {out}")


def selftest() -> None:
    print("  Self-test: scoring one built-in sample (validates venv + judge, no app) ...")
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
    ap = argparse.ArgumentParser(description="RAGAS retrieval eval for the Cosdata retriever.")
    ap.add_argument("--collect-only", action="store_true", help="collect + cache, no scoring")
    ap.add_argument("--score-from", metavar="CSV", help="score a previously collected CSV (skip app)")
    ap.add_argument("--selftest", action="store_true", help="score one built-in sample (no app)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.score_from:
        score(load_collected(args.score_from))
        return

    rows = _read_rows()
    print(f"  Benchmark: {CSV_PATH.name} | intents={EVAL_INTENTS_FILTER or 'all'} "
          f"| limit={EVAL_LIMIT or 'all'} | rows={len(rows)}")
    ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = collect(rows)
    save_collected(records, ts)
    if not args.collect_only:
        score(records)


if __name__ == "__main__":
    main()
