"""
OAN Prompt Quality Evaluation — powered by DeepEval.

Captures a quality baseline BEFORE prompt changes, then run again AFTER
to measure improvement. Three independent LLM judges score each response;
scores are averaged into an ensemble to reduce individual model bias.

Install inside the container first:
    docker exec oan_app pip install deepeval

Run baseline (before prompt changes):
    docker exec oan_app python3 /app/tests/eval_prompt_quality.py

Override judges:
    EVAL_JUDGES=openrouter:qwen/qwen3.5-flash-02-23,openrouter:google/gemini-2.5-flash,openrouter:openai/gpt-5.4-mini

Compare two runs:
    python3 /app/tests/eval_prompt_quality.py \\
        --compare results/eval_prompt_v1_<ts>.csv results/eval_prompt_v2_<ts>.csv

Options (env vars):
    RUN_NAME=prompt_v1      Label for this run (default: prompt_v1)

    EVAL_JUDGES             Comma-separated judges. Each entry is:
                              gemma                        — your vLLM, no external key
                              openrouter:<model-id>        — specific model via OpenRouter
                            Default: the 3 judges below

    OPENROUTER_API_KEY      Required for openrouter judges (read from .env)
    EVAL_LIMIT=50           Evaluate only first N queries (default: all)
    EVAL_INTENTS=unknown    Comma-separated intent filter
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BASE                = "http://localhost:8000/api"
RUN_NAME            = os.getenv("RUN_NAME", "prompt_v1")
EVAL_LIMIT          = int(os.getenv("EVAL_LIMIT", "0"))
EVAL_INTENTS_FILTER = [
    i.strip() for i in os.getenv("EVAL_INTENTS", "").split(",") if i.strip()
]

_DEFAULT_JUDGES = (
    "openrouter:qwen/qwen3.5-flash-02-23,"
    "openrouter:google/gemini-2.5-flash,"
    "openrouter:openai/gpt-5.4-mini"
)

_here       = Path(__file__).parent
CSV_PATH    = Path(os.getenv("BENCH_CSV", str(_here.parent / "results" / "benchmark_samples.csv")))
RESULTS_DIR = _here.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Judge parsing ─────────────────────────────────────────────────────────────
# Each EVAL_JUDGES entry → (label, judge_type, model_id)
# label   : short name used in CSV columns  e.g. "qwen3.5-flash-02-23"
# type    : "gemma" | "openrouter"
# model_id: full model path for openrouter  e.g. "qwen/qwen3.5-flash-02-23"

def _parse_judges(raw: str) -> list[tuple[str, str, str]]:
    result = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry == "gemma":
            result.append(("gemma", "gemma", ""))
        elif entry.startswith("openrouter:"):
            model_id = entry[len("openrouter:"):]
            label    = model_id.split("/")[-1]
            result.append((label, "openrouter", model_id))
        else:
            result.append((entry, entry, ""))
    return result

EVAL_JUDGES: list[tuple[str, str, str]] = _parse_judges(
    os.getenv("EVAL_JUDGES", os.getenv("EVAL_JUDGE", _DEFAULT_JUDGES))
)


# ── DeepEval guard ────────────────────────────────────────────────────────────

def _require_deepeval():
    try:
        import deepeval  # noqa: F401
    except ImportError:
        print("ERROR: deepeval is not installed.")
        print("       Run inside the container: pip install deepeval")
        sys.exit(1)


# ── Judge implementations ─────────────────────────────────────────────────────

def _make_gemma_judge():
    from deepeval.models.base_model import DeepEvalBaseLLM

    class GemmaJudge(DeepEvalBaseLLM):
        def __init__(self):
            self._url   = (
                os.getenv("OPENAI_BASE_URL", "http://52.66.116.220:8080").rstrip("/")
                + "/v1/chat/completions"
            )
            self._model = os.getenv("LLM_MODEL_NAME", "gemma-4-26b-a4b")
            self._key   = os.getenv("TRITON_LLM_API_KEY", "")

        def load_model(self): return self

        def _call(self, prompt: str) -> str:
            headers = {"Content-Type": "application/json"}
            if self._key:
                headers["Authorization"] = f"Bearer {self._key}"
            payload = json.dumps({
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(self._url, data=payload, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=90)
                r = json.loads(resp.read().decode())
                return r["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return f"ERROR: {e}"

        def generate(self, prompt: str) -> str: return self._call(prompt)
        async def a_generate(self, prompt: str) -> str:
            import asyncio
            return await asyncio.to_thread(self._call, prompt)
        def get_model_name(self) -> str: return self._model

    return GemmaJudge()


def _make_openrouter_judge(model_id: str):
    from deepeval.models.base_model import DeepEvalBaseLLM

    class OpenRouterJudge(DeepEvalBaseLLM):
        def __init__(self):
            self._url   = "https://openrouter.ai/api/v1/chat/completions"
            self._model = model_id
            self._key   = os.getenv("OPENROUTER_API_KEY", "")

        def load_model(self): return self

        def _call(self, prompt: str) -> str:
            if not self._key:
                return "ERROR: OPENROUTER_API_KEY not set in .env"
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self._key}",
                "HTTP-Referer":  "https://oan.app",
            }
            is_qwen = "qwen" in self._model.lower()
            messages = [{"role": "user", "content": prompt}]
            payload_obj = {
                "model": self._model,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.1,
            }
            if is_qwen:
                # Qwen3.5 reasoning tokens are NOT bounded by max_tokens and blow up
                # to ~9k tokens/call (the /nothink system-prompt hack is ignored over
                # OpenRouter). Disable reasoning outright so the judge behaves like the
                # Gemini/GPT judges (~1k output) and the 2000-token budget goes to the
                # scored answer, not chain-of-thought.
                payload_obj["reasoning"] = {"enabled": False}
            payload = json.dumps(payload_obj).encode()
            req = urllib.request.Request(self._url, data=payload, headers=headers)
            try:
                resp = urllib.request.urlopen(req, timeout=120)
                r = json.loads(resp.read().decode())
                content = (r["choices"][0]["message"].get("content") or "").strip()
                # Strip Qwen3 thinking tokens that leak into content
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if not content:
                    print(
                        f"\n[WARN] {self._model} returned empty content. "
                        f"finish_reason={r['choices'][0].get('finish_reason')} "
                        f"usage={r.get('usage')}",
                        file=sys.stderr,
                    )
                return content
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                print(f"\n[ERROR] {self._model} HTTP {e.code}: {body[:400]}", file=sys.stderr)
                return f"ERROR: HTTP {e.code}"
            except Exception as e:
                print(f"\n[ERROR] {self._model} call failed: {e}", file=sys.stderr)
                return f"ERROR: {e}"

        def generate(self, prompt: str) -> str: return self._call(prompt)
        # Intentionally blocking: sequential calls avoid OpenRouter rate-limit spikes
        # that cause DeepEval's per-test-case timeout to fire under full concurrency.
        async def a_generate(self, prompt: str) -> str: return self._call(prompt)
        def get_model_name(self) -> str: return self._model

    return OpenRouterJudge()


def _make_judge(judge_entry: tuple[str, str, str]):
    label, judge_type, model_id = judge_entry
    if judge_type == "gemma":
        return _make_gemma_judge()
    if judge_type == "openrouter":
        return _make_openrouter_judge(model_id)
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def build_metrics(judge):
    from deepeval.metrics import GEval
    try:
        from deepeval.test_case import SingleTurnParams as EvalParams
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams as EvalParams  # older versions

    kw = {"model": judge} if judge else {}

    return [
        GEval(
            name="Answer Relevancy",
            criteria=(
                "Evaluate whether the response directly and completely answers the question asked. "
                "High score (close to 1): the response addresses the exact question with relevant information. "
                "Low score (close to 0): the response is off-topic, ignores the question, or only partially answers it."
            ),
            evaluation_steps=[
                "Identify the specific information requested in the input (e.g., crop price, livestock price, available crops, marketplace list).",
                "Check if the actual output directly provides that specific information.",
                "If the output gives the exact requested information (e.g., a price when a price is asked), assign a high score (0.8-1.0).",
                "If the output is partially relevant, provides related but incomplete information, or asks for unnecessary clarification, assign a medium score (0.3-0.7).",
                "If the output is off-topic, an error message, or completely fails to address the question, assign a low score (0.0-0.2).",
            ],
            evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT],
            threshold=0.5,
            **kw,
        ),

        GEval(
            name="Agricultural Specificity",
            criteria=(
                "Evaluate whether the response gives specific, actionable agricultural "
                "information relevant to Ethiopian farming. "
                "High score (close to 1): contains concrete details — prices in ETB, "
                "marketplace names, crop or livestock names, or market dates. "
                "Low score (close to 0): vague, generic, or off-topic reply."
            ),
            evaluation_steps=[
                "Check whether the output contains specific agricultural data such as: a price value in ETB (Ethiopian Birr), a marketplace name, a crop or livestock name, a unit (Quintal or Head), or a market date.",
                "If the output includes a numerical price with currency (ETB), a marketplace name, and a unit, assign a high score (0.8-1.0).",
                "If the output mentions relevant crop/livestock names or marketplace names but lacks specific price data, assign a medium score (0.3-0.6).",
                "If the output is a generic message, an error, or lacks any agricultural specifics, assign a low score (0.0-0.2).",
            ],
            evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT],
            threshold=0.5,
            **kw,
        ),

        GEval(
            name="Answer Correctness",
            criteria=(
                "Evaluate whether the actual response conveys the same key facts as the "
                "expected answer. High score: same core information (prices, dates, names, "
                "agronomic advice). Low score: contradicts or omits critical facts."
            ),
            evaluation_steps=[
                "Read the expected answer and identify its key facts (e.g. price value, "
                "marketplace name, agronomic advice, recommended practice).",
                "Check whether the actual output contains those same key facts.",
                "If all key facts are present and consistent, assign a high score (0.8–1.0).",
                "If some facts are missing or only partially correct, assign a medium score (0.3–0.7).",
                "If the actual output contradicts or entirely omits the expected facts, "
                "assign a low score (0.0–0.2).",
            ],
            evaluation_params=[EvalParams.INPUT, EvalParams.ACTUAL_OUTPUT, EvalParams.EXPECTED_OUTPUT],
            threshold=0.5,
            **kw,
        ),

    ]


# ── App call ──────────────────────────────────────────────────────────────────

def call_app(query: str, lang: str, session_id: str) -> tuple[str, str, float]:
    payload = json.dumps({
        "query":       query,
        "session_id":  session_id,
        "source_lang": lang,
        "target_lang": lang,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        raw  = resp.read().decode()
        wall_ms = (time.perf_counter() - t0) * 1000
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        r = json.loads(lines[-1]) if lines else {}
        return r.get("response", ""), r.get("metrics", {}).get("path", "llm"), round(wall_ms, 1)
    except Exception as e:
        wall_ms = (time.perf_counter() - t0) * 1000
        return f"ERROR: {e}", "error", round(wall_ms, 1)


# ── Score extraction ──────────────────────────────────────────────────────────

def _extract_scores(eval_result, metric_names: list[str]) -> list[dict]:
    per_query = []
    for tr in eval_result.test_results:
        scores = {}
        for md in tr.metrics_data or []:
            name = getattr(md, "name", str(md))
            scores[name] = {
                "score":  round(md.score, 3) if md.score is not None else None,
                "passed": md.success,
                "reason": md.reason or "",
            }
        per_query.append(scores)
    return per_query


# ── Run evaluation ────────────────────────────────────────────────────────────

def run_eval():
    _require_deepeval()
    from deepeval.test_case import LLMTestCase

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if EVAL_INTENTS_FILTER:
        rows = [r for r in rows if r.get("expected_intent", "") in EVAL_INTENTS_FILTER]
    if EVAL_LIMIT:
        rows = rows[:EVAL_LIMIT]

    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = os.urandom(3).hex()

    judge_display = ", ".join(
        f"{label} ({model_id})" if model_id else label
        for label, _, model_id in EVAL_JUDGES
    )

    print()
    print("=" * 90)
    print("  OAN PROMPT QUALITY EVALUATION")
    print(f"  Run name : {RUN_NAME}")
    print(f"  Judges   : {judge_display}")
    print(f"  Queries  : {len(rows)}  |  CSV: {CSV_PATH}")
    print(f"  Filters  : intents={EVAL_INTENTS_FILTER or 'all'}  limit={EVAL_LIMIT or 'all'}")
    print("=" * 90)

    # ── Step 1: collect app responses ─────────────────────────────────────────
    print("\n  Step 1/2 — Calling app for each query ...")
    raw_results: list[dict] = []
    test_cases  = []

    for i, row in enumerate(rows, 1):
        query    = row["query"]
        lang     = row["lang"]
        expected = row.get("expected_answer", "").strip()
        response, path, ms = call_app(query, lang, f"eval-{run_id}-{i}")

        raw_results.append({
            "label":           row.get("label", ""),
            "lang":            lang,
            "query":           query,
            "expected_intent": row.get("expected_intent", ""),
            "expected_answer": expected,
            "path":            path,
            "wall_ms":         ms,
            "actual_output":   response,
        })
        test_cases.append(LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected or None,
        ))
        icon = "⚡" if path == "fast" else ("🔁" if path == "llm" else "❌")
        print(f"  [{i:3d}/{len(rows)}] {icon} {ms:6.1f}ms  {query[:55]}")

    # ── Step 2: score with each judge ─────────────────────────────────────────
    sample_metrics = build_metrics(None)
    metric_names   = [getattr(m, "name", type(m).__name__) for m in sample_metrics]

    judge_scores: dict[str, list[dict]] = {}

    for j_idx, judge_entry in enumerate(EVAL_JUDGES, 1):
        label, judge_type, model_id = judge_entry

        judge_model = _make_judge(judge_entry)
        metrics     = build_metrics(judge_model)

        display = f"{label} ({model_id})" if model_id else label
        print(f"\n  Step 2 — Judge [{j_idx}/{len(EVAL_JUDGES)}]: {display}")
        print(f"  {len(metrics)} metrics × {len(test_cases)} queries = {len(metrics)*len(test_cases)} judge calls\n")

        # Score each test case synchronously — avoids DeepEval's asyncio timeout
        # machinery which causes IncompleteRead when blocking HTTP calls are cancelled.
        scores_per_tc: list[dict] = []
        for tc_idx, test_case in enumerate(test_cases, 1):
            tc_scores: dict = {}
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    tc_scores[metric.name] = {
                        "score":  round(metric.score, 3) if metric.score is not None else None,
                        "passed": bool(getattr(metric, "success", False)),
                        "reason": getattr(metric, "reason", "") or "",
                    }
                except Exception as e:
                    tc_scores[metric.name] = {"score": None, "passed": False, "reason": f"ERROR: {e}"}
            scores_per_tc.append(tc_scores)
            score_line = "  ".join(
                f"{n[:14]}: {s.get('score', 'ERR')}" for n, s in tc_scores.items()
            )
            print(f"    [{tc_idx}/{len(test_cases)}] {score_line}")
        judge_scores[label] = scores_per_tc

    # ── Compute ensemble (average across judges) ───────────────────────────────
    labels = [e[0] for e in EVAL_JUDGES]
    ensemble_scores: list[dict] = []
    for i in range(len(raw_results)):
        ens = {}
        for name in metric_names:
            vals = [
                judge_scores[lbl][i].get(name, {}).get("score")
                for lbl in labels
                if judge_scores[lbl][i].get(name, {}).get("score") is not None
            ]
            ens[name] = round(statistics.mean(vals), 3) if vals else None
        ensemble_scores.append(ens)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def _agg(score_list: list[dict]) -> dict:
        result = {}
        for name in metric_names:
            vals   = [s[name]["score"] for s in score_list
                      if isinstance(s.get(name), dict) and s[name].get("score") is not None]
            passed = sum(1 for s in score_list
                         if isinstance(s.get(name), dict) and s[name].get("passed", False))
            result[name] = {
                "avg_score": round(statistics.mean(vals), 3) if vals else None,
                "pass_rate": round(passed / len(score_list), 3) if score_list else None,
            }
        return result

    aggregate = {lbl: _agg(judge_scores[lbl]) for lbl in labels}
    aggregate["ensemble"] = {
        name: {
            "avg_score": round(
                statistics.mean(
                    aggregate[lbl][name]["avg_score"]
                    for lbl in labels
                    if aggregate[lbl][name]["avg_score"] is not None
                ), 3
            ),
            "pass_rate": None,
        }
        for name in metric_names
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    col_w = 16
    print()
    print("=" * 90)
    print(f"  RESULTS — {RUN_NAME}")
    print("=" * 90)
    header = f"  {'Metric':<36}"
    for lbl in labels:
        header += f" {lbl[:col_w]:>{col_w}}"
    header += f" {'ensemble':>{col_w}}"
    print(header)
    print("  " + "─" * (36 + (len(labels) + 1) * (col_w + 1)))
    for name in metric_names:
        row_str = f"  {name:<36}"
        for lbl in labels:
            s = aggregate[lbl][name]["avg_score"]
            row_str += f" {(f'{s:.3f}' if s is not None else 'N/A'):>{col_w}}"
        ens = aggregate["ensemble"][name]["avg_score"]
        row_str += f" {(f'{ens:.3f}' if ens is not None else 'N/A'):>{col_w}}"
        print(row_str)

    print("=" * 90)

    # ── Write CSV ──────────────────────────────────────────────────────────────
    def _slug(name: str) -> str:
        return name.lower().replace(" ", "_")

    fieldnames = [
        "run_name", "timestamp", "judges",
        "label", "lang", "query", "expected_intent", "expected_answer",
        "path", "wall_ms", "actual_output",
    ]
    for lbl in labels:
        for name in metric_names:
            s = f"judge_{lbl}_{_slug(name)}"
            fieldnames += [f"{s}_score", f"{s}_passed", f"{s}_reason"]
    for name in metric_names:
        fieldnames.append(f"ensemble_{_slug(name)}_score")

    out_path = RESULTS_DIR / f"eval_{RUN_NAME}_{ts}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, rr in enumerate(raw_results):
            row = {
                "run_name":        RUN_NAME,
                "timestamp":       ts,
                "judges":          "|".join(labels),
                "label":           rr["label"],
                "lang":            rr["lang"],
                "query":           rr["query"],
                "expected_intent": rr["expected_intent"],
                "expected_answer": rr["expected_answer"],
                "path":            rr["path"],
                "wall_ms":         rr["wall_ms"],
                "actual_output":   rr["actual_output"],
            }
            for lbl in labels:
                for name in metric_names:
                    s  = f"judge_{lbl}_{_slug(name)}"
                    sc = judge_scores[lbl][i].get(name, {})
                    row[f"{s}_score"]  = sc.get("score", "")
                    row[f"{s}_passed"] = sc.get("passed", "")
                    row[f"{s}_reason"] = sc.get("reason", "")
            for name in metric_names:
                row[f"ensemble_{_slug(name)}_score"] = ensemble_scores[i].get(name, "")
            writer.writerow(row)

    print(f"\n  Saved → {out_path}")
    print(f"  To compare after your prompt changes, run:")
    print(f"    python3 /app/tests/eval_prompt_quality.py \\")
    print(f"      --compare results/eval_{RUN_NAME}_{ts}.csv results/eval_prompt_v2_<ts>.csv")


# ── Compare two runs ──────────────────────────────────────────────────────────

def _load_csv_run(path: str) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        print(f"ERROR: {path} is empty")
        sys.exit(1)

    meta = {
        "run_name":  rows[0].get("run_name", Path(path).stem),
        "timestamp": rows[0].get("timestamp", ""),
        "judges":    rows[0].get("judges", "?"),
        "n":         len(rows),
    }

    ensemble_cols = [c for c in rows[0] if c.startswith("ensemble_") and c.endswith("_score")]
    score_cols    = ensemble_cols or [c for c in rows[0] if c.endswith("_score")]
    slugs         = [
        (c.removeprefix("ensemble_") if c.startswith("ensemble_") else c)[:-6]
        for c in score_cols
    ]
    metric_names = [s.replace("_", " ").title() for s in slugs]

    aggregate = {}
    for col, name in zip(score_cols, metric_names):
        vals = [float(r[col]) for r in rows if r.get(col) not in ("", None)]
        aggregate[name] = {"avg_score": round(statistics.mean(vals), 3) if vals else None}

    return {"meta": meta, "metric_names": metric_names, "aggregate": aggregate}


def compare_runs(path_a: str, path_b: str):
    a = _load_csv_run(path_a)
    b = _load_csv_run(path_b)
    ma, mb      = a["meta"], b["meta"]
    all_metrics = sorted(set(a["metric_names"] + b["metric_names"]))

    print()
    print("=" * 90)
    print("  PROMPT QUALITY COMPARISON")
    print(f"  Baseline : {ma['run_name']}  ({ma['timestamp']})  [{ma['n']} queries]")
    print(f"  Updated  : {mb['run_name']}  ({mb['timestamp']})  [{mb['n']} queries]")
    print(f"  Judges   : {ma['judges']} → {mb['judges']}")
    print("=" * 90)
    print(f"\n  {'Metric':<38} {ma['run_name']:>14} {mb['run_name']:>14} {'Delta':>8}  Change")
    print("  " + "─" * 82)
    for name in all_metrics:
        sa = a["aggregate"].get(name, {}).get("avg_score")
        sb = b["aggregate"].get(name, {}).get("avg_score")
        if sa is None or sb is None:
            print(f"  {name:<38} {'N/A':>14} {'N/A':>14}")
            continue
        delta = sb - sa
        pct   = (delta / sa * 100) if sa else 0.0
        arrow = "✅" if delta > 0.01 else ("❌" if delta < -0.01 else "➡️ ")
        print(f"  {name:<38} {sa:>14.3f} {sb:>14.3f} {delta:>+8.3f}  {arrow} {pct:+.1f}%")
    print("=" * 90)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OAN prompt quality evaluation using DeepEval.")
    parser.add_argument(
        "--compare", nargs=2,
        metavar=("BASELINE_CSV", "UPDATED_CSV"),
        help="Compare two saved eval runs instead of running a new one.",
    )
    args = parser.parse_args()
    if args.compare:
        compare_runs(*args.compare)
    else:
        run_eval()


if __name__ == "__main__":
    main()
