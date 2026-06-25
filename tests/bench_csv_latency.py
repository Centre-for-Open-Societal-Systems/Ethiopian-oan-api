"""
E2E Latency Benchmark: Intent Router Fast Path vs LLM Baseline
Reads queries from downloads/benchmark_samples.csv (200 queries: 100 EN + 100 AM).

Records per query:
  - Latency stats (avg, p50, p90, p99, min, max) for both paths
  - Actual response text from the last sample of each path
  - Which path was taken (fast / llm / error)
  - Speedup multiplier

Usage inside the app container:
    docker exec oan_app python3 /app/tests/bench_csv_latency.py

Options (env vars):
    BENCH_LLM_N=3       LLM baseline samples per query (default 3)
    BENCH_FAST_N=5      fast-path samples per query (default 5)
    BENCH_WARMUP=1      warmup calls to discard (default 1)
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BASE        = "http://localhost:8000/api"
LLM_URL     = os.getenv("OPENAI_BASE_URL", "http://52.66.116.220:8080").rstrip("/") + "/v1/chat/completions"
LLM_MODEL   = os.getenv("LLM_MODEL_NAME", "gemma-4-26b-a4b")
LLM_API_KEY = os.getenv("TRITON_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")

BENCH_LLM_N  = int(os.getenv("BENCH_LLM_N",  "3"))
BENCH_FAST_N = int(os.getenv("BENCH_FAST_N", "5"))
BENCH_WARMUP = int(os.getenv("BENCH_WARMUP", "1"))

_here = Path(__file__).parent
CSV_PATH    = Path(os.getenv("BENCH_CSV", str(_here.parent / "downloads" / "benchmark_samples.csv")))
RESULTS_DIR = _here.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are OAN, an agricultural AI assistant for Ethiopian farmers. "
    "Answer questions about crop prices, livestock prices, and marketplaces."
)


def load_queries(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Quality metrics ───────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
    return dp[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance / len(reference). Capped at 1.0."""
    ref = reference.strip()
    if not ref:
        return 0.0
    return min(_levenshtein(ref, hypothesis.strip()) / len(ref), 1.0)


def f1_token(reference: str, hypothesis: str) -> float:
    """Token-overlap F1 (bag-of-words, case-insensitive)."""
    import re
    ref_tokens  = re.findall(r"\w+", reference.lower())
    hyp_tokens  = re.findall(r"\w+", hypothesis.lower())
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_set, hyp_set = set(ref_tokens), set(hyp_tokens)
    common = ref_set & hyp_set
    if not common:
        return 0.0
    precision = len(common) / len(hyp_set)
    recall    = len(common) / len(ref_set)
    return round(2 * precision * recall / (precision + recall), 3)


def _post(url: str, payload: dict, headers: dict | None = None, timeout: int = 60) -> tuple[dict, int]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode()
        lines = [l for l in raw.splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else {}, resp.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read()), e.code
        except Exception:
            return {"error": str(e)}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def chat_fast(query: str, lang: str, session_id: str) -> tuple[float, str, str]:
    """Returns (wall_ms, path, response_text)."""
    t0 = time.perf_counter()
    r, code = _post(f"{BASE}/chat/", {
        "query": query,
        "session_id": session_id,
        "source_lang": lang,
        "target_lang": lang,
    })
    wall_ms = (time.perf_counter() - t0) * 1000
    if code not in (200, 201) or not isinstance(r, dict):
        return wall_ms, "error", f"HTTP {code}"
    path = r.get("metrics", {}).get("path", "llm")
    response_text = r.get("response", "")
    return wall_ms, path, response_text


def chat_llm_baseline(query: str) -> tuple[float, str]:
    """Returns (wall_ms, response_text)."""
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    t0 = time.perf_counter()
    r, code = _post(LLM_URL, payload, headers=headers, timeout=90)
    wall_ms = (time.perf_counter() - t0) * 1000
    if code not in (200, 201) or not isinstance(r, dict):
        return -1.0, f"HTTP {code}"
    try:
        text = r["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        text = str(r)
    return wall_ms, text


def pct(samples: list[float], p: int) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def summarize(samples: list[float]) -> dict:
    valid = [s for s in samples if s > 0]
    if not valid:
        return {"avg": 0, "p50": 0, "p90": 0, "p99": 0, "min": 0, "max": 0, "n": 0}
    return {
        "avg": round(statistics.mean(valid), 1),
        "p50": round(pct(valid, 50), 1),
        "p90": round(pct(valid, 90), 1),
        "p99": round(pct(valid, 99), 1),
        "min": round(min(valid), 1),
        "max": round(max(valid), 1),
        "n":   len(valid),
    }


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        sys.exit(1)

    queries = load_queries(CSV_PATH)
    run_id  = os.urandom(3).hex()
    ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print()
    print("=" * 110)
    print("  OAN INTENT ROUTER — CSV E2E LATENCY BENCHMARK")
    print(f"  CSV: {CSV_PATH}  |  queries: {len(queries)}  |  LLM model: {LLM_MODEL}")
    print(f"  fast-path samples: {BENCH_FAST_N}  |  LLM samples: {BENCH_LLM_N}  |  warmup: {BENCH_WARMUP}")
    print("=" * 110)

    results = []

    for idx, row in enumerate(queries, 1):
        label           = row["label"]
        lang            = row["lang"]
        query           = row["query"]
        expected_intent = row.get("expected_intent", "")
        expected_answer = row.get("expected_answer", "").strip()

        flag = "[EN]" if lang == "en" else "[AM]"
        print(f"\n  ({idx}/{len(queries)}) {flag} {label}")
        print(f"  Q: {query}")
        if expected_answer:
            print(f"  Expected: {expected_answer[:100]}")

        # Fast path — capture response from the last measured sample
        fast_samples = []
        last_path = "?"
        last_fast_response = ""
        for i in range(BENCH_WARMUP + BENCH_FAST_N):
            sid = f"bench-{run_id}-{idx}-{i}"
            ms, path, response = chat_fast(query, lang, sid)
            if i >= BENCH_WARMUP:
                fast_samples.append(ms)
                last_path = path
                last_fast_response = response

        fast = summarize(fast_samples)
        icon = "⚡" if last_path == "fast" else ("🔁" if last_path == "llm" else "❌")
        print(f"  {icon} fast-path  avg={fast['avg']:6.1f}ms  p50={fast['p50']:6.1f}  "
              f"p90={fast['p90']:6.1f}  min={fast['min']:5.1f}  max={fast['max']:6.1f}")
        print(f"     Response: {last_fast_response[:100]}")

        # LLM baseline — capture response from the last measured sample
        print(f"  ⏳ LLM baseline ({BENCH_LLM_N} call(s)) ...", end="", flush=True)
        llm_samples = []
        last_llm_response = ""
        for i in range(BENCH_WARMUP + BENCH_LLM_N):
            ms, response = chat_llm_baseline(query)
            if i >= BENCH_WARMUP:
                llm_samples.append(ms)
                last_llm_response = response
            print(".", end="", flush=True)
        print()
        llm = summarize(llm_samples)
        print(f"  🔁 LLM base    avg={llm['avg']:6.1f}ms  p50={llm['p50']:6.1f}  "
              f"p90={llm['p90']:6.1f}  min={llm['min']:5.1f}  max={llm['max']:6.1f}")
        print(f"     Response: {last_llm_response[:100]}")

        speedup = round(llm["avg"] / fast["avg"], 1) if fast["avg"] > 0 else 0.0
        print(f"  📉 Speedup: {speedup}×")

        # Quality metrics vs expected_answer (only when ground truth is available)
        quality: dict = {}
        if expected_answer:
            fast_cer = cer(expected_answer, last_fast_response)
            fast_f1  = f1_token(expected_answer, last_fast_response)
            llm_cer  = cer(expected_answer, last_llm_response)
            llm_f1   = f1_token(expected_answer, last_llm_response)
            quality  = {
                "fast_cer": round(fast_cer, 3),
                "fast_f1":  fast_f1,
                "llm_cer":  round(llm_cer, 3),
                "llm_f1":   llm_f1,
            }
            print(f"  📊 Quality  fast→ CER={fast_cer:.3f} F1={fast_f1:.3f}  "
                  f"llm→ CER={llm_cer:.3f} F1={llm_f1:.3f}")

        results.append({
            "label":           label,
            "lang":            lang,
            "query":           query,
            "expected_intent": expected_intent,
            "expected_answer": expected_answer,
            "path":            last_path,
            "fast":            fast,
            "fast_response":   last_fast_response,
            "llm":             llm,
            "llm_response":    last_llm_response,
            "speedup":         speedup,
            "quality":         quality,
        })

    # Summary table
    print()
    print("=" * 110)
    print("  SUMMARY TABLE")
    print("=" * 110)
    print(f"  {'Query':<45} {'Lang':>4} {'Fast avg':>10} {'LLM avg':>10} {'Speedup':>8} {'Path':>6}")
    print("  " + "─" * 106)
    for r in results:
        icon = "⚡" if r["path"] == "fast" else "🔁"
        print(f"  {icon} {r['label']:<44} {r['lang']:>4} "
              f"{r['fast']['avg']:>9.1f}ms {r['llm']['avg']:>9.1f}ms {r['speedup']:>7.1f}×")
    print("  " + "─" * 106)

    fast_rows = [r for r in results if r["path"] == "fast"]
    llm_rows  = [r for r in results if r["path"] != "fast"]
    en_fast   = [r for r in fast_rows if r["lang"] == "en"]
    am_fast   = [r for r in fast_rows if r["lang"] == "am"]

    def agg(rows: list) -> dict | None:
        if not rows:
            return None
        return {
            "count":       len(rows),
            "avg_fast_ms": round(statistics.mean(r["fast"]["avg"] for r in rows), 1),
            "avg_llm_ms":  round(statistics.mean(r["llm"]["avg"]  for r in rows), 1),
            "avg_speedup": round(statistics.mean(r["speedup"]      for r in rows), 1),
        }

    aggregate = {
        "total_queries":      len(results),
        "fast_path_count":    len(fast_rows),
        "llm_fallback_count": len(llm_rows),
        "overall":            agg(fast_rows),
        "english":            agg(en_fast),
        "amharic":            agg(am_fast),
    }

    if aggregate["overall"]:
        o = aggregate["overall"]
        print(f"\n  Overall fast-path ({o['count']} queries):  "
              f"avg fast={o['avg_fast_ms']}ms  avg LLM={o['avg_llm_ms']}ms  avg speedup={o['avg_speedup']}×")
    if aggregate["english"]:
        e = aggregate["english"]
        print(f"  English  ({e['count']:2d} queries):  avg fast={e['avg_fast_ms']}ms  speedup={e['avg_speedup']}×")
    if aggregate["amharic"]:
        a = aggregate["amharic"]
        print(f"  Amharic  ({a['count']:2d} queries):  avg fast={a['avg_fast_ms']}ms  speedup={a['avg_speedup']}×")

    # Quality aggregate (only queries that had an expected_answer)
    scored = [r for r in results if r.get("quality")]
    if scored:
        avg_fast_cer = round(statistics.mean(r["quality"]["fast_cer"] for r in scored), 3)
        avg_fast_f1  = round(statistics.mean(r["quality"]["fast_f1"]  for r in scored), 3)
        avg_llm_cer  = round(statistics.mean(r["quality"]["llm_cer"]  for r in scored), 3)
        avg_llm_f1   = round(statistics.mean(r["quality"]["llm_f1"]   for r in scored), 3)
        aggregate["quality"] = {
            "scored_queries": len(scored),
            "fast_avg_cer": avg_fast_cer,
            "fast_avg_f1":  avg_fast_f1,
            "llm_avg_cer":  avg_llm_cer,
            "llm_avg_f1":   avg_llm_f1,
        }
        print(f"\n  Quality ({len(scored)} scored queries):")
        print(f"    Fast-path → avg CER={avg_fast_cer:.3f}  avg F1={avg_fast_f1:.3f}")
        print(f"    LLM base  → avg CER={avg_llm_cer:.3f}  avg F1={avg_llm_f1:.3f}")
        print(f"    (lower CER = better match, higher F1 = better overlap with expected answer)")
    print("=" * 110)

    out_path = RESULTS_DIR / f"bench_{ts}.json"
    out = {
        "run_id":    run_id,
        "timestamp": ts,
        "model":     LLM_MODEL,
        "csv_source": str(CSV_PATH),
        "fast_n":    BENCH_FAST_N,
        "llm_n":     BENCH_LLM_N,
        "warmup":    BENCH_WARMUP,
        "results":   results,
        "aggregate": aggregate,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
