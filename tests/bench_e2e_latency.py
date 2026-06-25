"""
E2E Latency Benchmark: Intent Router Fast Path vs LLM Baseline

Measures the real wall-clock latency improvement the intent router delivers by
comparing two paths for the same agricultural queries:

  BEFORE — direct LLM call (Gemma on EC2) per query, no routing
  AFTER  — /api/chat/ with intent router fast path engaged

Results are printed as a formatted table and saved to bench_results.json.

Run inside the app container:
    docker exec oan_app python3 /app/tests/bench_e2e_latency.py

Options (env vars):
    BENCH_LLM_N=3       number of LLM baseline samples per query (default 3)
    BENCH_FAST_N=5      number of fast-path samples per query (default 5)
    BENCH_WARMUP=1      warmup calls to discard (default 1)
    BENCH_JSON=1        write bench_results.json (default 1)
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BASE        = "http://localhost:8000/api"
LLM_URL     = os.getenv("OPENAI_BASE_URL", "http://52.66.116.220:8080").rstrip("/") + "/v1/chat/completions"
LLM_MODEL   = os.getenv("LLM_MODEL_NAME", "gemma-4-26b-a4b")
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

BENCH_LLM_N  = int(os.getenv("BENCH_LLM_N",  "3"))
BENCH_FAST_N = int(os.getenv("BENCH_FAST_N", "5"))
BENCH_WARMUP = int(os.getenv("BENCH_WARMUP", "1"))
BENCH_JSON   = os.getenv("BENCH_JSON", "1") != "0"

# ── Test corpus ──────────────────────────────────────────────────────────────
QUERIES = [
    # (label, lang, query)
    # Full match = both crop/livestock AND marketplace found → decision=high → real DB data returned
    # Partial     = only crop/livestock found → decision=medium → follow-up question (no LLM)
    ("EN crop price — full match",     "en", "What is the price of maize in Bishoftu?"),
    ("EN crop price — partial",        "en", "What is the current price of teff?"),
    ("EN livestock price — full match","en", "How much does a goat cost in Miyo?"),
    ("EN livestock price — partial",   "en", "What is the current price of sheep?"),
    ("EN marketplace listing",         "en", "Show me all active crop marketplaces"),
    ("AM crop price — full match",      "am", "በቢሾፍቱ ገበያ የስንዴ ዋጋ ስንት ነው?"),
    ("AM crop price — full match 2",   "am", "በቢሾፍቱ ገበያ የበቆሎ ዋጋ ምን ያህል ነው?"),
    ("AM crop price — partial",        "am", "የጤፍ ዋጋ ምን ያህል ነው?"),
    ("AM crop price — partial 2",      "am", "የስንዴ ዋጋ ምን ያህል ነው?"),
    ("AM livestock price — full match","am", "በሜኢሶ የፍየል ዋጋ ምን ያህል ነው?"),
    ("AM livestock price — full match 2","am","በያሎ የበሬ ዋጋ ምን ያህል ነው?"),
    ("AM livestock price — partial",   "am", "የፍየል ዋጋ ስንት ነው?"),
    ("AM livestock price — partial 2", "am", "የበግ ዋጋ ምን ያህል ነው?"),
    ("AM marketplace listing",         "am", "ሁሉንም ንቁ ገበያዎች አሳየኝ"),
    ("AM livestock marketplaces",      "am", "ሁሉንም የከብት ገበያዎች ዝርዝር"),
    ("AM crop listing at market",      "am", "የቢሾፍቱ የሰብል ዝርዝር አሳየኝ"),
]

SYSTEM_PROMPT = (
    "You are OAN, an agricultural AI assistant for Ethiopian farmers. "
    "Answer questions about crop prices, livestock prices, and marketplaces."
)


# ── HTTP helpers ─────────────────────────────────────────────────────────────
def _post(url: str, payload: dict, headers: dict | None = None, timeout: int = 60) -> tuple[dict, int]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json", **(headers or {})})
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
    """Call /api/chat/ and return (wall_ms, path, router_time_ms)."""
    t0 = time.perf_counter()
    r, code = _post(f"{BASE}/chat/", {
        "query": query,
        "session_id": session_id,
        "source_lang": lang,
        "target_lang": lang,
    })
    wall_ms = (time.perf_counter() - t0) * 1000
    if code not in (200, 201) or not isinstance(r, dict):
        return wall_ms, "error", "—"
    metrics = r.get("metrics", {})
    return wall_ms, metrics.get("path", "llm"), str(metrics.get("router_time", "—"))


def chat_llm_baseline(query: str) -> float:
    """Call EC2 Gemma directly (no routing). Returns wall_ms."""
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
    if code not in (200, 201):
        return -1.0
    return wall_ms


# ── Stats helpers ─────────────────────────────────────────────────────────────
def pct(samples: list[float], p: int) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import os
    run_id = os.urandom(3).hex()

    W = 30  # label column width
    print()
    print("=" * 100)
    print("  OAN INTENT ROUTER — E2E LATENCY BENCHMARK")
    print(f"  LLM baseline: {LLM_MODEL}  |  fast-path samples: {BENCH_FAST_N}  |  LLM samples: {BENCH_LLM_N}")
    print("=" * 100)
    print(f"\n  Warming up ({BENCH_WARMUP} call(s) per query discarded) ...")

    results = []

    for label, lang, query in QUERIES:
        print(f"\n  [{label}]  lang={lang}")
        print(f"  Q: {query}")

        # ── Fast path ──────────────────────────────────────────────────────
        fast_samples = []
        last_path = "?"
        last_router_t = "—"
        for i in range(BENCH_WARMUP + BENCH_FAST_N):
            sid = f"bench-{run_id}-fast-{label[:8]}-{i}"
            ms, path, rt = chat_fast(query, lang, sid)
            if i >= BENCH_WARMUP:
                fast_samples.append(ms)
                last_path = path
                last_router_t = rt
        fast = summarize(fast_samples)
        icon = "⚡" if last_path == "fast" else "🔁"
        print(f"  {icon} fast-path  avg={fast['avg']:6.1f}ms  p50={fast['p50']:6.1f}  "
              f"p90={fast['p90']:6.1f}  min={fast['min']:6.1f}  max={fast['max']:6.1f}  "
              f"[router_overhead≈{last_router_t}ms]")

        # ── LLM baseline ───────────────────────────────────────────────────
        llm_samples = []
        print(f"  ⏳ measuring LLM baseline ({BENCH_LLM_N} calls) ...", end="", flush=True)
        for i in range(BENCH_WARMUP + BENCH_LLM_N):
            ms = chat_llm_baseline(query)
            if i >= BENCH_WARMUP:
                llm_samples.append(ms)
            print(".", end="", flush=True)
        print()
        llm = summarize(llm_samples)
        print(f"  🔁 LLM base   avg={llm['avg']:6.1f}ms  p50={llm['p50']:6.1f}  "
              f"p90={llm['p90']:6.1f}  min={llm['min']:6.1f}  max={llm['max']:6.1f}")

        speedup = round(llm["avg"] / fast["avg"], 1) if fast["avg"] > 0 else 0
        print(f"  📉 Speedup: {speedup}×  ({llm['avg']:.0f}ms → {fast['avg']:.0f}ms)")

        results.append({
            "label":   label,
            "lang":    lang,
            "query":   query,
            "path":    last_path,
            "fast":    fast,
            "llm":     llm,
            "speedup": speedup,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    hdr = f"  {'Query':<40} {'Fast avg':>10} {'LLM avg':>10} {'Speedup':>9} {'Router overhead':>16}"
    print(hdr)
    print("  " + "─" * 96)
    for r in results:
        path_icon = "⚡" if r["path"] == "fast" else "🔁"
        print(f"  {path_icon} {r['label']:<39} {r['fast']['avg']:>9.1f}ms {r['llm']['avg']:>9.1f}ms "
              f"{r['speedup']:>8.1f}×")
    print("  " + "─" * 96)

    # Aggregates
    valid = [r for r in results if r["path"] == "fast"]
    if valid:
        avg_fast = statistics.mean(r["fast"]["avg"] for r in valid)
        avg_llm  = statistics.mean(r["llm"]["avg"]  for r in valid)
        avg_spd  = statistics.mean(r["speedup"]      for r in valid)
        print(f"  {'Average (fast-path queries)':<40} {avg_fast:>9.1f}ms {avg_llm:>9.1f}ms {avg_spd:>8.1f}×")
    print("=" * 100)
    print()

    # ── JSON export ───────────────────────────────────────────────────────────
    if BENCH_JSON:
        out = {
            "model":      LLM_MODEL,
            "fast_n":     BENCH_FAST_N,
            "llm_n":      BENCH_LLM_N,
            "warmup":     BENCH_WARMUP,
            "results":    results,
            "aggregate": {
                "avg_fast_ms":  round(avg_fast, 1) if valid else None,
                "avg_llm_ms":   round(avg_llm,  1) if valid else None,
                "avg_speedup":  round(avg_spd,  1) if valid else None,
            },
        }
        out_path = Path(__file__).parent / "bench_results.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
