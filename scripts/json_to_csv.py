"""
Convert a bench_*.json results file to a flat CSV.

Usage:
    python scripts/json_to_csv.py results/bench_20260603T130545Z.json
    python scripts/json_to_csv.py  # auto-picks the newest results/*.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

FIELDNAMES = [
    "label", "lang", "query", "expected_intent", "expected_answer",
    "path",
    "fast_avg_ms", "fast_p50_ms", "fast_p90_ms", "fast_p99_ms",
    "fast_min_ms", "fast_max_ms",
    "fast_response",
    "llm_avg_ms",  "llm_p50_ms",  "llm_p90_ms",  "llm_p99_ms",
    "llm_min_ms",  "llm_max_ms",
    "llm_response",
    "speedup",
    "fast_cer", "fast_f1", "llm_cer", "llm_f1",
]


def main():
    results_dir = Path(__file__).parent.parent / "results"

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        candidates = sorted(results_dir.glob("bench_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No bench_*.json files found in results/")
            sys.exit(1)
        json_path = candidates[0]

    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path

    print(f"Reading  {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    csv_path = json_path.with_suffix(".csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for r in data["results"]:
            fast = r.get("fast", {})
            llm  = r.get("llm",  {})
            q    = r.get("quality", {})

            writer.writerow({
                "label":            r.get("label", ""),
                "lang":             r.get("lang", ""),
                "query":            r.get("query", ""),
                "expected_intent":  r.get("expected_intent", ""),
                "expected_answer":  r.get("expected_answer", ""),
                "path":             r.get("path", ""),
                "fast_avg_ms":      fast.get("avg", ""),
                "fast_p50_ms":      fast.get("p50", ""),
                "fast_p90_ms":      fast.get("p90", ""),
                "fast_p99_ms":      fast.get("p99", ""),
                "fast_min_ms":      fast.get("min", ""),
                "fast_max_ms":      fast.get("max", ""),
                "fast_response":    r.get("fast_response", ""),
                "llm_avg_ms":       llm.get("avg", ""),
                "llm_p50_ms":       llm.get("p50", ""),
                "llm_p90_ms":       llm.get("p90", ""),
                "llm_p99_ms":       llm.get("p99", ""),
                "llm_min_ms":       llm.get("min", ""),
                "llm_max_ms":       llm.get("max", ""),
                "llm_response":     r.get("llm_response", ""),
                "speedup":          r.get("speedup", ""),
                "fast_cer":         q.get("fast_cer", ""),
                "fast_f1":          q.get("fast_f1", ""),
                "llm_cer":          q.get("llm_cer", ""),
                "llm_f1":           q.get("llm_f1", ""),
            })

    print(f"Written  {csv_path}")
    print(f"Rows     {len(data['results'])}")


if __name__ == "__main__":
    main()
