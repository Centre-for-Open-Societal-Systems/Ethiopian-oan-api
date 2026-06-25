#!/usr/bin/env python3
"""
RAG Evaluation Script — Qwen via /api/chat/

Sends benchmark Q&A pairs through the live chat endpoint (with RAG),
scores answers against ground truth, and writes CSV results.

Usage:
    pip install openpyxl httpx sacrebleu editdistance

    # Quick test (5 EN + 5 AM)
    N_SAMPLES=5 python tests/rag_eval.py

    # Full run (116 EN + 116 AM)
    python tests/rag_eval.py

    # Custom backend
    BACKEND_URL=http://localhost:8000 python tests/rag_eval.py
"""

import csv
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

import editdistance
import httpx
import openpyxl
from sacrebleu import CHRF

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL    = os.getenv("BACKEND_URL", "https://98.90.156.237.nip.io")
BENCHMARK_PATH = os.getenv(
    "BENCHMARK_PATH",
    r"C:/Users/dadsh/Downloads/oan/RAG_Evaluation_Benchmark_Set-116QA (1).xlsx"
)
N_SAMPLES      = int(os.getenv("N_SAMPLES", "0"))   # 0 = all 116 per lang
DELAY          = float(os.getenv("DELAY", "1.5"))    # seconds between requests
RESULTS_DIR    = os.getenv("RESULTS_DIR", "results/rag_eval")
# MODEL_NAME     = os.getenv("MODEL_NAME", "qwen/qwen3.5-flash-02-23")  # label only
MODEL_NAME     = os.getenv("MODEL_NAME", "google/gemma-4-26b-a4b-it")  # label only

# RESULTS_CSV = os.path.join(RESULTS_DIR, "rag_benchmark_results.csv")
# SAMPLES_CSV = os.path.join(RESULTS_DIR, "rag_benchmark_samples.csv")
RESULTS_CSV = os.path.join(RESULTS_DIR, "gemma4_benchmark_results.csv")
SAMPLES_CSV = os.path.join(RESULTS_DIR, "gemma4_benchmark_samples.csv")

RESULTS_FIELDS = ["Dataset", "Model", "N_Samples", "F1", "EM", "chrF++", "CER", "NED", "Avg_Latency_ms", "Timestamp"]
SAMPLES_FIELDS = ["ID", "Lang", "Model", "Question", "REF", "PRED", "F1", "EM", "CER", "NED", "Latency_ms"]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (from llm_eval.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())


def calc_f1(pred, ref):
    p_tok = _normalize(pred).split()
    r_tok = _normalize(ref).split()
    common = sum((Counter(p_tok) & Counter(r_tok)).values())
    if not p_tok or not r_tok or common == 0:
        return 0.0
    prec = common / len(p_tok)
    rec  = common / len(r_tok)
    return 2 * prec * rec / (prec + rec)


def calc_em(pred, ref):
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def calc_cer(pred, ref):
    if not ref:
        return 1.0 if pred else 0.0
    return editdistance.distance(pred, ref) / len(ref)


def calc_ned(pred, ref):
    mx = max(len(pred), len(ref))
    if mx == 0:
        return 0.0
    return editdistance.distance(pred, ref) / mx


def calc_chrf(preds, refs):
    if not preds or not refs:
        return 0.0
    chrf = CHRF(word_order=2)
    return chrf.corpus_score(preds, [[r] for r in refs]).score / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Excel Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_benchmark(path, n_samples=0):
    """Load EN + AM samples from benchmark Excel. Returns list of dicts."""
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    samples = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        id_, en_q, am_q, en_a, am_a, *_ = row
        if en_q and en_a:
            samples.append({"id": id_, "question": str(en_q), "ref": str(en_a), "lang": "en"})
        if am_q and am_a:
            samples.append({"id": id_, "question": str(am_q), "ref": str(am_a), "lang": "am"})

    if n_samples:
        en = [s for s in samples if s["lang"] == "en"][:n_samples]
        am = [s for s in samples if s["lang"] == "am"][:n_samples]
        return en + am
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Chat API Caller
# ═══════════════════════════════════════════════════════════════════════════════

def parse_response(text):
    """Parse answer and latency from plain JSON or SSE response."""
    answer, latency = "", 0.0
    # Try plain JSON first
    try:
        d = json.loads(text.strip())
        if d.get("status") == "success":
            return d.get("response", ""), d.get("metrics", {}).get("total_e2e_latency", 0.0)
    except json.JSONDecodeError:
        pass
    # Fall back to SSE (data: lines)
    for line in text.splitlines():
        if line.startswith("data:"):
            raw = line[5:].strip()
            if raw:
                try:
                    d = json.loads(raw)
                    if d.get("status") == "success":
                        answer = d.get("response", "")
                        latency = d.get("metrics", {}).get("total_e2e_latency", 0.0)
                except json.JSONDecodeError:
                    pass
    return answer, latency


def query_chat(question, lang, backend_url, retries=2):
    """POST to /api/chat/ and parse the response. Returns (answer, latency_ms)."""
    payload = {
        "query": question,
        "source_lang": lang,
        "target_lang": lang,
    }
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=90, verify=False) as client:
                resp = client.post(f"{backend_url}/api/chat/", json=payload)
            return parse_response(resp.text)
        except Exception as e:
            if attempt < retries:
                print(f"    Retry {attempt + 1}/{retries} after error: {e}")
                time.sleep(3)
            else:
                print(f"    FAILED after {retries + 1} attempts: {e}")
                return "", 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def append_csv(path, fieldnames, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_lang(samples, lang, backend_url):
    """Evaluate all samples for a given language. Returns aggregate dict."""
    lang_samples = [s for s in samples if s["lang"] == lang]
    if not lang_samples:
        return None

    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}   Samples: {len(lang_samples)}")
    print(f"{'='*60}")

    f1_scores, em_scores, cer_scores, ned_scores = [], [], [], []
    latencies = []
    all_preds, all_refs = [], []

    for i, s in enumerate(lang_samples):
        pred, latency = query_chat(s["question"], lang, backend_url)
        ref = s["ref"]

        f1  = calc_f1(pred, ref)
        em  = calc_em(pred, ref)
        cer = calc_cer(pred, ref)
        ned = calc_ned(pred, ref)

        f1_scores.append(f1)
        em_scores.append(em)
        cer_scores.append(cer)
        ned_scores.append(ned)
        latencies.append(latency)
        all_preds.append(pred)
        all_refs.append(ref)

        status = "EM!" if em == 1.0 else f"F1={f1:.3f}"
        preview = pred[:70].encode('ascii', errors='replace').decode('ascii')
        print(f"  [{i+1:>3}/{len(lang_samples)}] {status} | {preview}")

        append_csv(SAMPLES_CSV, SAMPLES_FIELDS, {
            "ID": s["id"], "Lang": lang, "Model": MODEL_NAME,
            "Question": s["question"], "REF": ref, "PRED": pred,
            "F1": round(f1, 4), "EM": round(em, 4),
            "CER": round(cer, 4), "NED": round(ned, 4),
            "Latency_ms": round(latency, 1),
        })

        if i < len(lang_samples) - 1:
            time.sleep(DELAY)

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    try:
        chrf = calc_chrf(all_preds, all_refs)
    except Exception as e:
        print(f"  chrF++ error: {e}")
        chrf = 0.0

    agg = {
        "Dataset":        f"rag:{lang}",
        "Model":          MODEL_NAME,
        "N_Samples":      len(lang_samples),
        "F1":             round(avg(f1_scores), 4),
        "EM":             round(avg(em_scores), 4),
        "chrF++":         round(chrf, 4),
        "CER":            round(avg(cer_scores), 4),
        "NED":            round(avg(ned_scores), 4),
        "Avg_Latency_ms": round(avg(latencies), 1),
        "Timestamp":      datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n  => F1={agg['F1']:.4f}  EM={agg['EM']:.4f}  "
          f"chrF++={agg['chrF++']:.4f}  CER={agg['CER']:.4f}  "
          f"NED={agg['NED']:.4f}  Latency={agg['Avg_Latency_ms']:.0f}ms")

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Backend:   {BACKEND_URL}")
    print(f"Model:     {MODEL_NAME}")
    print(f"Benchmark: {BENCHMARK_PATH}")
    print(f"Samples:   {N_SAMPLES or 'all'} per language")
    print(f"Delay:     {DELAY}s between requests")

    samples = load_benchmark(BENCHMARK_PATH, N_SAMPLES)
    en_count = sum(1 for s in samples if s["lang"] == "en")
    am_count = sum(1 for s in samples if s["lang"] == "am")
    print(f"Loaded:    {en_count} EN + {am_count} AM = {len(samples)} total\n")

    results = []
    for lang in ["en", "am"]:
        agg = evaluate_lang(samples, lang, BACKEND_URL)
        if agg:
            append_csv(RESULTS_CSV, RESULTS_FIELDS, agg)
            results.append(agg)

    # Summary table
    print(f"\n{'='*75}")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Per-sample CSV:   {SAMPLES_CSV}")
    print(f"{'='*75}")
    if results:
        print(f"\n{'Lang':<6} {'F1':<8} {'EM':<8} {'chrF++':<9} {'CER':<8} {'NED':<8} {'Latency'}")
        print("-" * 60)
        for r in results:
            lang = r["Dataset"].split(":")[-1].upper()
            print(f"{lang:<6} {r['F1']:<8.4f} {r['EM']:<8.4f} {r['chrF++']:<9.4f} "
                  f"{r['CER']:<8.4f} {r['NED']:<8.4f} {r['Avg_Latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
