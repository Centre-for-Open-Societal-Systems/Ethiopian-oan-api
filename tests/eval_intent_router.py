"""
Intent Router Live Evaluation — price, livestock, marketplace queries (EN + AM)

Run inside the app container:
  docker exec oan_app python3 /app/tests/eval_intent_router.py
"""
import json, time, urllib.request, urllib.error, os

BASE = "http://localhost:8000/api"
# Unique run prefix so each eval run gets fresh sessions (no Redis history bleed)
_RUN_ID = os.urandom(4).hex()

QUERIES = [
    # label, lang, query
    # ── CROP PRICE ──────────────────────────────────────────────────────────
    ("EN crop+market (full match)",   "en", "What is the price of teff in Adama?"),
    ("EN crop+market (full match)",   "en", "How much does wheat cost in Merkato today?"),
    ("EN crop+market (full match)",   "en", "What is the price of maize in Bishoftu?"),
    ("EN crop only (partial)",        "en", "What is the current price of sorghum?"),
    ("EN market only (partial)",      "en", "What is selling at Jimma Town market?"),

    # ── LIVESTOCK PRICE ─────────────────────────────────────────────────────
    ("EN livestock+market (full)",    "en", "What is the price of oxen in Adama?"),
    ("EN livestock+market (full)",    "en", "How much does a goat cost in Miyo?"),
    ("EN livestock+market (full)",    "en", "What is the price of cattle in Merkato?"),
    ("EN livestock only (partial)",   "en", "What is the current price of sheep?"),

    # ── MARKETPLACE LISTING ─────────────────────────────────────────────────
    ("EN all marketplaces",           "en", "Show me all active crop marketplaces"),
    ("EN livestock marketplaces",     "en", "List all livestock markets"),

    # ── AMHARIC CROP PRICE ──────────────────────────────────────────────────
    ("AM crop+market (full match)",   "am", "የጤፍ ዋጋ በአዳማ ምን ያህል ነው?"),
    ("AM crop+market (full match)",   "am", "በቢሾፍቱ ገበያ የስንዴ ዋጋ ስንት ነው?"),
    ("AM crop+market (full match)",   "am", "በጅማ የበቆሎ ዋጋ ምን ያህል ነው?"),
    ("AM crop only (partial)",        "am", "የጤፍ ዋጋ ምን ያህል ነው?"),

    # ── AMHARIC LIVESTOCK ───────────────────────────────────────────────────
    ("AM livestock+market (full)",    "am", "በአዳማ የበሬ ዋጋ ምን ያህል ነው?"),
    ("AM livestock+market (full)",    "am", "በሜኢሶ የፍየል ዋጋ ምን ያህል ነው?"),
    ("AM livestock only (partial)",   "am", "የፍየል ዋጋ ስንት ነው?"),
    ("AM livestock only (partial)",   "am", "የበግ ዋጋ ምን ያህል ነው?"),

    # ── AMHARIC MARKETPLACE LISTING ─────────────────────────────────────────
    ("AM all marketplaces",           "am", "ሁሉንም ንቁ ገበያዎች አሳየኝ"),
    ("AM livestock marketplaces",     "am", "ሁሉንም የከብት ገበያዎች ዝርዝር"),
    ("AM crop listing at market",     "am", "የቢሾፍቱ የሰብል ዝርዝር አሳየኝ"),

    # ── EXPECTED LLM FALLTHROUGH ────────────────────────────────────────────
    ("EN general (expect LLM)",       "en", "How do I prevent wheat rust disease?"),
    ("EN greeting (expect LLM)",      "en", "Hello, what can you help me with?"),
    ("AM general (expect LLM)",       "am", "ስንዴ ዝገት በሽታ እንዴት ልቀነስ?"),
]

def chat(query, lang, timeout=20):
    payload = json.dumps({
        "query": query,
        "session_id": f"eval-{_RUN_ID}-{abs(hash(query)) % 9999}",
        "source_lang": lang,
        "target_lang": lang,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode()
        lines = [l for l in raw.splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else {}, resp.status
    except urllib.error.HTTPError as e:
        try: return json.loads(e.read()), e.code
        except: return {"error": str(e)}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def fmt_row(label, query, lang, r, code, ms):
    metrics  = r.get("metrics", {}) if isinstance(r, dict) else {}
    path     = metrics.get("path", "llm")
    intent   = metrics.get("intent", "—")
    decision = metrics.get("decision", "—")
    r_ms     = metrics.get("router_time", "—")
    response = (r.get("response") or r.get("error") or "") if isinstance(r, dict) else str(r)

    icon = "⚡" if path == "fast" else "🔁"
    print(f"\n  [{label}]")
    print(f"  Q : {query}")
    print(f"  {icon} path={path:<4} | intent={intent:<20} | decision={decision} | router={r_ms}ms | wall={ms}ms")
    print(f"  A : {response[:200]}")


print("\n" + "=" * 70)
print("  INTENT ROUTER LIVE EVALUATION")
print("=" * 70)

fast = slow = errors = 0
results = []

for label, lang, query in QUERIES:
    t0 = time.time()
    r, code = chat(query, lang)
    ms = round((time.time() - t0) * 1000)
    fmt_row(label, query, lang, r, code, ms)

    path = r.get("metrics", {}).get("path", "llm") if isinstance(r, dict) else "llm"
    if code == 0 or (isinstance(r, dict) and r.get("error")):
        errors += 1
        results.append(("ERROR", label, query))
    elif path == "fast":
        fast += 1
        results.append(("FAST", label, query))
    else:
        slow += 1
        results.append(("LLM", label, query))

print("\n" + "=" * 70)
print(f"  RESULTS: {fast} fast-path  |  {slow} LLM fallback  |  {errors} errors")
print(f"  Total queries: {len(QUERIES)}")
print("=" * 70)

print("\n  Fast-path hits:")
for res, lbl, q in results:
    if res == "FAST":
        print(f"    ✅  {lbl}: {q[:60]}")

print("\n  LLM fallback:")
for res, lbl, q in results:
    if res == "LLM":
        print(f"    🔁  {lbl}: {q[:60]}")

if errors:
    print("\n  Errors:")
    for res, lbl, q in results:
        if res == "ERROR":
            print(f"    ❌  {lbl}: {q[:60]}")

print()
