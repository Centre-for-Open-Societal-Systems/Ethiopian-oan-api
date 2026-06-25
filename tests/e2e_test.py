"""
End-to-end test: Intent Router · LLM (Gemma via OpenRouter) · TTS (Triton) · STT (OmniASR)

Run inside the app container:
  docker exec oan_app python3 /app/e2e_test.py
"""

import json, base64, time
import urllib.request, urllib.error

BASE = "http://localhost:8000/api"
SEP = "=" * 62


def post(path, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read().decode()
        # chat endpoint yields one or more JSON lines (plain, not SSE)
        lines = [l for l in raw.splitlines() if l.strip()]
        if len(lines) == 1:
            return json.loads(lines[0]), resp.status
        return [json.loads(l) for l in lines if l.strip()], resp.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read()), e.code
        except Exception:
            return {"error": str(e)}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def fmt_chat(r, code, elapsed):
    if isinstance(r, list):
        r = r[-1] if r else {}
    if not isinstance(r, dict):
        return f"  [{code}] {elapsed}ms — unexpected: {str(r)[:200]}"
    metrics = r.get("metrics", {})
    path    = metrics.get("path", "llm")
    intent  = metrics.get("intent", "—")
    latency = metrics.get("total_e2e_latency", elapsed)
    router  = metrics.get("router_time", "—")
    response = r.get("response", "")
    out = [
        f"  HTTP {code} | wall {elapsed}ms | e2e {latency}ms | path={path}",
        f"  intent={intent} | router_time={router}ms",
        f"  response: {response[:300]}",
    ]
    if r.get("status") == "blocked":
        out.append(f"  ⚠️  BLOCKED by moderation: {r.get('moderation')}")
    if r.get("error"):
        out.append(f"  ❌  ERROR: {r['error']}")
    return "\n".join(out)


# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 1: Intent Router — English crop price (expect: fast path)")
print(SEP)
t0 = time.time()
r, code = post("/chat/", {
    "query": "What is the price of teff in Addis Ababa?",
    "session_id": "e2e-1", "source_lang": "en", "target_lang": "en",
})
print(fmt_chat(r, code, round((time.time()-t0)*1000)))

# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 2: Intent Router — English weather (expect: fast path)")
print(SEP)
t0 = time.time()
r, code = post("/chat/", {
    "query": "What is the weather like in Addis Ababa today?",
    "session_id": "e2e-2", "source_lang": "en", "target_lang": "en",
}, timeout=30)
print(fmt_chat(r, code, round((time.time()-t0)*1000)))

# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 3: Intent Router — Amharic crop price (expect: fast path)")
print(SEP)
t0 = time.time()
r, code = post("/chat/", {
    "query": "የጤፍ ዋጋ በአዳማ ምን ያህል ነው?",
    "session_id": "e2e-3", "source_lang": "am", "target_lang": "am",
})
print(fmt_chat(r, code, round((time.time()-t0)*1000)))

# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 4: LLM (Gemma/OpenRouter) — general ag question (expect: llm path)")
print(SEP)
t0 = time.time()
r, code = post("/chat/", {
    "query": "What are the best practices to prevent wheat rust disease in Ethiopia?",
    "session_id": "e2e-4", "source_lang": "en", "target_lang": "en",
}, timeout=90)
print(fmt_chat(r, code, round((time.time()-t0)*1000)))

# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 5: TTS — English  (Triton → mms-tts-eng)")
print(SEP)
t0 = time.time()
r, code = post("/tts/", {
    "text": "The price of teff in Addis Ababa is fifty birr per kilogram.",
    "lang_code": "en", "session_id": "e2e-tts",
}, timeout=60)
elapsed = round((time.time()-t0)*1000)
if code == 200 and isinstance(r, dict) and r.get("audio_content"):
    ab = base64.b64decode(r["audio_content"])
    print(f"  HTTP {code} | {elapsed}ms | {len(ab):,} bytes | ~{len(ab)/2/24000:.2f}s @ 24kHz PCM  ✅")
else:
    print(f"  HTTP {code} | {elapsed}ms | ERROR: {r}")

# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 6: TTS — Amharic (Triton → mms-tts-amh)")
print(SEP)
t0 = time.time()
r, code = post("/tts/", {
    "text": "ሰላም፣ ዛሬ የጤፍ ዋጋ ምን ያህል ነው?",
    "lang_code": "am", "session_id": "e2e-tts-am",
}, timeout=60)
elapsed = round((time.time()-t0)*1000)
if code == 200 and isinstance(r, dict) and r.get("audio_content"):
    ab = base64.b64decode(r["audio_content"])
    print(f"  HTTP {code} | {elapsed}ms | {len(ab):,} bytes | ~{len(ab)/2/24000:.2f}s @ 24kHz PCM  ✅")
else:
    print(f"  HTTP {code} | {elapsed}ms | ERROR: {r}")

# ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST 7: STT — OmniASR (EC2) with test_query.wav")
print(SEP)
try:
    with open("/app/test_query.wav", "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    r, code = post("/transcribe/", {
        "audio_content": wav_b64, "lang_code": "am", "session_id": "e2e-stt",
    }, timeout=90)
    elapsed = round((time.time()-t0)*1000)
    if code == 200 and isinstance(r, dict):
        print(f"  HTTP {code} | {elapsed}ms  ✅")
        print(f"  Transcript : '{r.get('text', '')}'")
        print(f"  Lang       : {r.get('lang_code')}")
    else:
        print(f"  HTTP {code} | {elapsed}ms | ERROR: {r}")
except FileNotFoundError:
    print("  SKIP: /app/test_query.wav not found")

print(f"\n{SEP}")
print("ALL TESTS DONE")
print(SEP)
