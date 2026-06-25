"""
Quick test for the voice pipeline WebSocket endpoint.
Synthesizes a query via MMS-TTS, sends it as audio, and prints responses.
"""
import asyncio
import json
import numpy as np
import struct
import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

async def test_voice_pipeline(query_text: str = "What is the price of teff in Adama?", lang: str = "en"):
    import websockets

    # Step 1: Generate audio from text using MMS-TTS
    print(f"[1/4] Synthesizing query: '{query_text}'")
    from transformers import VitsModel, AutoTokenizer
    import torch

    model_id = "facebook/mms-tts-eng" if lang == "en" else "facebook/mms-tts-amh"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer(query_text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform[0].numpy()  # float32, 16kHz
    # Normalize to use full dynamic range (TTS output can be quiet)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.9  # normalize to 90% of full scale
    print(f"    Generated {len(waveform)} samples ({len(waveform)/16000:.2f}s) at 16kHz (peak normalized)")

    # Step 2: Connect to WebSocket
    ws_url = f"ws://localhost:8000/api/conv/ws?lang={lang}"
    print(f"[2/4] Connecting to {ws_url}")

    async with websockets.connect(ws_url) as ws:
        print("    Connected!")

        # Step 3: Send audio in chunks (512 samples = 32ms at 16kHz, as float32)
        print(f"[3/4] Streaming audio...")
        chunk_size = 512
        total_chunks = 0

        # Add small silence at start (0.3s)
        silence = np.zeros(int(16000 * 0.3), dtype=np.float32)
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i:i + chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.032)

        # Send actual audio
        for i in range(0, len(waveform), chunk_size):
            chunk = waveform[i:i + chunk_size].astype(np.float32)
            await ws.send(chunk.tobytes())
            total_chunks += 1
            await asyncio.sleep(0.032)  # ~32ms between chunks (real-time pacing)

        # Add silence at end (1.5s) to trigger VAD stop
        silence_end = np.zeros(int(16000 * 1.5), dtype=np.float32)
        for i in range(0, len(silence_end), chunk_size):
            chunk = silence_end[i:i + chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.032)

        print(f"    Sent {total_chunks} audio chunks")

        # Step 4: Listen for responses
        print(f"[4/4] Waiting for responses...")
        start = time.time()
        timeout = 30  # seconds

        try:
            while time.time() - start < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    if isinstance(msg, bytes):
                        print(f"    🔊 Audio chunk received: {len(msg)} bytes")
                    else:
                        data = json.loads(msg)
                        msg_type = data.get("type", "unknown")
                        if msg_type == "speech_start":
                            print("    🟢 Speech detected")
                        elif msg_type == "speech_end":
                            print("    🔴 Speech ended")
                        elif msg_type == "transcription":
                            print(f"    📝 Transcript: '{data.get('text', '')}' (final={data.get('is_final')})")
                        elif msg_type == "llm_chunk":
                            print(f"    💬 Response: {data.get('text', '')[:100]}...")
                        elif msg_type == "metrics":
                            metrics = data.get("data", {})
                            print(f"    📊 Metrics: e2e={metrics.get('e2e_latency')}ms, "
                                  f"stt={metrics.get('stt_duration')}ms, "
                                  f"tts={metrics.get('tts_synthesis')}ms")
                            break  # Metrics come last, we're done
                        else:
                            print(f"    📨 {msg_type}: {data}")
                except asyncio.TimeoutError:
                    print("    ⏳ No message for 5s, continuing to wait...")
        except Exception as e:
            print(f"    ⚠️  Receive error: {e}")

        elapsed = time.time() - start
        print(f"\nDone! Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the price of teff in Adama?"
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"
    asyncio.run(test_voice_pipeline(query, lang))
