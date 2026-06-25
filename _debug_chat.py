"""Debug script to call FastGeminiService directly and see the actual error."""
import asyncio
import traceback
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def main():
    from app.services.fast_gemini import FastGeminiService

    svc = FastGeminiService(lang="en")
    print("FastGeminiService initialized OK")
    print(f"Model: {svc.model}")
    print("Sending test message...")

    metrics = {}
    full_text = ""
    try:
        async for chunk in svc.generate_response("Hello, what can you help me with?", metrics):
            print(f"CHUNK: {repr(chunk)}")
            full_text += chunk
    except Exception as e:
        print(f"\nEXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()

    print(f"\nFull response: {full_text}")
    print(f"Metrics: {metrics}")

asyncio.run(main())
