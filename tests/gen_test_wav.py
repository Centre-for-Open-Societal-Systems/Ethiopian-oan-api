"""Generate a test WAV file using MMS-TTS, then send to faster-whisper for transcription."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import VitsModel, AutoTokenizer
import torch, numpy as np, wave

print("Loading MMS-TTS model...")
tokenizer = AutoTokenizer.from_pretrained('facebook/mms-tts-eng')
model = VitsModel.from_pretrained('facebook/mms-tts-eng')
model.eval()

print("Synthesizing...")
inputs = tokenizer('What is the price of teff in Adama?', return_tensors='pt')
with torch.no_grad():
    waveform = model(**inputs).waveform[0].numpy()

peak = np.max(np.abs(waveform))
if peak > 0:
    waveform = waveform / peak * 0.9

audio_int16 = (waveform * 32767).astype(np.int16)
out_path = os.path.join(os.path.dirname(__file__), 'test_query.wav')
with wave.open(out_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(audio_int16.tobytes())

print(f"Saved: {out_path} ({len(audio_int16)} samples, {len(audio_int16)/16000:.2f}s)")

# Now test faster-whisper
print("\nSending to faster-whisper...")
import httpx
with open(out_path, 'rb') as f:
    wav_bytes = f.read()

resp = httpx.post('http://localhost:8010/v1/audio/transcriptions',
    files={'file': ('audio.wav', wav_bytes, 'audio/wav')},
    data={'model': 'Systran/faster-whisper-medium', 'language': 'en'},
    timeout=30.0)
print(f"STT response: {resp.json()}")
