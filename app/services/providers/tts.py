"""
Production-Ready TTS Provider — MMS-TTS (HuggingFace VITS)

Env-configurable: swap models by changing TTS_MODEL_AM / TTS_MODEL_EN.
"""

import os
import json
import asyncio
import struct
from typing import AsyncGenerator, Optional
from helpers.utils import get_logger
from helpers.langfuse_client import observe, update_current_observation
import re

logger = get_logger(__name__)


def convert_numbers_to_words(text: str, lang: str) -> str:
    """
    Convert numbers in text to words for better TTS pronunciation.

    Args:
        text: Text containing numbers
        lang: Language code ('en' or 'am')

    Returns:
        Text with numbers converted to words
    """
    if lang == 'en':
        # English number conversion (basic implementation)
        def num_to_words_en(n):
            if n == 0:
                return 'zero'

            ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
                    'sixteen', 'seventeen', 'eighteen', 'nineteen']
            tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

            if n < 10:
                return ones[n]
            elif n < 20:
                return teens[n - 10]
            elif n < 100:
                return tens[n // 10] + ('' if n % 10 == 0 else ' ' + ones[n % 10])
            elif n < 1000:
                return ones[n // 100] + ' hundred' + ('' if n % 100 == 0 else ' ' + num_to_words_en(n % 100))
            elif n < 1000000:
                return num_to_words_en(n // 1000) + ' thousand' + ('' if n % 1000 == 0 else ' ' + num_to_words_en(n % 1000))
            else:
                return str(n)  # Fallback for very large numbers

        # Replace numbers with words
        def replace_num(match):
            num_str = match.group(0).replace(',', '')
            try:
                num = int(num_str)
                return num_to_words_en(num)
            except:
                return match.group(0)

        text = re.sub(r'\b\d{1,3}(?:,\d{3})*\b', replace_num, text)

    elif lang == 'am':
        # Amharic number conversion
        def num_to_words_am(n):
            if n == 0:
                return 'ዜሮ'

            ones = ['', 'አንድ', 'ሁለት', 'ሦስት', 'አራት', 'አምስት', 'ስድስት', 'ሰባት', 'ስምንት', 'ዘጠኝ']
            tens = ['', 'አስር', 'ሃያ', 'ሰላሳ', 'አርባ', 'ሃምሳ', 'ስልሳ', 'ሰባ', 'ሰማንያ', 'ዘጠና']

            if n < 10:
                return ones[n]
            elif n == 10:
                return 'አስር'
            elif n < 20:
                return 'አስራ ' + ones[n - 10]
            elif n < 100:
                return tens[n // 10] + ('' if n % 10 == 0 else ' ' + ones[n % 10])
            elif n < 1000:
                return ones[n // 100] + ' መቶ' + ('' if n % 100 == 0 else ' ' + num_to_words_am(n % 100))
            elif n < 1000000:
                return num_to_words_am(n // 1000) + ' ሺህ' + ('' if n % 1000 == 0 else ' ' + num_to_words_am(n % 1000))
            else:
                return str(n)  # Fallback

        # Replace numbers with Amharic words
        def replace_num(match):
            num_str = match.group(0).replace(',', '')
            try:
                num = int(num_str)
                return num_to_words_am(num)
            except:
                return match.group(0)

        text = re.sub(r'\b\d{1,3}(?:,\d{3})*\b', replace_num, text)

    return text


class TTSProvider:
    """Abstract base class for TTS providers"""

    async def preload(self, langs: list[str]) -> None:
        """Pre-load models/tokenizers for the given language codes."""
        pass

    async def stream_audio(
        self,
        text_stream: AsyncGenerator[str, None],
        lang: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError


class MMSTTSProvider(TTSProvider):
    """TTS using facebook/mms-tts VITS models (local inference via HuggingFace transformers).

    Fully env-configurable: change TTS_MODEL_AM / TTS_MODEL_EN to swap models.
    """

    def __init__(self):
        self.model_map = {
            "am": os.getenv("TTS_MODEL_AM", "facebook/mms-tts-amh"),
            "en": os.getenv("TTS_MODEL_EN", "facebook/mms-tts-eng"),
        }
        self.native_sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "16000"))
        self.target_sample_rate = 24000  # pipeline standard
        self.device = os.getenv("TTS_DEVICE", "cpu")

        self._models = {}       # lang -> VitsModel
        self._tokenizers = {}   # lang -> AutoTokenizer
        self._resamplers = {}   # (native, target) -> Resample transform
        self._lock = asyncio.Lock()

        logger.info(
            f"MMSTTSProvider initialized: models={self.model_map}, "
            f"native_sr={self.native_sample_rate}, target_sr={self.target_sample_rate}, "
            f"device={self.device}"
        )

    async def _load_model(self, lang: str):
        """Lazy-load model + tokenizer for a language. Thread-safe via asyncio.Lock."""
        if lang in self._models:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if lang in self._models:
                return

            model_id = self.model_map.get(lang)
            if not model_id:
                raise ValueError(f"No TTS model configured for language '{lang}'. "
                                 f"Set TTS_MODEL_{lang.upper()} env var.")

            logger.info(f"Loading TTS model for '{lang}': {model_id} (first call — may download ~300-500MB)...")

            import torch
            from transformers import VitsModel, AutoTokenizer

            loop = asyncio.get_event_loop()

            # Load in executor to avoid blocking the event loop
            def _load():
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = VitsModel.from_pretrained(model_id).to(self.device)
                model.eval()
                return model, tokenizer

            model, tokenizer = await loop.run_in_executor(None, _load)
            self._models[lang] = model
            self._tokenizers[lang] = tokenizer
            logger.info(f"TTS model loaded for '{lang}': {model_id} on {self.device}")

    def _get_resampler(self):
        """Get or create a resampler from native to target sample rate."""
        key = (self.native_sample_rate, self.target_sample_rate)
        if key not in self._resamplers:
            import torchaudio
            self._resamplers[key] = torchaudio.transforms.Resample(
                orig_freq=self.native_sample_rate,
                new_freq=self.target_sample_rate,
            )
        return self._resamplers[key]

    def _synthesize_sync(self, text: str, lang: str) -> bytes:
        """Synchronous synthesis: text -> 24kHz int16 PCM bytes."""
        import torch
        import numpy as np

        text = convert_numbers_to_words(text.strip(), lang)
        if not text:
            return b""

        model = self._models[lang]
        tokenizer = self._tokenizers[lang]

        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = model(**inputs)

        waveform = output.waveform[0]  # shape: (samples,)

        # Resample 16kHz -> 24kHz if needed
        if self.native_sample_rate != self.target_sample_rate:
            resampler = self._get_resampler()
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        # float32 -> int16 PCM bytes
        audio_np = waveform.cpu().numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @observe(name="tts.mms", as_type="generation")
    async def _synthesize(self, text: str, lang: str) -> bytes:
        """Async synthesis: ensures model is loaded, then runs inference in executor."""
        await self._load_model(lang)
        update_current_observation(
            input=text,
            metadata={"provider": "mms_tts", "lang": lang, "model": self.model_map.get(lang)},
        )
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, self._synthesize_sync, text, lang)
        update_current_observation(metadata={"pcm_bytes": len(pcm)})
        return pcm

    async def preload(self, langs: list[str]) -> None:
        for lang in langs:
            await self._load_model(lang)

    async def _synthesize_chunk(self, text: str, lang: str) -> bytes:
        """Async wrapper for pipeline.py compatibility (lines 1017, 1191)."""
        return await self._synthesize(text, lang)

    async def stream_audio(
        self,
        text_stream: AsyncGenerator[str, None],
        lang: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        """Buffer sentences from text_stream, synthesize each, yield 4096-byte PCM chunks."""
        lang_code = "en" if lang.startswith("en") else "am"
        await self._load_model(lang_code)

        buffer = ""
        delimiters = {".", "!", "?", ";", "\n", ","}
        chunk_size = 4096

        async def yield_pcm(text: str):
            pcm = await self._synthesize(text, lang_code)
            if not pcm:
                return
            offset = 0
            while offset < len(pcm):
                yield pcm[offset:offset + chunk_size]
                offset += chunk_size

        async for text_chunk in text_stream:
            if not text_chunk:
                continue
            buffer += text_chunk
            if any(c in delimiters for c in text_chunk):
                split_idx = max((i for i, c in enumerate(buffer) if c in delimiters), default=-1)
                if split_idx != -1:
                    to_synth = buffer[:split_idx + 1].strip()
                    buffer = buffer[split_idx + 1:].strip()
                    if to_synth:
                        async for audio_bytes in yield_pcm(to_synth):
                            yield audio_bytes
            elif len(buffer) > 80:
                async for audio_bytes in yield_pcm(buffer):
                    yield audio_bytes
                buffer = ""

        if buffer.strip():
            async for audio_bytes in yield_pcm(buffer.strip()):
                yield audio_bytes

    def cleanup(self):
        """Free model memory."""
        for lang, model in self._models.items():
            del model
        self._models.clear()
        self._tokenizers.clear()
        self._resamplers.clear()
        logger.info("MMSTTSProvider: models cleaned up")


class TritonMMSTTSProvider(TTSProvider):
    """TTS provider that routes MMS and Piper voices to a Triton Inference Server.

    - MMS family (mms-tts-eng / -amh / -orm): 16kHz output, IO = input_ids +
      attention_mask → waveform. HuggingFace VitsTokenizer is used locally.
    - Piper voices (piper-am, …): 22050 Hz output, IO = input (phoneme IDs) +
      input_lengths + scales + sid → output. Phonemization runs locally via the
      espeak-ng binary and a phoneme_id_map loaded from the voice's .onnx.json.

    Env vars:
        TRITON_TTS_URL      — Triton base URL, e.g. http://13.232.101.17:8000
        TRITON_TTS_MODEL_AM — Triton model name for Amharic (default: mms-tts-amh)
        TRITON_TTS_MODEL_EN — Triton model name for English (default: mms-tts-eng)
    """

    # lang_code → (default Triton model name, HF tokenizer id) for MMS-TTS family
    MMS_LANGS = {
        "en": ("mms-tts-eng", "facebook/mms-tts-eng"),
        "am": ("mms-tts-amh", "facebook/mms-tts-amh"),
        "om": ("mms-tts-orm", "facebook/mms-tts-orm"),
    }
    # Triton model name → HF tokenizer id (for explicit model overrides)
    MODEL_TO_TOKENIZER = {
        "mms-tts-eng": "facebook/mms-tts-eng",
        "mms-tts-amh": "facebook/mms-tts-amh",
        "mms-tts-orm": "facebook/mms-tts-orm",
    }
    # Piper voices: Triton model name → path to .onnx.json with phoneme/speaker maps
    PIPER_CONFIG_PATHS = {
        "piper-am": "assets/piper/am/am_ET-l_geez-medium.onnx.json",
    }
    PIPER_NATIVE_SAMPLE_RATE = 22050

    def __init__(self):
        self.triton_url = os.getenv("TRITON_TTS_URL", "").rstrip("/").replace("http://", "")
        # Per-lang Triton model name, overridable via env (TRITON_TTS_MODEL kept for back-compat).
        self.triton_models = {
            "en": os.getenv("TRITON_TTS_MODEL_EN", "mms-tts-eng"),
            "am": os.getenv(
                "TRITON_TTS_MODEL_AM", os.getenv("TRITON_TTS_MODEL", "mms-tts-amh")
            ),
            "om": os.getenv("TRITON_TTS_MODEL_OM", "mms-tts-orm"),
        }
        self.native_sample_rate = 16000
        self.target_sample_rate = 24000

        self._tokenizers: dict[str, object] = {}  # HF tokenizer id → tokenizer
        self._piper_configs: dict[str, dict] = {}  # model_name → loaded JSON
        self._resampler = None
        self._piper_resampler = None
        self._lock = asyncio.Lock()

        logger.info(
            f"TritonMMSTTSProvider initialized: triton={self.triton_url}, "
            f"models={self.triton_models}"
        )

    async def _load_tokenizer(self, model_name: str):
        """Load the MMS tokenizer for the given Triton model name. Idempotent."""
        tok_id = self.MODEL_TO_TOKENIZER.get(model_name)
        if tok_id is None:
            raise ValueError(f"No tokenizer mapping for model '{model_name}'")
        if tok_id in self._tokenizers:
            return
        async with self._lock:
            if tok_id in self._tokenizers:
                return
            from transformers import VitsTokenizer
            loop = asyncio.get_event_loop()
            tokenizer = await loop.run_in_executor(
                None, VitsTokenizer.from_pretrained, tok_id
            )
            self._tokenizers[tok_id] = tokenizer
            logger.info(f"MMS tokenizer loaded: {tok_id}")

    def _get_resampler(self):
        if self._resampler is None:
            import torchaudio
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=self.native_sample_rate,
                new_freq=self.target_sample_rate,
            )
        return self._resampler

    def _waveform_to_pcm(self, waveform_f32) -> bytes:
        """float32 waveform (numpy, 16kHz) → int16 PCM bytes at 24kHz."""
        import torch
        import numpy as np

        t = torch.from_numpy(waveform_f32).float()
        resampler = self._get_resampler()
        t = resampler(t.unsqueeze(0)).squeeze(0)
        audio_np = t.numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        return (audio_np * 32767).astype(np.int16).tobytes()

    def _triton_infer_sync(self, text: str, model_name: str) -> bytes:
        """Tokenize locally, run inference on Triton, return int16 PCM at 24kHz."""
        import numpy as np
        import tritonclient.http as httpclient

        tok_id = self.MODEL_TO_TOKENIZER[model_name]
        tokenizer = self._tokenizers[tok_id]

        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        client = httpclient.InferenceServerClient(url=self.triton_url)

        infer_inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        infer_inputs[0].set_data_from_numpy(input_ids)
        infer_inputs[1].set_data_from_numpy(attention_mask)

        infer_outputs = [httpclient.InferRequestedOutput("waveform")]
        response = client.infer(model_name, infer_inputs, outputs=infer_outputs)
        waveform = response.as_numpy("waveform").squeeze()  # float32 [samples] at 16kHz
        return self._waveform_to_pcm(waveform)

    def _get_piper_resampler(self):
        if self._piper_resampler is None:
            import torchaudio
            self._piper_resampler = torchaudio.transforms.Resample(
                orig_freq=self.PIPER_NATIVE_SAMPLE_RATE,
                new_freq=self.target_sample_rate,
            )
        return self._piper_resampler

    def _load_piper_config(self, model_name: str) -> dict:
        """Load + cache the .onnx.json (phoneme_id_map, speaker_id_map, inference scales)."""
        if model_name in self._piper_configs:
            return self._piper_configs[model_name]
        path = self.PIPER_CONFIG_PATHS.get(model_name)
        if not path:
            raise ValueError(f"No Piper voice config registered for model '{model_name}'")
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self._piper_configs[model_name] = cfg
        logger.info(
            f"Piper voice config loaded: {model_name} from {path} "
            f"(speakers={cfg.get('num_speakers')}, "
            f"sr={cfg.get('audio', {}).get('sample_rate')})"
        )
        return cfg

    @staticmethod
    def _piper_phonemize(text: str, voice: str) -> str:
        """Run espeak-ng to phonemize text into IPA. Returns a single string of
        IPA characters (sentence boundaries flattened to a single space)."""
        import subprocess
        try:
            result = subprocess.run(
                ["espeak-ng", "-q", "--ipa=3", "-v", voice, text],
                capture_output=True, text=True, timeout=15,
            )
        except FileNotFoundError as e:
            raise RuntimeError("espeak-ng binary not found; install with `brew install espeak-ng`") from e
        if result.returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {result.stderr.strip()}")
        return " ".join(line.strip() for line in result.stdout.splitlines() if line.strip())

    def _piper_text_to_ids(self, text: str, cfg: dict) -> list[int]:
        """Phonemize and encode per Piper convention:
        [BOS, p1, PAD, p2, PAD, …, pN, PAD, EOS]."""
        voice = cfg.get("espeak", {}).get("voice", "am")
        pim = cfg["phoneme_id_map"]
        BOS = pim.get("^", [1])
        EOS = pim.get("$", [2])
        PAD = pim.get("_", [0])
        phonemes = self._piper_phonemize(text, voice)
        ids: list[int] = list(BOS)
        skipped = 0
        for ch in phonemes:
            tok = pim.get(ch)
            if tok is None:
                skipped += 1
                continue
            ids.extend(tok)
            ids.extend(PAD)
        ids.extend(EOS)
        if skipped:
            logger.debug(f"Piper phonemize: skipped {skipped} unknown phoneme chars")
        return ids

    def _piper_waveform_to_pcm(self, waveform_f32) -> bytes:
        """FP32 waveform at 22050 Hz → int16 PCM bytes at target_sample_rate (24kHz)."""
        import torch, numpy as np
        t = torch.from_numpy(waveform_f32).float()
        resampler = self._get_piper_resampler()
        t = resampler(t.unsqueeze(0)).squeeze(0)
        audio_np = t.numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        return (audio_np * 32767).astype(np.int16).tobytes()

    def _piper_infer_sync(self, text: str, model_name: str, speaker_id: int) -> bytes:
        """Phonemize locally, call Triton, return int16 PCM at target_sample_rate."""
        import numpy as np
        import tritonclient.http as httpclient

        cfg = self._piper_configs[model_name]  # already loaded by caller
        ids = self._piper_text_to_ids(text, cfg)
        if len(ids) <= 2:  # only BOS+EOS — empty after skipping
            return b""

        input_ids = np.array([ids], dtype=np.int64)
        input_lengths = np.array([len(ids)], dtype=np.int64)
        inf = cfg.get("inference", {})
        scales = np.array(
            [inf.get("noise_scale", 0.667),
             inf.get("length_scale", 1.0),
             inf.get("noise_w", 0.8)],
            dtype=np.float32,
        )
        sid = np.array([int(speaker_id)], dtype=np.int64)

        client = httpclient.InferenceServerClient(url=self.triton_url)
        infer_inputs = [
            httpclient.InferInput("input", input_ids.shape, "INT64"),
            httpclient.InferInput("input_lengths", input_lengths.shape, "INT64"),
            httpclient.InferInput("scales", scales.shape, "FP32"),
            httpclient.InferInput("sid", sid.shape, "INT64"),
        ]
        infer_inputs[0].set_data_from_numpy(input_ids)
        infer_inputs[1].set_data_from_numpy(input_lengths)
        infer_inputs[2].set_data_from_numpy(scales)
        infer_inputs[3].set_data_from_numpy(sid)

        infer_outputs = [httpclient.InferRequestedOutput("output")]
        response = client.infer(model_name, infer_inputs, outputs=infer_outputs)

        waveform = response.as_numpy("output").reshape(-1)  # FP32 [samples] @ 22050 Hz
        return self._piper_waveform_to_pcm(waveform)

    def _resolve_model(self, lang: str, model: Optional[str] = None) -> str:
        """Pick the Triton model name. Explicit `model` wins; else map by lang."""
        if model:
            if model not in self.MODEL_TO_TOKENIZER:
                raise ValueError(f"Unsupported TTS model '{model}' on TritonMMSTTSProvider")
            return model
        key = "am" if lang == "am" else ("om" if lang == "om" else "en")
        return self.triton_models[key]

    async def preload(self, langs: list[str]) -> None:
        for lang in langs:
            key = "am" if lang == "am" else ("om" if lang == "om" else "en")
            await self._load_tokenizer(self.triton_models[key])

    @observe(name="tts.triton", as_type="generation")
    async def _synthesize(self, text: str, lang: str,
                          model: Optional[str] = None,
                          speaker_id: Optional[int] = None) -> bytes:
        text = convert_numbers_to_words(text.strip(), lang)
        if not text:
            return b""

        # Piper voices use a different IO contract (phoneme IDs + sid + scales)
        # and ship a per-voice config JSON with the phoneme/speaker maps.
        if model and model.startswith("piper"):
            update_current_observation(
                input=text,
                metadata={"provider": "triton_piper", "lang": lang,
                          "model": model, "speaker_id": speaker_id or 0},
            )
            self._load_piper_config(model)
            sid = speaker_id if speaker_id is not None else 0
            loop = asyncio.get_event_loop()
            pcm = await loop.run_in_executor(
                None, self._piper_infer_sync, text, model, sid
            )
            update_current_observation(metadata={"pcm_bytes": len(pcm)})
            return pcm

        # MMS-TTS path (mms-tts-eng / -amh / -orm)
        model_name = self._resolve_model(lang, model)

        update_current_observation(
            input=text,
            metadata={"provider": "triton_mms_tts", "lang": lang, "model": model_name},
        )

        await self._load_tokenizer(model_name)
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, self._triton_infer_sync, text, model_name)

        update_current_observation(metadata={"pcm_bytes": len(pcm)})
        return pcm

    async def _synthesize_chunk(self, text: str, lang: str) -> bytes:
        return await self._synthesize(text, lang)

    async def stream_audio(
        self,
        text_stream: AsyncGenerator[str, None],
        lang: str = "en"
    ) -> AsyncGenerator[bytes, None]:
        lang_code = "am" if lang.startswith("am") else ("om" if lang.startswith("om") else "en")
        await self._load_tokenizer(self.triton_models[lang_code])

        buffer = ""
        delimiters = {".", "!", "?", ";", "\n", ","}
        chunk_size = 4096

        async def yield_pcm(text: str):
            pcm = await self._synthesize(text, lang_code)
            if not pcm:
                return
            offset = 0
            while offset < len(pcm):
                yield pcm[offset:offset + chunk_size]
                offset += chunk_size

        async for text_chunk in text_stream:
            if not text_chunk:
                continue
            buffer += text_chunk
            if any(c in delimiters for c in text_chunk):
                split_idx = max((i for i, c in enumerate(buffer) if c in delimiters), default=-1)
                if split_idx != -1:
                    to_synth = buffer[:split_idx + 1].strip()
                    buffer = buffer[split_idx + 1:].strip()
                    if to_synth:
                        async for audio_bytes in yield_pcm(to_synth):
                            yield audio_bytes
            elif len(buffer) > 80:
                async for audio_bytes in yield_pcm(buffer):
                    yield audio_bytes
                buffer = ""

        if buffer.strip():
            async for audio_bytes in yield_pcm(buffer.strip()):
                yield audio_bytes

    def cleanup(self):
        self._tokenizers.clear()
        self._piper_configs.clear()
        self._resampler = None
        self._piper_resampler = None
        logger.info("TritonMMSTTSProvider: cleaned up")


# Singleton
_tts_provider: Optional[TTSProvider] = None


def get_tts_provider() -> TTSProvider:
    """Get the TTS provider singleton.

    Uses TritonMMSTTSProvider when TRITON_TTS_URL is set,
    otherwise falls back to local MMSTTSProvider.
    """
    global _tts_provider
    if _tts_provider is None:
        triton_url = os.getenv("TRITON_TTS_URL", "")
        if triton_url:
            _tts_provider = TritonMMSTTSProvider()
            logger.info("TTS Provider initialized: triton_mms_tts")
        else:
            _tts_provider = MMSTTSProvider()
            logger.info("TTS Provider initialized: mms_tts")

    return _tts_provider


def cleanup_tts_provider():
    """Cleanup TTS provider resources"""
    global _tts_provider
    if _tts_provider is not None and hasattr(_tts_provider, 'cleanup'):
        _tts_provider.cleanup()
        logger.info("TTS Provider cleaned up")
