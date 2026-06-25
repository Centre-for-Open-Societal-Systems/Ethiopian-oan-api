"""
Production-Ready TTS Provider — MMS-TTS (HuggingFace VITS)

Env-configurable: swap models by changing TTS_MODEL_AM / TTS_MODEL_EN.
"""

import os
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


# Singleton
_tts_provider: Optional[TTSProvider] = None


def get_tts_provider() -> TTSProvider:
    """Get the MMS-TTS provider (singleton)."""
    global _tts_provider
    if _tts_provider is None:
        _tts_provider = MMSTTSProvider()
        logger.info("TTS Provider initialized: mms_tts")

    return _tts_provider


def cleanup_tts_provider():
    """Cleanup TTS provider resources"""
    global _tts_provider
    if _tts_provider is not None and hasattr(_tts_provider, 'cleanup'):
        _tts_provider.cleanup()
        logger.info("TTS Provider cleaned up")
