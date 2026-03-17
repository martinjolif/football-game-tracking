"""Qwen3-TTS model singleton and speech generation."""

import os
import logging

import numpy as np

logger = logging.getLogger(__name__)

_tts_model = None
_tts_sample_rate = None
_tts_loaded = False


def get_tts_model():
    """Lazy-load Qwen3-TTS CustomVoice model. Returns (model, sample_rate) or (None, None)."""
    global _tts_model, _tts_sample_rate, _tts_loaded

    if _tts_loaded:
        return _tts_model, _tts_sample_rate

    _tts_loaded = True

    try:
        import torch
        from qwen_tts import Qwen3TTSModel

        model_name = os.environ.get(
            "QWEN_TTS_MODEL", "weights/tts/hf_weights"
        )

        logger.info("Loading TTS model: %s", model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        _tts_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, #MPS does NOT support bfloat16
        )
        # Probe sample rate with a tiny generation
        _, _tts_sample_rate = _tts_model.generate_custom_voice(
            text="test",
            language="English",
            speaker=os.environ.get("TTS_SPEAKER", "Ryan"),
        )
        logger.info("TTS model loaded, sample_rate=%d", _tts_sample_rate)
    except Exception:
        logger.exception("Failed to load TTS model — TTS will be disabled")
        _tts_model = None
        _tts_sample_rate = None

    return _tts_model, _tts_sample_rate


def generate_tts_audio(text: str) -> tuple[np.ndarray | None, int | None]:
    """Generate speech waveform from text. Returns (waveform_ndarray, sample_rate) or (None, None)."""
    model, sr = get_tts_model()
    if model is None:
        return None, None

    speaker = os.environ.get("TTS_SPEAKER", "Ryan")
    language = os.environ.get("TTS_LANGUAGE", "English")

    try:
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
        return wavs[0], sample_rate
    except Exception:
        logger.exception("TTS generation failed for text: %.80s...", text)
        return None, None
