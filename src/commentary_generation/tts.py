import os
import tempfile

import soundfile as sf

TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "weights/tts/hf_weights")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "Ryan")


class TTSGenerator:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            self._model = Qwen3TTSModel.from_pretrained(
                TTS_MODEL_PATH,
                device_map=device,
            )

    def generate_wav(self, text: str) -> str:
        """Generate speech for text, write to a temp WAV file, and return its path.

        The caller is responsible for deleting the file when done.
        """
        self._load()
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language="English",
            speaker=TTS_SPEAKER,
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, wavs[0], sr)
        return tmp.name
