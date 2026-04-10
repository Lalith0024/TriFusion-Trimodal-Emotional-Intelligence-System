"""
src/text/transcriber.py
───────────────────────
Whisper-based speech-to-text transcription.

Design choices:
  • "tiny" model (~39 M params) runs in ~150 ms on CPU per 3-second chunk —
    fast enough for near-real-time use without GPU.
  • We pass language="en" to skip language detection overhead.
  • fp16=False forces FP32 on CPU (MPS/CUDA users can enable fp16).
  • If transcription produces empty/noise-only output, returns an empty
    string so downstream TextInference returns uniform emotion distribution.
"""

import numpy as np
import whisper
import logging

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Wraps OpenAI Whisper for low-latency speech-to-text.

    Whisper expects float32 waveform at 16 kHz.
    The output text is passed directly to TextInference.predict().
    """

    def __init__(self, model_size: str = "tiny"):
        """
        Args:
            model_size: One of "tiny", "base", "small", "medium", "large".
                        "tiny" is recommended for real-time use on CPU.
        """
        logger.info(f"Loading Whisper '{model_size}' model — this may take a moment...")
        self.model = whisper.load_model(model_size)
        logger.info("Whisper loaded successfully.")

    def transcribe(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe an audio waveform to text.

        Args:
            waveform:    1-D float32 numpy array (values in [-1, 1]).
            sample_rate: Source sampling rate (Whisper resamples internally
                         to 16 kHz if different).

        Returns:
            Stripped transcription string. Empty string if nothing audible.
        """
        # Whisper's transcribe() accepts a raw numpy array directly
        result = self.model.transcribe(
            waveform,
            language="en",
            fp16=False,                  # CPU-safe
            condition_on_previous_text=False   # avoids hallucination carry-over
        )
        text = result.get("text", "").strip()
        logger.debug(f"Transcription: '{text}'")
        return text
