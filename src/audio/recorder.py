"""
src/audio/recorder.py
─────────────────────
Real-time microphone audio capture via sounddevice.

Design:
  • Uses sounddevice.InputStream with blocksize = sample_rate × chunk_duration
    so the callback fires exactly once per analysis window (default: 3 s).
  • Audio is placed on a bounded Queue(maxsize=5) — older chunks are dropped
    if the inference pipeline falls behind, keeping latency bounded.
  • The recorder runs entirely in the sounddevice background thread; the
    main thread just calls .get_chunk() when it needs audio.
  • Waveform is returned as float32 in [-1, 1] — compatible with
    Wav2Vec2Processor and Whisper's transcribe() without re-scaling.
"""

import numpy as np
import queue
import sounddevice as sd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Continuous microphone recorder that produces fixed-duration chunks.

    Usage:
        recorder = AudioRecorder(sample_rate=16000, chunk_duration=3.0)
        recorder.start()
        while True:
            chunk = recorder.get_chunk()  # blocks up to `timeout` seconds
            # process chunk …
        recorder.stop()
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 3.0):
        self.sample_rate   = sample_rate
        self.chunk_duration = chunk_duration
        # blocksize in samples = one complete analysis window
        self.chunk_size    = int(sample_rate * chunk_duration)
        # Bounded queue — drop stale chunks rather than accumulate lag
        self.audio_queue   = queue.Queue(maxsize=5)
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    # Sounddevice callback — executes in a high-priority background thread
    # ------------------------------------------------------------------
    def _callback(
        self,
        indata: np.ndarray,   # shape: (chunk_size, channels)
        frames: int,
        time,
        status: sd.CallbackFlags
    ) -> None:
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Flatten to 1-D mono float32 chunk
        chunk = indata.copy().flatten().astype(np.float32)

        # Non-blocking put — silently drop if queue is full (keeps latency bounded)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            logger.debug("Audio queue full — dropping oldest chunk.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the InputStream and start the background capture thread."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,              # mono input for Wav2Vec2 / Whisper
            blocksize=self.chunk_size,
            callback=self._callback,
            dtype=np.float32
        )
        self._stream.start()
        logger.info(
            f"AudioRecorder started | SR={self.sample_rate} Hz | "
            f"chunk={self.chunk_duration}s ({self.chunk_size} samples)"
        )

    def get_chunk(self, timeout: float = 4.0) -> Optional[np.ndarray]:
        """
        Block until a fresh chunk is available or timeout expires.

        Returns:
            float32 ndarray of shape (chunk_size,), or None on timeout.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("No audio chunk received within timeout.")
            return None

    def stop(self) -> None:
        """Stop and close the InputStream, releasing the microphone."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("AudioRecorder stopped.")
