"""
src/audio/recorder.py
─────────────────────
Real-time microphone capture using sounddevice.
Captures 3-second rolling windows at 16kHz for Wav2Vec2 processing.

Threading model:
  - AudioRecorder runs its own background thread via sd.InputStream callback
  - Frames accumulate in a circular buffer
  - get_chunk() returns the latest 3-second window (non-blocking)
  - If less than 3 seconds of audio available, returns silence
"""

import numpy as np
import threading
import logging
from collections import deque

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  # 48,000 samples


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate    = sample_rate
        self._buffer        = deque(maxlen=CHUNK_SAMPLES * 2)  # 6 seconds circular
        self._lock          = threading.Lock()
        self._stream        = None
        self._active        = False

    def start(self):
        """Start continuous microphone capture."""
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=1600,  # 0.1 second blocks for low latency
                callback=self._callback
            )
            self._stream.start()
            self._active = True
            logger.info("Microphone stream started.")
        except Exception as e:
            logger.warning(f"Microphone unavailable: {e}. Audio will use silence fallback.")
            self._active = False

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called by sounddevice on every block. Appends to circular buffer."""
        if status:
            logger.debug(f"Audio status: {status}")
        with self._lock:
            self._buffer.extend(indata[:, 0].tolist())  # mono channel

    def get_chunk(self) -> np.ndarray:
        """
        Returns the latest CHUNK_SAMPLES samples as a float32 numpy array.
        If not enough audio is buffered, returns silence (zeros).
        Safe to call from any thread.
        """
        with self._lock:
            buf = list(self._buffer)

        if len(buf) >= CHUNK_SAMPLES:
            # Take the most recent 3 seconds
            chunk = np.array(buf[-CHUNK_SAMPLES:], dtype=np.float32)
        else:
            # Pad with silence on the left
            chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            if buf:
                chunk[-len(buf):] = np.array(buf, dtype=np.float32)

        return chunk

    def stop(self):
        """Stop and release microphone stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active
