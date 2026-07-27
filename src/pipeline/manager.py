"""
src/pipeline/manager.py
────────────────────────
Decoupled pipeline manager for high-FPS real-time operation.

Architecture:
  - CaptureThread : grabs frames at full camera speed (30+ FPS), zero ML work
  - InferenceThread: consumes frames, runs all three models, writes results
  - Streamlit reads from shared state — always has fresh frame + last results

Root cause of the 10-20 FPS problem (now fixed):
  Previously, VideoCapture.read() and EfficientNet inference ran sequentially
  in the SAME thread.  Each iteration blocked for ~80ms (33ms capture + 50ms
  model) → 12 FPS ceiling.

  Decoupled solution:
    Capture thread  runs at 30+ FPS  (zero ML, just cv2.VideoCapture.read)
    Inference thread runs at ~12 FPS  (all ML work here)
    UI always shows the LATEST raw frame from the capture queue → 30 FPS feel
"""

import threading
import queue
import time
import cv2
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ── Toggle ─────────────────────────────────────────────────────────────────────
# Keep True until all four model weights exist in models/ subdirs.
# Set False ONLY after running all four training scripts successfully.
# Set True ONLY for UI development without model weights
# Set False when models/ folders contain trained checkpoints
import os
IS_CLOUD = "STREAMLIT_SERVER_PORT" in os.environ or os.path.exists("/home/appuser")
SIMULATION_MODE = os.environ.get("SIMULATION_MODE", str(IS_CLOUD)).lower() in ("true", "1", "t")


class PipelineManager:
    """
    Manages two daemon threads and shared state buffers.
    Instantiate once per Streamlit session via st.session_state.
    """

    def __init__(self):
        self.running = False

        self._frame_q   = queue.Queue(maxsize=1)   # BGR frames → inference
        self._display_frame = None                 # RGB frame buffer for UI
        self._display_lock  = threading.Lock()

        # Thread-safe result store — written by inference, read by UI
        self._result_lock = threading.Lock()
        self._result: Dict = self._empty_result()

        # FPS tracking (capture thread updates these atomically)
        self._fps_lock    = threading.Lock()
        self._fps         = 0.0
        self._frame_count = 0

        # Threads (initialised in start())
        self._capture_thread   = None
        self._inference_thread = None

        # Last transcribed text — updated every 5th inference frame
        self._last_transcribed = ""

        # Load heavy ML models ONCE at construction time (not per frame)
        if not SIMULATION_MODE:
            self._load_models()
            # Audio recorder — started when pipeline starts
            from src.audio.recorder import AudioRecorder
            self._audio_recorder = AudioRecorder()

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_models(self):
        """Load all four inference pipelines. Called once at startup."""
        import os
        from src.vision.inference  import VisionInference
        from src.audio.inference   import AudioInference
        from src.text.inference    import TextInference
        from src.fusion.inference  import FusionInference

        logger.info("Loading models — this may take 30-60 seconds on first run...")
        self.vision_inf  = VisionInference(
            os.getenv("VISION_MODEL_PATH",  "models/vision/efficientnet_fer2013.pth"))
        self.audio_inf   = AudioInference(
            os.getenv("AUDIO_MODEL_PATH",   "models/audio/wav2vec2_ravdess"))
        self.text_inf    = TextInference(
            os.getenv("TEXT_MODEL_PATH",    "models/text/roberta_goemotions"))
        self.fusion_inf  = FusionInference(
            os.getenv("FUSION_MODEL_PATH",  "models/fusion/fusion_mlp.pth"))
        logger.info("All models loaded.")

        # Whisper STT — tiny model for real-time transcription
        from src.text.transcriber import Transcriber
        self.transcriber = Transcriber(model_size="tiny")
        logger.info("Whisper STT loaded.")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="trifusion-capture")
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="trifusion-inference")
        self._capture_thread.start()
        self._inference_thread.start()
        if not SIMULATION_MODE and hasattr(self, '_audio_recorder'):
            self._audio_recorder.start()
        logger.info("Pipeline started (capture + inference threads).")

    def stop(self):
        self.running = False
        # Unblock any blocking .get() inside threads so they can exit cleanly
        try:
            self._frame_q.put_nowait(None)
        except queue.Full:
            pass
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._inference_thread:
            self._inference_thread.join(timeout=2)
        if not SIMULATION_MODE and hasattr(self, '_audio_recorder'):
            self._audio_recorder.stop()
        with self._display_lock:
            self._display_frame = None

        logger.info("Pipeline stopped.")

    # ── Capture thread ───────────────────────────────────────────────────────

    def _capture_loop(self):
        """
        Runs at full camera speed.  Zero ML work here.
        CAP_PROP_BUFFERSIZE=1 prevents OpenCV from buffering stale frames.
        Uses AVFoundation backend on macOS for maximum FPS.
        """
        import platform
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            cap = cv2.VideoCapture(0)  # retry with default backend

        if not cap.isOpened():
            logger.warning("No camera found — using blank placeholder frame.")
            while self.running:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "No camera detected", (160, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 120), 2)
                cv2.putText(blank, "Check permissions / connection", (150, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 100), 1)
                rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
                with self._display_lock:
                    self._display_frame = rgb
                self._put_latest(self._frame_q, blank)
                time.sleep(0.1)
            return

        # Force MJPEG codec for maximum USB bandwidth efficiency
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)

        fps_tick  = time.time()
        fps_count = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)   # mirror for natural interaction

                # Push to inference queue (non-blocking)
                self._put_latest(self._frame_q, frame.copy())

                # Push RGB copy to display buffer (ensure contiguous for Streamlit)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                with self._display_lock:
                    self._display_frame = rgb

                # FPS measurement (updated every second)
                fps_count += 1
                now = time.time()
                if now - fps_tick >= 1.0:
                    with self._fps_lock:
                        self._fps = round(fps_count / (now - fps_tick), 1)
                        self._frame_count += fps_count
                    fps_count = 0
                    fps_tick  = now
        finally:
            cap.release()
            logger.info("Camera released.")

    # ── Inference thread ─────────────────────────────────────────────────────

    def _inference_loop(self):
        """
        Consumes raw BGR frames from _frame_q.
        Runs full trimodal inference and writes results to shared dict.
        Runs at whatever speed the models allow (~10-15 FPS on CPU — fine).
        UI always reads LATEST raw frame from _display_q for smoothness.
        """
        agent_state    = None
        agent_response = "Session starting — I'll be right with you."
        agent_ticker   = 0   # only call WellnessAgent every N frames (Groq API cost)

        while self.running:
            try:
                frame = self._frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                break

            try:
                if SIMULATION_MODE:
                    result = self._simulate()
                else:
                    # ── Real trimodal inference ──────────────────────────────
                    # Vision: BGR frame → 7-class FER2013 probs
                    v_res = self.vision_inf.predict(frame)

                    # Draw face overlay directly onto the display frame
                    # Done in inference thread so capture thread stays pure
                    if v_res.get("face_detected") and v_res.get("bbox"):
                        annotated = self.vision_inf.face_detector.draw_overlay(
                            frame.copy(),
                            v_res["bbox"],
                            v_res["dominant"],
                            v_res["confidence"]
                        )
                        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        rgb_annotated = np.ascontiguousarray(rgb_annotated)
                        with self._display_lock:
                            self._display_frame = rgb_annotated

                    # Audio: real microphone waveform via AudioRecorder
                    waveform = (
                        self._audio_recorder.get_chunk()
                        if hasattr(self, '_audio_recorder') and self._audio_recorder.is_active
                        else np.zeros(16000, dtype=np.float32)
                    )
                    a_res = self.audio_inf.predict(waveform)

                    # Text: Whisper STT (every 5th frame to save CPU)
                    if agent_ticker % 5 == 0 and hasattr(self, '_audio_recorder'):
                        waveform_for_stt = (
                            self._audio_recorder.get_chunk()
                            if self._audio_recorder.is_active
                            else np.zeros(16000, dtype=np.float32)
                        )
                        transcribed_text = self.transcriber.transcribe(waveform_for_stt)
                        self._last_transcribed = transcribed_text
                    else:
                        transcribed_text = self._last_transcribed
                    t_res = self.text_inf.predict(transcribed_text)

                    # Fusion: late-fusion MLP + KL incongruence scorer
                    f_res = self.fusion_inf.fuse(v_res, a_res, t_res)

                    # ── WellnessAgent (throttled — every 30 inference frames) ─
                    # 30 frames × ~80ms/frame ≈ 2.4 sec between agent calls
                    if agent_ticker % 30 == 0:
                        try:
                            from src.agent.graph import wellness_agent
                            from src.agent.state import (
                                AgentState, EmotionFrame, ModalityResult)

                            if agent_state is None:
                                agent_state = AgentState()

                            ef = EmotionFrame(
                                dominant_emotion   = f_res["dominant_emotion"],
                                confidence         = f_res["confidence"],
                                incongruence_score = f_res["incongruence_score"],
                                vision = ModalityResult(
                                    dominant      = v_res["dominant"],
                                    confidence    = v_res["confidence"],
                                    probabilities = v_res["probabilities"]),
                                audio  = ModalityResult(
                                    dominant      = a_res["dominant"],
                                    confidence    = a_res["confidence"],
                                    probabilities = a_res["probabilities"]),
                                text   = ModalityResult(
                                    dominant      = t_res["dominant"],
                                    confidence    = t_res["confidence"],
                                    probabilities = t_res["probabilities"]),
                                user_text = self._last_transcribed
                            )
                            agent_state.current_frame = ef

                            # LangGraph invoke returns a dictionary; cast it back to AgentState
                            # to preserve dot-notation compatibility in the next loop iteration.
                            output = wellness_agent.invoke(agent_state.model_dump())
                            agent_state = AgentState(**output)
                            agent_response = agent_state.agent_response
                        except Exception as e:
                            logger.warning(f"WellnessAgent error: {e}")

                    agent_ticker += 1
                    result = {
                        "vision":         v_res,
                        "audio":          a_res,
                        "text":           t_res,
                        "fusion":         f_res,
                        "agent_response": agent_response,
                    }

                # Thread-safe result write
                with self._result_lock:
                    self._result.update(result)

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)

    # ── Public API (Streamlit reads from here) ───────────────────────────────

    def get_latest(self) -> dict:
        """
        Returns latest frame + inference results.
        Non-blocking — safe to call from Streamlit's main thread at any rate.
        """
        # Get freshest frame from buffer
        with self._display_lock:
            frame = self._display_frame.copy() if self._display_frame is not None else None

        with self._result_lock:
            result = dict(self._result)

        with self._fps_lock:
            result["fps"]         = self._fps
            result["frame_count"] = self._frame_count

        result["frame"] = frame
        return result

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _put_latest(q: queue.Queue, item):
        """
        Non-blocking put.  If the queue is full, evict the stale item first
        so we always keep the LATEST data, never accumulate a backlog.
        """
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    @staticmethod
    def _empty_result() -> dict:
        """Default result before inference produces real data."""
        from config.emotions import UNIFIED_EMOTIONS
        uniform = {e: 1.0 / len(UNIFIED_EMOTIONS) for e in UNIFIED_EMOTIONS}
        return {
            "vision":  {"dominant": "neutral", "confidence": 0.0, "probabilities": uniform},
            "audio":   {"dominant": "neutral", "confidence": 0.0, "probabilities": uniform},
            "text":    {"dominant": "neutral", "confidence": 0.0, "probabilities": uniform},
            "fusion":  {
                "dominant_emotion":   "neutral",
                "confidence":         0.0,
                "incongruence_score": 0.0,
                "fused_probabilities": uniform,
            },
            "agent_response": "Session starting — I'll be right with you.",
            "fps":            0.0,
            "frame_count":    0,
        }

    # ── Simulation ───────────────────────────────────────────────────────────

    def _simulate(self) -> dict:
        """
        Smooth simulation for demo / testing without trained model weights.
        Emotions transition smoothly over ~3 second windows rather than
        jumping randomly each frame, which is much more realistic.
        """
        import random
        from config.emotions import UNIFIED_EMOTIONS

        if not hasattr(self, "_sim_target"):
            self._sim_target = "neutral"
            self._sim_tick   = 0
            self._sim_probs  = {
                k: {e: 1.0 / len(UNIFIED_EMOTIONS) for e in UNIFIED_EMOTIONS}
                for k in ["v", "a", "t", "f"]
            }
            self._sim_agent_responses = [
                "I'm noticing some changes in your expression. Take a deep breath.",
                "You seem calm right now. Keep it up!",
                "I'm here to listen if you want to talk about what's on your mind.",
                "Your signals show a bit of variation. Is everything okay?",
                "Let's take a moment to ground ourselves together.",
                "You are doing great — keep going. 💙",
            ]
            self._sim_resp_idx = 0

        self._sim_tick += 1

        # Change target emotion roughly every 3 seconds at 30 FPS
        if self._sim_tick % 90 == 0:
            self._sim_target = random.choice(UNIFIED_EMOTIONS)

        def smooth_toward(cur: dict, dominant: str, momentum: float = 0.97) -> dict:
            """Exponential moving average toward a target distribution."""
            target = {e: 0.04 for e in UNIFIED_EMOTIONS}
            target[dominant] = 0.65
            smoothed = {e: cur[e] * momentum + target[e] * (1.0 - momentum)
                        for e in UNIFIED_EMOTIONS}
            total = sum(smoothed.values())
            return {k: v / total for k, v in smoothed.items()}

        # Each modality lags slightly differently → realistic incongruence
        self._sim_probs["v"] = smooth_toward(self._sim_probs["v"], self._sim_target, 0.97)
        self._sim_probs["a"] = smooth_toward(self._sim_probs["a"], self._sim_target, 0.96)
        self._sim_probs["t"] = smooth_toward(self._sim_probs["t"], self._sim_target, 0.98)
        self._sim_probs["f"] = smooth_toward(self._sim_probs["f"], self._sim_target, 0.95)

        def to_result(p: dict) -> dict:
            dom = max(p, key=p.get)
            return {"dominant": dom, "confidence": p[dom], "probabilities": dict(p)}

        # Rotate agent response every 120 frames (~4 seconds)
        if self._sim_tick % 120 == 0:
            self._sim_resp_idx = (self._sim_resp_idx + 1) % len(self._sim_agent_responses)

        v_dom = max(self._sim_probs["v"], key=self._sim_probs["v"].get)
        a_dom = max(self._sim_probs["a"], key=self._sim_probs["a"].get)
        t_dom = max(self._sim_probs["t"], key=self._sim_probs["t"].get)
        inc   = 0.08 if v_dom == a_dom == t_dom else 0.52

        return {
            "vision": to_result(self._sim_probs["v"]),
            "audio":  to_result(self._sim_probs["a"]),
            "text":   to_result(self._sim_probs["t"]),
            "fusion": {
                "dominant_emotion":    max(self._sim_probs["f"], key=self._sim_probs["f"].get),
                "confidence":          max(self._sim_probs["f"].values()),
                "incongruence_score":  round(inc + (self._sim_tick % 7) * 0.01, 3),
                "fused_probabilities": dict(self._sim_probs["f"]),
            },
            "agent_response": self._sim_agent_responses[self._sim_resp_idx],
        }
