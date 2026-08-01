import threading
import queue
import time
import cv2
import numpy as np
import os
import logging
from typing import Optional, Dict
from src.pipeline.startup_validator import validate_checkpoints

logger = logging.getLogger(__name__)

_sim_mode_env = os.environ.get("SIMULATION_MODE", "false").lower() in ("true", "1", "t")

_checkpoints_ok, _missing_cps = validate_checkpoints()
if not _checkpoints_ok and not _sim_mode_env:
    logger.warning(f"Checkpoints missing: {_missing_cps}. Running in live mode anyway; models may random-init if missing.")
    SIMULATION_MODE = False
else:
    SIMULATION_MODE = _sim_mode_env

class PipelineManager:
    def __init__(self):
        self.running = False
        self._frame_q   = queue.Queue(maxsize=1)   # BGR frames → inference
        
        self._latest_vision = None
        self._latest_audio = None
        self._latest_text = None
        self._latest_fusion = None

        self._display_frame = None                 # RGB frame buffer for UI
        self._display_lock  = threading.Lock()

        self._result_lock = threading.Lock()
        self._result: Dict = self._empty_result()
        
        self._agent_q = queue.Queue(maxsize=1)

        self._fps_lock    = threading.Lock()
        self._fps         = 0.0
        self._frame_count = 0

        self._last_transcribed = ""

        self._capture_thread = None
        self._vision_thread = None
        self._audio_thread = None
        self._text_thread = None
        self._fusion_thread = None
        self._agent_thread = None
        
        if not SIMULATION_MODE:
            self._load_models()
            from src.audio.recorder import AudioRecorder
            self._audio_recorder = AudioRecorder()

    def _load_models(self):
        from src.vision.inference  import VisionInference
        from src.audio.inference   import AudioInference
        from src.text.inference    import TextInference
        from src.fusion.inference  import FusionInference
        from src.text.transcriber import Transcriber

        logger.info("Loading models...")
        self.vision_inf  = VisionInference(os.getenv("VISION_MODEL_PATH", "models/vision/efficientnet_fer2013.pth"))
        self.audio_inf   = AudioInference(os.getenv("AUDIO_MODEL_PATH", "models/audio/wav2vec2_ravdess"))
        self.text_inf    = TextInference(os.getenv("TEXT_MODEL_PATH", "models/text/roberta_goemotions"))
        self.fusion_inf  = FusionInference(os.getenv("FUSION_MODEL_PATH", "models/fusion/fusion_mlp.pth"))
        self.transcriber = Transcriber(model_size="tiny")
        logger.info("All models loaded.")

        # WebRTC Audio Buffer (3 seconds at 16000 Hz)
        self.SAMPLE_RATE = 16000
        self.AUDIO_BUFFER_SIZE = self.SAMPLE_RATE * 3
        self._audio_buffer = np.zeros(self.AUDIO_BUFFER_SIZE, dtype=np.float32)
        self._audio_buffer_lock = threading.Lock()
        
    def feed_video_frame(self, frame: np.ndarray):
        """Called by WebRTC video_frame_callback."""
        if not self.running: return
        self._put_latest(self._frame_q, frame.copy())
        
        # Calculate FPS
        now = time.time()
        if not hasattr(self, '_fps_tick'):
            self._fps_tick = now
            self._fps_count = 0
            
        self._fps_count += 1
        if now - self._fps_tick >= 1.0:
            with self._fps_lock:
                self._fps = round(self._fps_count / (now - self._fps_tick), 1)
                self._frame_count += self._fps_count
            self._fps_count = 0
            self._fps_tick = now

    def feed_audio_chunk(self, waveform: np.ndarray):
        """Called by WebRTC audio_frame_callback. waveform should be (N,)."""
        if not self.running: return
        chunk_len = len(waveform)
        with self._audio_buffer_lock:
            if chunk_len >= self.AUDIO_BUFFER_SIZE:
                self._audio_buffer[:] = waveform[-self.AUDIO_BUFFER_SIZE:]
            else:
                self._audio_buffer[:-chunk_len] = self._audio_buffer[chunk_len:]
                self._audio_buffer[-chunk_len:] = waveform

    def start(self):
        if self.running: return
        self.running = True
        
        if SIMULATION_MODE:
            self._fusion_thread = threading.Thread(target=self._simulate_loop, daemon=True, name="trifusion-sim")
            self._fusion_thread.start()
        else:
            self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="trifusion-vision")
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True, name="trifusion-audio")
            self._text_thread = threading.Thread(target=self._text_loop, daemon=True, name="trifusion-text")
            self._fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True, name="trifusion-fusion")
            self._agent_thread = threading.Thread(target=self._agent_loop, daemon=True, name="trifusion-agent")
            
            self._vision_thread.start()
            self._audio_thread.start()
            self._text_thread.start()
            self._fusion_thread.start()
            self._agent_thread.start()
        
        logger.info("Pipeline started.")

    def stop(self):
        self.running = False
        self._put_latest(self._frame_q, None)
        self._put_latest(self._agent_q, None)
        
        if not SIMULATION_MODE:
            if self._vision_thread: self._vision_thread.join(timeout=2)
            if self._audio_thread: self._audio_thread.join(timeout=2)
            if self._text_thread: self._text_thread.join(timeout=2)
            if self._fusion_thread: self._fusion_thread.join(timeout=2)
            if self._agent_thread: self._agent_thread.join(timeout=2)
        elif hasattr(self, '_fusion_thread') and self._fusion_thread:
            self._fusion_thread.join(timeout=2)

        with self._display_lock:
            self._display_frame = None

    def _vision_loop(self):
        while self.running:
            try:
                frame = self._frame_q.get(timeout=0.1)
                if frame is None: break
                v_res = self.vision_inf.predict(frame)
                with self._result_lock:
                    self._latest_vision = v_res
                    self._result["vision"] = v_res
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Vision error: {e}")

    def _audio_loop(self):
        while self.running:
            try:
                with self._audio_buffer_lock:
                    waveform = self._audio_buffer.copy()
                if waveform.sum() != 0:
                    a_res = self.audio_inf.predict(waveform)
                    with self._result_lock:
                        self._latest_audio = a_res
                        self._result["audio"] = a_res
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Audio error: {e}")

    def _text_loop(self):
        while self.running:
            try:
                with self._audio_buffer_lock:
                    waveform = self._audio_buffer.copy()
                if waveform.sum() != 0:
                    transcribed = self.transcriber.transcribe(waveform)
                    if transcribed and transcribed.strip():
                        self._last_transcribed = transcribed
                        t_res = self.text_inf.predict(transcribed)
                        with self._result_lock:
                            self._latest_text = t_res
                            self._result["text"] = t_res
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Text error: {e}")

    def _fusion_loop(self):
        agent_ticker = 0
        while self.running:
            try:
                with self._result_lock:
                    v_res = self._latest_vision
                    a_res = self._latest_audio
                    t_res = self._latest_text
                    
                if v_res and a_res and t_res:
                    f_res = self.fusion_inf.fuse(v_res, a_res, t_res)
                    with self._result_lock:
                        self._result["fusion"] = f_res
                        
                    if agent_ticker % 30 == 0:
                        from src.agent.state import EmotionFrame, ModalityResult
                        ef = EmotionFrame(
                            dominant_emotion = f_res["dominant_emotion"],
                            confidence = f_res["confidence"],
                            incongruence_score = f_res["incongruence_score"],
                            vision = ModalityResult(dominant=v_res["dominant"], confidence=v_res["confidence"], probabilities=v_res["probabilities"]),
                            audio = ModalityResult(dominant=a_res["dominant"], confidence=a_res["confidence"], probabilities=a_res["probabilities"]),
                            text = ModalityResult(dominant=t_res["dominant"], confidence=t_res["confidence"], probabilities=t_res["probabilities"]),
                            user_text = self._last_transcribed
                        )
                        self._put_latest(self._agent_q, ef)
                
                agent_ticker += 1
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Fusion error: {e}")

    def _agent_loop(self):
        from src.agent.state import AgentState
        agent_state = AgentState()
        
        while self.running:
            try:
                ef = self._agent_q.get(timeout=1.0)
                if ef is None: break
                
                from src.agent.graph import wellness_agent
                agent_state.current_frame = ef
                output = wellness_agent.invoke(agent_state.model_dump())
                agent_state = AgentState(**output)
                
                with self._result_lock:
                    self._result["agent_response"] = agent_state.agent_response
            except queue.Empty:
                pass
            except Exception as e:
                logger.warning(f"Agent error: {e}")

    def _simulate_loop(self):
        import random
        from config.emotions import UNIFIED_EMOTIONS
        
        self._sim_target = "neutral"
        self._sim_tick = 0
        self._sim_probs = {k: {e: 1.0/len(UNIFIED_EMOTIONS) for e in UNIFIED_EMOTIONS} for k in ["v", "a", "t", "f"]}
        self._sim_resps = ["Take a deep breath.", "You seem calm.", "I'm listening.", "Is everything okay?", "You're doing great."]
        
        while self.running:
            self._sim_tick += 1
            if self._sim_tick % 90 == 0: self._sim_target = random.choice(UNIFIED_EMOTIONS)
            
            def smooth(cur, dom, mom=0.97):
                tgt = {e: 0.04 for e in UNIFIED_EMOTIONS}
                tgt[dom] = 0.65
                sm = {e: cur[e]*mom + tgt[e]*(1-mom) for e in UNIFIED_EMOTIONS}
                tot = sum(sm.values())
                return {k: v/tot for k, v in sm.items()}
                
            self._sim_probs["v"] = smooth(self._sim_probs["v"], self._sim_target, 0.97)
            self._sim_probs["a"] = smooth(self._sim_probs["a"], self._sim_target, 0.96)
            self._sim_probs["t"] = smooth(self._sim_probs["t"], self._sim_target, 0.98)
            self._sim_probs["f"] = smooth(self._sim_probs["f"], self._sim_target, 0.95)
            
            def to_res(p):
                dom = max(p, key=p.get)
                return {"dominant": dom, "confidence": p[dom], "probabilities": dict(p)}
                
            v_dom = max(self._sim_probs["v"], key=self._sim_probs["v"].get)
            a_dom = max(self._sim_probs["a"], key=self._sim_probs["a"].get)
            t_dom = max(self._sim_probs["t"], key=self._sim_probs["t"].get)
            inc = 0.08 if v_dom == a_dom == t_dom else 0.52
            
            with self._result_lock:
                self._result["vision"] = to_res(self._sim_probs["v"])
                self._result["audio"] = to_res(self._sim_probs["a"])
                self._result["text"] = to_res(self._sim_probs["t"])
                self._result["fusion"] = {
                    "dominant_emotion": max(self._sim_probs["f"], key=self._sim_probs["f"].get),
                    "confidence": max(self._sim_probs["f"].values()),
                    "incongruence_score": round(inc + (self._sim_tick % 7) * 0.01, 3),
                    "fused_probabilities": dict(self._sim_probs["f"])
                }
                if self._sim_tick % 120 == 0:
                    self._result["agent_response"] = random.choice(self._sim_resps)
                    
            time.sleep(0.1)

    def get_latest(self) -> dict:
        with self._display_lock:
            frame = self._display_frame.copy() if self._display_frame is not None else None

        with self._result_lock:
            result = dict(self._result)

        if frame is not None and not SIMULATION_MODE and "vision" in result:
            v_res = result["vision"]
            if v_res.get("face_detected") and v_res.get("bbox") and hasattr(self, 'vision_inf'):
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                bgr_annotated = self.vision_inf.face_detector.draw_overlay(
                    bgr_frame, v_res["bbox"], v_res["dominant"], v_res["confidence"])
                frame = cv2.cvtColor(bgr_annotated, cv2.COLOR_BGR2RGB)

        with self._fps_lock:
            result["fps"] = self._fps
            result["frame_count"] = self._frame_count

        result["frame"] = frame
        result["simulation_mode"] = SIMULATION_MODE
        result["missing_checkpoints"] = _missing_cps
        return result

    @staticmethod
    def _put_latest(q: queue.Queue, item):
        try: q.put_nowait(item)
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            try: q.put_nowait(item)
            except queue.Full: pass

    @staticmethod
    def _empty_result() -> dict:
        from config.emotions import UNIFIED_EMOTIONS
        uniform = {e: 1.0 / len(UNIFIED_EMOTIONS) for e in UNIFIED_EMOTIONS}
        return {
            "vision": {"dominant": "neutral", "confidence": 0.0, "probabilities": uniform},
            "audio": {"dominant": "neutral", "confidence": 0.0, "probabilities": uniform},
            "text": {"dominant": "neutral", "confidence": 0.0, "probabilities": uniform},
            "fusion": {"dominant_emotion": "neutral", "confidence": 0.0, "incongruence_score": 0.0, "fused_probabilities": uniform},
            "agent_response": "Session starting — I'll be right with you.",
            "fps": 0.0,
            "frame_count": 0,
        }
