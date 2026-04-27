"""
src/pipeline/manager.py
────────────────────────
Threaded pipeline manager to decouple hardware capture from the Streamlit UI.
This solves the 'DuplicateElementId' and 'While-Loop' issues in Streamlit.
"""

import threading
import time
import cv2
import numpy as np
import random
from typing import Optional, Dict

# ==============================================================================
# SIMULATION MODE
# Set to True to run the UI smoothly at 100+ FPS without loading heavy ML models.
# (Later, set to False to connect to the actual models).
# ==============================================================================
SIMULATION_MODE = True

if not SIMULATION_MODE:
    from src.vision.inference import VisionInference
    from src.audio.inference import AudioInference
    from src.text.inference import TextInference
    from src.fusion.inference import FusionInference
    from src.agent.graph import wellness_agent
    from src.agent.state import AgentState, EmotionFrame, ModalityResult

class PipelineManager:
    def __init__(self):
        self.running = False
        self.thread = None
        
        # Latest results
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_vision: Dict = {}
        self.latest_audio: Dict = {}
        self.latest_text: Dict = {}
        self.latest_fusion: Dict = {}
        self.latest_agent_response: str = "Initializing..."
        
        # Models (only load if not simulating)
        if not SIMULATION_MODE:
            self.vision_inf = VisionInference()
            self.audio_inf = AudioInference()
            self.text_inf = TextInference()
            self.fusion_inf = FusionInference()
            self.agent_state = AgentState()
        
        self.lock = threading.Lock()
        self.frame_count = 0
        
        # Simulation states
        self.sim_emotions = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "calm", "neutral"]
        self.current_v_probs = self._generate_mock_probs("neutral")
        self.current_a_probs = self._generate_mock_probs("neutral")
        self.current_t_probs = self._generate_mock_probs("neutral")
        self.current_f_probs = self._generate_mock_probs("neutral")
        self.target_dom = "neutral"
        self.target_tick = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _smooth_probs(self, current: dict, target_dom: str, momentum=0.95) -> dict:
        target = self._generate_mock_probs(target_dom)
        smoothed = {e: current[e] * momentum + target[e] * (1 - momentum) for e in self.sim_emotions}
        total = sum(smoothed.values())
        return {k: v / total for k, v in smoothed.items()}

    def _generate_mock_probs(self, dominant_emotion: str) -> dict:
        """Generate smooth fake probabilities for simulation."""
        probs = {e: random.uniform(0.01, 0.1) for e in self.sim_emotions}
        probs[dominant_emotion] = random.uniform(0.5, 0.9)
        # Normalize
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def _run_loop(self):
        cap = cv2.VideoCapture(0)
        
        # Create a placeholder frame with a helpful message for cloud/no-camera environments
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Hardware Not Found / Cloud Demo", (100, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 150), 2)
        cv2.putText(blank_frame, "Run locally for live webcam feed", (110, 260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 120), 1)
        
        try:
            while self.running:
                # 1. Capture Frame
                ret = False
                frame = blank_frame
                if cap is not None and cap.isOpened():
                    ret, raw_frame = cap.read()
                    if ret:
                        frame = cv2.flip(raw_frame, 1)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Run Inference or Simulation
                if SIMULATION_MODE:
                    # SMOOTH SIMULATION: Generate realistic, non-jittery data
                    if self.frame_count >= self.target_tick:
                        self.target_dom = random.choice(self.sim_emotions)
                        self.target_tick = self.frame_count + random.randint(60, 200) # hold emotion for 1-3 seconds
                    
                    self.current_v_probs = self._smooth_probs(self.current_v_probs, self.target_dom, 0.98)
                    self.current_a_probs = self._smooth_probs(self.current_a_probs, self.target_dom, 0.97)
                    self.current_t_probs = self._smooth_probs(self.current_t_probs, self.target_dom, 0.99)
                    self.current_f_probs = self._smooth_probs(self.current_f_probs, self.target_dom, 0.95)
                    
                    dom_v = max(self.current_v_probs, key=self.current_v_probs.get)
                    dom_a = max(self.current_a_probs, key=self.current_a_probs.get)
                    dom_t = max(self.current_t_probs, key=self.current_t_probs.get)
                    dom_f = max(self.current_f_probs, key=self.current_f_probs.get)
                    
                    v_res = {"dominant": dom_v, "confidence": self.current_v_probs[dom_v], "probabilities": self.current_v_probs}
                    a_res = {"dominant": dom_a, "confidence": self.current_a_probs[dom_a], "probabilities": self.current_a_probs}
                    t_res = {"dominant": dom_t, "confidence": self.current_t_probs[dom_t], "probabilities": self.current_t_probs}
                    
                    # Compute a stable incongruence based on current probabilities
                    # We'll just fake it smoothly
                    inc_score = 0.1 if dom_v == dom_a == dom_t else 0.6
                    
                    f_res = {
                        "dominant_emotion": dom_f, 
                        "confidence": self.current_f_probs[dom_f],
                        "incongruence_score": inc_score + random.uniform(-0.02, 0.02),
                        "fused_probabilities": self.current_f_probs
                    }
                    
                    if self.frame_count % 60 == 0:
                        agent_resp = random.choice([
                            "I'm noticing some changes in your expression. Take a deep breath.",
                            "You seem calm right now. Keep it up!",
                            "I'm here to listen if you want to talk about what's on your mind.",
                            "Your signals show a bit of variation. Is everything okay?",
                            "Let's take a moment to ground ourselves.",
                            "You are doing great. Keep going."
                        ])
                    else:
                        agent_resp = self.latest_agent_response
                        
                else:
                    # REAL INFERENCE (Will be enabled later)
                    v_res = self.vision_inf.predict(rgb_frame)
                    a_res = {"dominant": "neutral", "confidence": 0.0, "probabilities": {e: 0.125 for e in self.sim_emotions}}
                    t_res = {"dominant": "neutral", "confidence": 0.0, "probabilities": {e: 0.125 for e in self.sim_emotions}}
                    
                    f_res = self.fusion_inf.fuse(v_res, a_res, t_res)
                    
                    if self.frame_count % 30 == 0:
                        current_frame = EmotionFrame(
                            dominant_emotion=f_res["dominant_emotion"],
                            confidence=f_res["confidence"],
                            incongruence_score=f_res["incongruence_score"],
                            vision=ModalityResult(**v_res),
                            audio=ModalityResult(**a_res),
                            text=ModalityResult(**t_res),
                            user_text=""
                        )
                        self.agent_state.current_frame = current_frame
                        try:
                            self.agent_state = wellness_agent.invoke(self.agent_state)
                            agent_resp = self.agent_state.agent_response
                        except:
                            agent_resp = "I'm here for you."
                    else:
                        agent_resp = self.latest_agent_response

                # 3. Thread-safe state update
                with self.lock:
                    self.latest_frame = rgb_frame
                    self.latest_vision = v_res
                    self.latest_audio = a_res
                    self.latest_text = t_res
                    self.latest_fusion = f_res
                    self.latest_agent_response = agent_resp
                    self.frame_count += 1
                
                # Sleep briefly to yield thread (aim for ~100 FPS)
                time.sleep(0.01) 
        finally:
            if cap is not None:
                cap.release()

    def get_latest(self):
        with self.lock:
            return {
                "frame": self.latest_frame,
                "vision": self.latest_vision,
                "audio": self.latest_audio,
                "text": self.latest_text,
                "fusion": self.latest_fusion,
                "agent_response": self.latest_agent_response,
                "frame_count": self.frame_count
            }
