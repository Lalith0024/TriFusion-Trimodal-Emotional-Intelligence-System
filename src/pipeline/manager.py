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
from typing import Optional, Dict
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
        
        # Models
        self.vision_inf = VisionInference()
        self.audio_inf = AudioInference()
        self.text_inf = TextInference()
        self.fusion_inf = FusionInference()
        self.agent_state = AgentState()
        
        self.lock = threading.Lock()
        self.frame_count = 0

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

    def _run_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.running = False
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Pre-process
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inference (Vision every frame, others mocked for now)
                v_res = self.vision_inf.predict(rgb_frame)
                a_res = {"dominant": "neutral", "confidence": 0.0, "probabilities": {e: 0.125 for e in ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "calm", "neutral"]}}
                t_res = {"dominant": "neutral", "confidence": 0.0, "probabilities": {e: 0.125 for e in ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "calm", "neutral"]}}
                
                f_res = self.fusion_inf.fuse(v_res, a_res, t_res)
                
                # Agent (Every ~3 seconds)
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

                # Thread-safe update
                with self.lock:
                    self.latest_frame = rgb_frame
                    self.latest_vision = v_res
                    self.latest_audio = a_res
                    self.latest_text = t_res
                    self.latest_fusion = f_res
                    self.latest_agent_response = agent_resp
                    self.frame_count += 1
                
                # Tighten loop for high FPS on M3 Pro
                time.sleep(0.005) 
        finally:
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
