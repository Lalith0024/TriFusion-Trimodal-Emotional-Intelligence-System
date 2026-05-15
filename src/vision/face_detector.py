"""
src/vision/face_detector.py
───────────────────────────
MediaPipe-based real-time face detector.

Key design decisions:
  • Uses the lightweight MediaPipe Face Detection model (not Face Mesh) for
    low-latency bounding-box extraction — good enough for emotion crop.
  • Adds proportional padding around the detected box so the model sees
    some forehead/chin context, which helps FER2013-trained models.
  • Falls back gracefully (returns None) when no face is found — callers
    handle the no-face case by returning uniform emotion distributions.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Wraps MediaPipe FaceDetection for single-frame face crop + overlay."""

    def __init__(self, min_detection_confidence: float = 0.7):
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            # Lower model_selection=0 → short-range (≤2 m) — ideal for webcam use
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_detection_confidence
            )
            self.active = True
            logger.info("MediaPipe face detection initialised.")
        except (AttributeError, ImportError, Exception):
            logger.warning("MediaPipe legacy solutions not available. Falling back to OpenCV Haar Cascades.")
            self.active = False
            self.detector = None
            # Load OpenCV's built-in Haar Cascade detector
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_and_crop(
        self,
        frame: np.ndarray,
        target_size: int = 224,
        padding: float = 0.2
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Detect the highest-confidence face in a BGR frame.

        Args:
            frame:       BGR numpy array from OpenCV capture.
            target_size: Output crop will be resized to (target_size × target_size).
            padding:     Fractional padding added on each side of the bbox.

        Returns:
            (cropped_face, bbox_info) — both None if no face is detected.
            cropped_face: BGR uint8 array of shape (target_size, target_size, 3)
            bbox_info:    dict with keys x, y, w, h, confidence
        """
        if not self.active or self.detector is None:
            # Fallback: Use OpenCV Haar Cascades
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, None
                
            # Pick largest face
            (fx, fy, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding
            px = int(fw * padding)
            py = int(fh * padding)
            
            x, y = max(0, fx - px), max(0, fy - py)
            w_pad, h_pad = fw + 2 * px, fh + 2 * py
            
            h_img, w_img = frame.shape[:2]
            w_pad = min(w_pad, w_img - x)
            h_pad = min(h_pad, h_img - y)
            
            face_crop = frame[y : y + h_pad, x : x + w_pad]
            face_resized = cv2.resize(face_crop, (target_size, target_size))
            
            return face_resized, {"x": x, "y": y, "w": w_pad, "h": h_pad, "confidence": 0.8}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if not results.detections:
            return None, None

        # Pick the highest-confidence detection when multiple faces appear
        detection = max(results.detections, key=lambda d: d.score[0])
        bbox = detection.location_data.relative_bounding_box
        h, w = frame.shape[:2]

        # Expand bbox by proportional padding on all four sides
        x = int((bbox.xmin - padding * bbox.width) * w)
        y = int((bbox.ymin - padding * bbox.height) * h)
        fw = int(bbox.width * (1 + 2 * padding) * w)
        fh = int(bbox.height * (1 + 2 * padding) * h)

        # Clamp to frame boundaries to avoid out-of-bounds slicing
        x, y = max(0, x), max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)

        if fw <= 0 or fh <= 0:
            return None, None

        face_crop = frame[y : y + fh, x : x + fw]
        face_resized = cv2.resize(face_crop, (target_size, target_size))

        bbox_info = {
            "x": x, "y": y, "w": fw, "h": fh,
            "confidence": float(detection.score[0])
        }
        return face_resized, bbox_info

    def draw_overlay(
        self,
        frame: np.ndarray,
        bbox_info: dict,
        emotion: str,
        confidence: float
    ) -> np.ndarray:
        """
        Draw bounding box + emotion label directly on the input frame.
        Each emotion has a distinct colour so the UI feels expressive.
        """
        if bbox_info is None:
            return frame

        x, y, w, h = bbox_info["x"], bbox_info["y"], bbox_info["w"], bbox_info["h"]

        # BGR colour palette — one per unified emotion
        emotion_colors = {
            "happy":     (0,   255, 100),
            "calm":      (100, 255, 200),
            "neutral":   (200, 200, 200),
            "sad":       (100, 100, 255),
            "angry":     (0,   50,  255),
            "fearful":   (0,   165, 255),
            "surprised": (0,   255, 255),
            "disgusted": (100, 0,   200),
        }
        color = emotion_colors.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{emotion.upper()} {confidence:.0%}"
        # Draw a filled rect behind the text for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return frame

    def close(self):
        """Release MediaPipe resources when done."""
        if self.active and self.detector:
            self.detector.close()
