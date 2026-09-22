"""
Facial landmark detection and Region of Interest (ROI) extraction using MediaPipe FaceMesh.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2 as cv
import mediapipe as mp
import numpy as np

from config import (
    FOREHEAD_INDICES,
    LEFT_CHEEK_INDICES,
    RIGHT_CHEEK_INDICES,
    ROI_OVERLAY_ALPHA,
)


@dataclass
class FaceROIDetection:
    """Contains the results of face detection and ROI extraction for a single frame."""
    detected: bool
    mean_rgb: Optional[Tuple[float, float, float]] = None
    mask: Optional[np.ndarray] = None
    landmarks: Optional[List] = None


class FaceROIProcessor:
    """
    Manages MediaPipe FaceMesh detection and extracts skin ROIs
    (forehead, left cheek, right cheek) for rPPG color extraction.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()

    def process_frame(self, frame_bgr: np.ndarray) -> FaceROIDetection:
        """
        Processes a BGR video frame to detect the face and extract mean RGB from skin ROIs.

        Args:
            frame_bgr: Input video frame in BGR format.

        Returns:
            FaceROIDetection dataclass containing detection status, mean RGB, and binary mask.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            return FaceROIDetection(detected=False)

        landmarks = result.multi_face_landmarks[0].landmark

        # Convert normalized coordinates to absolute pixel coordinates
        forehead_roi = np.array([
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            for idx in FOREHEAD_INDICES
        ], dtype=np.int32)

        left_cheek_roi = np.array([
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            for idx in LEFT_CHEEK_INDICES
        ], dtype=np.int32)

        right_cheek_roi = np.array([
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            for idx in RIGHT_CHEEK_INDICES
        ], dtype=np.int32)

        # Build combined binary mask for all ROIs
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.fillConvexPoly(mask, forehead_roi, 255)
        cv.fillConvexPoly(mask, left_cheek_roi, 255)
        cv.fillConvexPoly(mask, right_cheek_roi, 255)

        # Extract mean RGB values across the masked skin region
        mean_bgr = cv.mean(frame_bgr, mask=mask)[:3]
        # Convert BGR mean to RGB tuple
        mean_rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

        return FaceROIDetection(
            detected=True,
            mean_rgb=mean_rgb,
            mask=mask,
            landmarks=landmarks,
        )

    @staticmethod
    def render_roi_overlay(
        frame_bgr: np.ndarray,
        mask: Optional[np.ndarray],
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = ROI_OVERLAY_ALPHA,
    ) -> np.ndarray:
        """
        Renders a semi-transparent colored tint over the extracted ROI on the BGR frame.

        Args:
            frame_bgr: Original BGR image.
            mask: Binary mask where 255 indicates ROI.
            overlay_color: BGR color for tinting (default green).
            alpha: Transparency factor (0.0 to 1.0).

        Returns:
            Display frame with green tinted ROIs.
        """
        display_frame = frame_bgr.copy()
        if mask is None or not np.any(mask > 0):
            return display_frame

        # Create overlay layer
        overlay = display_frame.copy()
        overlay[mask > 0] = overlay_color

        # Alpha blend only in the masked region
        cv.addWeighted(overlay, alpha, display_frame, 1.0 - alpha, 0, dst=display_frame)
        return display_frame
