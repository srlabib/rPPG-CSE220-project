"""
Facial landmark detection and Region of Interest (ROI) extraction using MediaPipe FaceMesh.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2 as cv
import mediapipe as mp
import numpy as np

from config import (
    PATCH_INDICES,
    ROI_OVERLAY_ALPHA,
)


@dataclass
class FaceROIDetection:
    """Contains the results of face detection and ROI extraction for a single frame."""
    detected: bool
    mean_rgbs: Optional[List[Tuple[float, float, float]]] = None
    masks: Optional[List[np.ndarray]] = None
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

        # Extract 12 patches
        masks = []
        mean_rgbs = []
        
        for patch_indices in PATCH_INDICES:
            roi_points = np.array([
                [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
                for idx in patch_indices
            ], dtype=np.int32)
            
            patch_mask = np.zeros((h, w), dtype=np.uint8)
            cv.fillConvexPoly(patch_mask, roi_points, 255)
            masks.append(patch_mask)
            
            mean_bgr = cv.mean(frame_bgr, mask=patch_mask)[:3]
            mean_rgbs.append((float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])))

        return FaceROIDetection(
            detected=True,
            mean_rgbs=mean_rgbs,
            masks=masks,
            landmarks=landmarks,
        )

    @staticmethod
    def render_roi_overlay(
        frame_bgr: np.ndarray,
        masks: Optional[List[np.ndarray]],
        active_indices: Optional[List[int]] = None,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = ROI_OVERLAY_ALPHA,
    ) -> np.ndarray:
        """
        Renders a semi-transparent colored tint over the active ROIs on the BGR frame.

        Args:
            frame_bgr: Original BGR image.
            masks: List of 12 binary masks.
            active_indices: List of indices (e.g. top K) to render. If None, renders all.
            overlay_color: BGR color for tinting (default green).
            alpha: Transparency factor (0.0 to 1.0).

        Returns:
            Display frame with green tinted ROIs.
        """
        display_frame = frame_bgr.copy()
        if masks is None or len(masks) == 0:
            return display_frame

        if active_indices is None:
            active_indices = list(range(len(masks)))

        # Create combined mask of active regions
        combined_mask = np.zeros_like(masks[0])
        for idx in active_indices:
            if idx < len(masks):
                cv.bitwise_or(combined_mask, masks[idx], dst=combined_mask)

        if not np.any(combined_mask > 0):
            return display_frame

        # Create overlay layer
        overlay = display_frame.copy()
        overlay[combined_mask > 0] = overlay_color

        # Alpha blend only in the masked region
        cv.addWeighted(overlay, alpha, display_frame, 1.0 - alpha, 0, dst=display_frame)
        return display_frame
