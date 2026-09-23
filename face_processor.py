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
    MAX_PROCESSING_DIMENSION,
)


@dataclass
class FaceROIDetection:
    """Contains the results of face detection and ROI extraction for a single frame."""
    detected: bool
    mean_rgbs: Optional[List[Tuple[float, float, float]]] = None
    masks: Optional[List[np.ndarray]] = None
    landmarks: Optional[List] = None
    polys: Optional[List[np.ndarray]] = None


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

    @staticmethod
    def _maybe_resize(
        frame: np.ndarray, max_dim: int = MAX_PROCESSING_DIMENSION
    ) -> np.ndarray:
        """
        Downscale the frame if its largest dimension exceeds max_dim.
        Preserves aspect ratio. Uses fast linear interpolation.
        """
        h, w = frame.shape[:2]
        largest = max(h, w)
        if largest <= max_dim:
            return frame
        scale = max_dim / largest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_LINEAR)

    def process_frame(self, frame_bgr: np.ndarray) -> FaceROIDetection:
        """
        Processes a BGR video frame to detect the face and extract mean RGB from skin ROIs.
        Large frames are automatically downscaled to keep processing fast.

        Args:
            frame_bgr: Input video frame in BGR format.

        Returns:
            FaceROIDetection dataclass containing detection status, mean RGB, binary mask, and polygons.
        """
        # Downscale large frames for faster MediaPipe inference
        frame_bgr = self._maybe_resize(frame_bgr)

        h, w, _ = frame_bgr.shape
        rgb_frame = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            return FaceROIDetection(detected=False)

        landmarks = result.multi_face_landmarks[0].landmark

        # Extract 12 patches efficiently using bounding-box cropped masks
        masks = []
        mean_rgbs = []
        polys = []

        for patch_indices in PATCH_INDICES:
            roi_points = np.array([
                [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
                for idx in patch_indices
            ], dtype=np.int32)
            polys.append(roi_points)

            # Crop to bounding box for fast color extraction
            x_min = max(0, int(np.min(roi_points[:, 0])))
            x_max = min(w, int(np.max(roi_points[:, 0])) + 1)
            y_min = max(0, int(np.min(roi_points[:, 1])))
            y_max = min(h, int(np.max(roi_points[:, 1])) + 1)

            if x_max > x_min and y_max > y_min:
                crop = frame_bgr[y_min:y_max, x_min:x_max]
                crop_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
                cv.fillConvexPoly(crop_mask, roi_points - [x_min, y_min], 255)
                mean_bgr = cv.mean(crop, mask=crop_mask)[:3]
                mean_rgbs.append((float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])))
            else:
                mean_rgbs.append((0.0, 0.0, 0.0))

        return FaceROIDetection(
            detected=True,
            mean_rgbs=mean_rgbs,
            masks=masks,
            landmarks=landmarks,
            polys=polys,
        )

    @staticmethod
    def render_roi_overlay(
        frame_bgr: np.ndarray,
        masks: Optional[List[np.ndarray]] = None,
        active_indices: Optional[List[int]] = None,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = ROI_OVERLAY_ALPHA,
        polys: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Renders a semi-transparent colored tint over the active ROIs on the BGR frame.
        Fast polygon-based rendering avoids large boolean mask arrays.
        """
        if active_indices is None or len(active_indices) == 0:
            return frame_bgr

        display_frame = frame_bgr.copy()
        overlay = display_frame.copy()
        has_overlay = False

        if polys is not None and len(polys) > 0:
            for idx in active_indices:
                if idx < len(polys):
                    cv.fillConvexPoly(overlay, polys[idx], overlay_color)
                    has_overlay = True
        elif masks is not None and len(masks) > 0:
            for idx in active_indices:
                if idx < len(masks):
                    overlay[masks[idx] > 0] = overlay_color
                    has_overlay = True

        if not has_overlay:
            return display_frame

        cv.addWeighted(overlay, alpha, display_frame, 1.0 - alpha, 0, dst=display_frame)
        return display_frame
