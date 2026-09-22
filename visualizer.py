"""
Visualization utilities for rPPG waveform plotting and HUD rendering.
"""

from typing import Union, Sequence, Optional
import cv2 as cv
import numpy as np

from config import GRAPH_WIDTH, GRAPH_HEIGHT


def create_pulse_graph(
    signal: Union[np.ndarray, Sequence[float]],
    width: int = GRAPH_WIDTH,
    height: int = GRAPH_HEIGHT
) -> np.ndarray:
    """
    Creates a real-time oscilloscope-style graph image for the accumulated pulse signal.

    Args:
        signal: 1D pulse signal values.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        BGR image showing the pulse wave.
    """
    graph = np.full((height, width, 3), 30, dtype=np.uint8)

    # Draw the central zero baseline
    zero_y = height // 2
    cv.line(graph, (0, zero_y), (width, zero_y), (80, 80, 80), 1)

    if len(signal) < 2:
        cv.putText(
            graph,
            "Waiting for pulse signal...",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            1,
            cv.LINE_AA
        )
        return graph

    values = np.asarray(signal, dtype=np.float32)
    amplitude = max(float(np.max(np.abs(values))), 1e-6)

    # Scale x coordinates across canvas width and y coordinates centered at zero line
    x_points = np.linspace(0, width - 1, len(values)).astype(np.int32)
    y_points = (zero_y - (values / amplitude) * (height * 0.42)).astype(np.int32)

    points = np.column_stack((x_points, y_points)).reshape((-1, 1, 2))
    cv.polylines(graph, [points], False, (0, 220, 120), 2, cv.LINE_AA)

    # Overlay metadata
    cv.putText(
        graph,
        "Master pulse signal",
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        1,
        cv.LINE_AA
    )
    cv.putText(
        graph,
        f"Latest: {values[-1]:.4f}",
        (width - 210, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv.LINE_AA
    )
    return graph


def draw_hud(
    frame: np.ndarray,
    bpm: Optional[float] = None,
    is_warming_up: bool = True,
    fps: Optional[float] = None
) -> np.ndarray:
    """
    Draws head-up display (HUD) statistics directly onto the display frame.

    Args:
        frame: BGR video frame to draw on.
        bpm: Estimated heart rate in BPM.
        is_warming_up: True if buffer is filling up before reliable BPM estimate.
        fps: Optional current frame rate to display.

    Returns:
        The frame with HUD elements rendered.
    """
    if is_warming_up or bpm is None or bpm <= 0:
        cv.putText(
            frame,
            "Calculating BPM...",
            (20, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv.LINE_AA
        )
    else:
        cv.putText(
            frame,
            f"BPM: {bpm:.1f}",
            (20, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
            cv.LINE_AA
        )

    if fps is not None and fps > 0:
        cv.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, frame.shape[0] - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv.LINE_AA
        )

    return frame
