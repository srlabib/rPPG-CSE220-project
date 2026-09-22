"""
Configuration constants for rPPG (remote photoplethysmography) processing.
"""

# MediaPipe FaceMesh landmark indices for Region of Interest (ROI) extraction
FOREHEAD_INDICES = [103, 104, 69, 108, 151, 337, 299, 333, 332, 297, 338, 10, 109, 67]
LEFT_CHEEK_INDICES = [117, 118, 119, 120, 100, 36, 206, 92, 216, 207, 187, 50]
RIGHT_CHEEK_INDICES = [349, 348, 347, 346, 280, 425, 427, 436, 322, 391, 423, 266, 329]

# Backward compatibility alias
FORHEAD_INDICES = FOREHEAD_INDICES

# Physiological frequency boundaries (human heart rate limits)
MIN_FREQ_HZ = 0.7   # ~42 BPM
MAX_FREQ_HZ = 3.0   # ~180 BPM
FILTER_ORDER = 6    # Butterworth filter order
POS_EPSILON = 1e-8  # Numerical stability constant for POS algorithm

# Temporal buffering constants
POS_WINDOW_TIME_SEC = 1.6     # Window duration for POS algorithm (seconds)
PULSE_BUFFER_MAX_LEN = 300    # Maximum number of accumulated pulse points
MIN_SAMPLES_FOR_BPM = 150     # Minimum pulse samples needed before computing BPM

# Video capture defaults
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FPS = 30.0

# Visualization settings
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 300
ROI_OVERLAY_ALPHA = 0.25
