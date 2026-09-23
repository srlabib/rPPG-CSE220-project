"""
Configuration constants for rPPG (remote photoplethysmography) processing.
"""

# 12 distinct facial patches for Spatial Sub-region rPPG processing
PATCH_INDICES = [
    # --- Forehead (6 patches) ---
    [103, 104, 69, 108, 109, 67],       # 0: Left Lower Forehead
    [108, 151, 337, 338, 10, 109],      # 1: Center Lower Forehead
    [337, 299, 333, 332, 297, 338],     # 2: Right Lower Forehead
    [104, 105, 66, 107, 108, 69],       # 3: Left Upper Forehead
    [107, 9, 336, 337, 151, 108],       # 4: Center Upper Forehead
    [336, 296, 334, 333, 299, 337],     # 5: Right Upper Forehead

    # --- Left Cheek (3 patches) ---
    [213, 187, 50, 118, 117, 111, 123, 147],   # 6: Left Upper Cheek (under eye)
    [213, 192, 214, 212, 216, 206, 205, 50, 187],  # 7: Left Lower Cheek (near mouth)
    [206, 36, 100, 120, 119, 118, 50, 205],        # 8: Left Outer Cheek (near ear)

    # --- Right Cheek (3 patches) ---
    [348, 330, 266, 426, 425, 280, 346, 347],      # 9: Right Upper Cheek (under eye)
    [346, 280, 411, 433, 376, 352, 345, 372, 340],  # 10: Right Lower Cheek (near mouth)
    [426, 436, 432, 434, 416, 433, 411, 280, 425], # 11: Right Outer Cheek (near ear)
]

# Provide backward compatibility aliases just in case
FOREHEAD_INDICES = (
    PATCH_INDICES[0] + PATCH_INDICES[1] + PATCH_INDICES[2] +
    PATCH_INDICES[3] + PATCH_INDICES[4] + PATCH_INDICES[5]
)
LEFT_CHEEK_INDICES = PATCH_INDICES[6] + PATCH_INDICES[7] + PATCH_INDICES[8]
RIGHT_CHEEK_INDICES = PATCH_INDICES[9] + PATCH_INDICES[10] + PATCH_INDICES[11]
FORHEAD_INDICES = FOREHEAD_INDICES

# Dynamic fusion parameters
NUM_PATCHES = 12
TOP_K_PATCHES = 6

# Physiological frequency boundaries (human heart rate limits)
MIN_FREQ_HZ = 0.75  # ~45 BPM (avoids filter transition edge and low-freq drift)
MAX_FREQ_HZ = 2.5   # ~150 BPM (resting range; blocks high-frequency noise and 2nd harmonic aliases)
FILTER_ORDER = 6    # Butterworth filter order
POS_EPSILON = 1e-8  # Numerical stability constant for POS algorithm
FFT_NFFT = 2048     # Zero-padded FFT points for fine BPM resolution (~0.88 BPM precision at 30 FPS)

# Temporal buffering constants
POS_WINDOW_TIME_SEC = 2.5     # Window duration for POS algorithm (covers 3+ cardiac cycles)
PULSE_BUFFER_MAX_LEN = 300    # Maximum number of accumulated pulse points
MIN_SAMPLES_FOR_BPM = 150     # Minimum pulse samples needed before computing BPM

# Video capture defaults
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_CAM_WIDTH = 640       # Recommended 640x480 for webcams to guarantee 30 FPS
DEFAULT_CAM_HEIGHT = 480
DEFAULT_FPS = 30.0

# Frame resizing: if the largest dimension exceeds this, downscale before processing.
# 640px ensures fast real-time 30+ FPS processing and fits comfortably on any screen.
MAX_PROCESSING_DIMENSION = 640

# Visualization settings
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 300
ROI_OVERLAY_ALPHA = 0.25
