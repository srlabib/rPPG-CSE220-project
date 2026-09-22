"""
Configuration constants for rPPG (remote photoplethysmography) processing.
"""

# 12 distinct facial patches for Spatial Sub-region rPPG processing
PATCH_INDICES = [
    # --- Forehead (3 patches) ---
    [103, 104, 69, 108,109,67],         # 0: Left Forehead
    [108, 151, 337, 338,10,109],                 # 1: Center Forehead
    [337,299,333,332,297,338],       # 2: Right Forehead
    [104,105,66,107,108,69],
    [107,9,336,337,151,108],
    [336,296,334,333,299,337],

    
    # --- Left Cheek (3 patches) ---
    [213, 187, 50, 118, 117, 111, 123, 147],       # 3: Left Upper Cheek (under eye)
    [213,192,214,212,216,206,205,50,187],        # 4: Left Lower Cheek (near mouth)
    [206,36,100,120,119,118,50,205],            # 5: Left Outer Cheek (near ear)
    
    # --- Right Cheek (3 patches) ---
    [348, 330, 266,426,425,280,346,347],       # 6: Right Upper Cheek (under eye)
    [346,280,411,433,376,352,345,372,340],       # 7: Right Lower Cheek (near mouth)
    [426,436,432,434,416,433,411,280,425],            # 8: Right Outer Cheek (near ear)
    
]

# Provide backward compatibility aliases just in case
FOREHEAD_INDICES = PATCH_INDICES[0] + PATCH_INDICES[1] + PATCH_INDICES[2]
LEFT_CHEEK_INDICES = PATCH_INDICES[3] + PATCH_INDICES[4] + PATCH_INDICES[5]
RIGHT_CHEEK_INDICES = PATCH_INDICES[6] + PATCH_INDICES[7] + PATCH_INDICES[8]
FORHEAD_INDICES = FOREHEAD_INDICES

# Dynamic fusion parameters
NUM_PATCHES = 12
TOP_K_PATCHES = 6

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
