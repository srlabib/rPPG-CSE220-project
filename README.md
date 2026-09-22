# rPPG Pulse & Heart Rate Extraction

Real-time remote photoplethysmography (rPPG) system in Python that measures human heart rate (BPM) from facial video using **MediaPipe FaceMesh** and the **POS (Plane-Orthogonal-to-Skin)** algorithm.

---

## Architecture Overview

The codebase is organized into modular components with clear separation of concerns:

```
rPPG/
├── config.py             # Global constants, ROI indices, and default parameters
├── signal_processing.py  # POS projection, Butterworth bandpass filter, Welch BPM calculation
├── face_processor.py     # MediaPipe FaceMesh wrapper & skin ROI extractor (forehead & cheeks)
├── visualizer.py         # Pulse wave oscilloscope plot and HUD overlay renderer
├── rppg_pipeline.py      # Core stateful pipeline managing buffers and heart rate estimation
└── VideoProcessing.py    # Main CLI application entry point and video loop orchestrator
```

### Module Responsibilities

| Module | Description |
| :--- | :--- |
| **`config.py`** | Centralized configuration holding MediaPipe landmark indices (`FOREHEAD_INDICES`, `LEFT_CHEEK_INDICES`, `RIGHT_CHEEK_INDICES`), physiological frequency boundaries (0.7 Hz - 3.0 Hz / 42 - 180 BPM), buffer lengths, and resolution defaults. |
| **`signal_processing.py`** | Pure mathematical signal processing functions: <br>• `execute_pos_algorithm()`: Wang et al. (2017) POS projection algorithm.<br>• `butter_bandpass_filter()`: 6th-order zero-phase Butterworth filter (`scipy.signal.filtfilt`).<br>• `calculate_bpm()`: Power Spectral Density (PSD) via Welch's method (`scipy.signal.welch`). |
| **`face_processor.py`** | `FaceROIProcessor` class encapsulating MediaPipe FaceMesh lifecycle, extracting forehead and cheek skin regions, calculating mean RGB color vectors, and rendering semi-transparent overlays. |
| **`visualizer.py`** | `create_pulse_graph()` for real-time oscilloscope display and `draw_hud()` for clean on-screen status readouts. |
| **`rppg_pipeline.py`** | `RPPGPipeline` class maintaining sliding RGB windows, accumulated pulse buffers, and updating filtered signals and BPM measurements. |
| **`VideoProcessing.py`** | Application runner with CLI argument parsing, webcam/video file abstraction, keyboard control (`q`/`ESC`), and guaranteed cleanup via `try...finally`. |

---

## How to Run

### 1. Run with Default Source
By default, the script looks for `data/10-gt/vid.avi` if present, or connects to camera `0`:
```bash
python VideoProcessing.py
```

### 2. Run with a Specific Video File
```bash
python VideoProcessing.py --source "path/to/video.mp4"
```

### 3. Run with Webcam / DroidCam
Use camera index (e.g. `0` for integrated webcam, `1` or `2` for DroidCam):
```bash
python VideoProcessing.py --source 0
```

### 4. Enable Camera Hardware Settings (Windows)
To manually disable auto-exposure on webcams / DroidCam:
```bash
python VideoProcessing.py --source 0 --camera-settings
```

### 5. CLI Options Reference
```
options:
  -h, --help            Show this help message and exit
  -s, --source SOURCE   Path to video file or camera index (default: data/10-gt/vid.avi or 0)
  --width WIDTH         Camera frame width (default: 1280)
  --height HEIGHT       Camera frame height (default: 720)
  --fps FPS             Override frame rate (default: auto-detected or 30.0)
  --camera-settings     Open camera driver hardware settings dialog (Windows)
  --no-graph            Disable the pulse graph window
```

---

## Backward Compatibility
Existing scripts importing legacy functions and constants directly from `VideoProcessing.py` continue to work:
```python
from VideoProcessing import (
    execute_pos_algorithm,
    calculate_bpm,
    butter_bandpass_filter,
    create_pulse_graph,
    FORHEAD_INDICES,
    FOREHEAD_INDICES
)
```
