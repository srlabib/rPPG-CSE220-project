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
├── VideoProcessing.py    # Main CLI application entry point and video loop orchestrator
└── requirements.txt      # Project dependencies and pinned versions
```

### Module Responsibilities

| Module | Description |
| :--- | :--- |
| **`config.py`** | Centralized configuration holding 12-patch facial indices, physiological frequency boundaries (0.75 Hz - 2.5 Hz / 45 - 150 BPM), POS window (2.5s), and camera resolution defaults. |
| **`signal_processing.py`** | Pure mathematical signal processing functions: <br>• `execute_pos_algorithm()` / `execute_pos_algorithm_batch()`: Wang et al. (2017) POS projection.<br>• `butter_bandpass_filter()`: 6th-order zero-phase Butterworth filter (`scipy.signal.filtfilt`).<br>• `calculate_bpm()`: Zero-padded 2048-pt FFT with linear detrending, Hann windowing, and sub-harmonic verification to eliminate harmonic doubling. |
| **`face_processor.py`** | `FaceROIProcessor` class encapsulating MediaPipe FaceMesh lifecycle, extracting forehead and cheek skin regions, calculating mean RGB color vectors, and rendering semi-transparent overlays. |
| **`visualizer.py`** | `create_pulse_graph()` for real-time oscilloscope display and `draw_hud()` for clean on-screen status readouts. |
| **`rppg_pipeline.py`** | `RPPGPipeline` class maintaining sliding RGB windows, Top-K SNR selection, Overlap-Add pulse accumulation, and temporal smoothing (moving median + EMA) for stable heart rate readout. |
| **`VideoProcessing.py`** | Application runner with CLI argument parsing, webcam/video file abstraction, keyboard control (`q`/`ESC`), and guaranteed cleanup via `try...finally`. |
| **`evaluate.py`** | Offline benchmark script for dataset folders with `.xmp` ground truth files (e.g., `data/10-gt`). |
| **`evaluate_subject.py`** | Offline benchmark script for UBFC-style datasets (e.g., `data/subject10`) with multi-line `ground_truth.txt` (PPG, HR, Timestep). Generates MAE, RMSE, Pearson $r$, Bias, and high-res comparison plots. |

---

## Installation

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Run the Web Application (Recommended)
Launch the modern medical dashboard interface with real-time video streaming, live BVP oscilloscope, and dynamic Ground Truth benchmarking:
```bash
python manage.py runserver
```
Then navigate to **`http://localhost:8000`** in your browser.
- **Webcam Mode**: Click "Start Stream" to monitor real-time pulse and BPM with face mesh tracking.
- **Video & Ground Truth Benchmark Mode**: Upload a video (e.g. `data/subject10/vid.avi`) and optional ground truth (`data/subject10/ground_truth.txt`). The ground truth curve pre-renders automatically, and real-time prediction tracks alongside it while reporting live **MAE**, **RMSE**, and **Pearson $r$**.

---

### 2. Run with Default CLI Source
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

### 5. Run Offline Evaluation on Datasets
- **For UBFC / Subject Datasets (e.g. `data/subject10`)**:
  ```bash
  python evaluate_subject.py --dataset data/subject10
  ```
- **For XMP Datasets (e.g. `data/10-gt`)**:
  ```bash
  python evaluate.py --dataset data/10-gt
  ```

### 6. CLI Options Reference
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
