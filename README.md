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
| **`config.py`** | Centralized configuration holding 12-patch facial indices, physiological frequency boundaries (0.75 Hz - 2.5 Hz / 45 - 150 BPM), POS window (2.5s), and camera resolution defaults. |
| **`signal_processing.py`** | Pure mathematical signal processing functions: <br>• `execute_pos_algorithm()` / `execute_pos_algorithm_batch()`: Wang et al. (2017) POS projection.<br>• `butter_bandpass_filter()`: 6th-order zero-phase Butterworth filter (`scipy.signal.filtfilt`).<br>• `calculate_bpm()`: Zero-padded 2048-pt FFT with linear detrending, Hann windowing, and sub-harmonic verification to eliminate harmonic doubling. |
| **`face_processor.py`** | `FaceROIProcessor` class encapsulating MediaPipe FaceMesh lifecycle, extracting forehead and cheek skin regions, calculating mean RGB color vectors, and rendering semi-transparent overlays. |
| **`visualizer.py`** | `create_pulse_graph()` for real-time oscilloscope display and `draw_hud()` for clean on-screen status readouts. |
| **`rppg_pipeline.py`** | `RPPGPipeline` class maintaining sliding RGB windows, Top-K SNR selection, Overlap-Add pulse accumulation, and temporal smoothing (moving median + EMA) for stable heart rate readout. |
| **`VideoProcessing.py`** | Application runner with CLI argument parsing, webcam/video file abstraction, keyboard control (`q`/`ESC`), and guaranteed cleanup via `try...finally`. |
| **`server.py`** | FastAPI server providing the localhost web studio, MJPEG streaming, WebSocket telemetry, and video upload API. |
| **`web_pipeline.py`** | Thread-safe background video acquisition and rPPG processing pipeline engine for web streaming. |
| **`static/`** | Web dashboard assets: modern dark-mode UI, HTML5 Canvas oscilloscope, live BPM monitor, and control suite. |
| **`evaluate.py`** | Offline benchmark script for dataset folders with `.xmp` ground truth files (e.g., `data/10-gt`). |
| **`evaluate_subject.py`** | Offline benchmark script for UBFC-style datasets (e.g., `data/subject10`) with multi-line `ground_truth.txt` (PPG, HR, Timestep). Generates MAE, RMSE, Pearson $r$, Bias, and high-res comparison plots. |

---

## How to Run

### 1. Launch the Localhost Web Frontend (Recommended)
Start the web dashboard server:
```bash
python server.py
# Or using conda environment:
conda run -n rppg-env python server.py
```
Open your browser and navigate to:
```
http://localhost:8000
```
**Features of the Web Studio:**
- **Live Facial Tracking**: Real-time video with toggleable green skin ROI polygon overlays.
- **Dynamic Vitals Card**: Large digital BPM display with pulsating heart icon physically synchronized to heart rate.
- **Oscilloscope Waveform**: 60 FPS Canvas visualizer rendering the zero-phase Butterworth bandpass filtered pulse wave.
- **Signal Quality Diagnostics**: Real-time SNR gauge and live 12-patch spatial sub-region fusion matrix.
- **Data Folder Explorer (`📁 Browse data/`)**: Recursively discovers all video files across `data/` and all nested subfolders (e.g., `data/10-gt/vid.avi`, `data/subject10/vid.avi`), with real-time search, subfolder pills, and instant rescan.
- **Manual Path Verification**: Enter any custom video path directly with automated resolution and frame rate validation.
- **Source Switching**: Seamlessly toggle between webcams (0, 1, 2) and stored/uploaded videos.
- **Drag-and-Drop Uploader**: Upload any `.mp4`, `.avi`, `.mov`, `.webm` file for instant analysis.
- **Session Recording & CSV Export**: Record pulse waveform and BPM time-series data with one click.

---

### 2. Run CLI Application with Default Source
By default, the script looks for `data/10-gt/vid.avi`, the first available video in `data/` (e.g., `data/gtdump.avi`), or connects to camera `0`:
```bash
python VideoProcessing.py
```

### 3. Interactively Select a Video from data/
List all videos discovered in `data/` and choose by number:
```bash
python VideoProcessing.py --select
```
Or view the list of discovered videos:
```bash
python VideoProcessing.py --list-videos
```

### 4. Run with a Specific Video File
```bash
python VideoProcessing.py --source "path/to/video.mp4"
```

### 5. Run with Webcam / DroidCam
Use camera index (e.g. `0` for integrated webcam, `1` or `2` for DroidCam):
```bash
python VideoProcessing.py --source 0
```

### 6. Enable Camera Hardware Settings (Windows)
To manually disable auto-exposure on webcams / DroidCam:
```bash
python VideoProcessing.py --source 0 --camera-settings
```

### 7. Run Offline Evaluation on Datasets
- **For UBFC / Subject Datasets (e.g. `data/subject10`)**:
  ```bash
  python evaluate_subject.py --dataset data/subject10
  ```
- **For XMP Datasets (e.g. `data/10-gt`)**:
  ```bash
  python evaluate.py --dataset data/10-gt
  ```

### 8. CLI Options Reference
```
options:
  -h, --help            Show this help message and exit
  -s, --source SOURCE   Path to video file or camera index (default: data/gtdump.avi or 0)
  --list-videos         List all video files discovered in data/ directory and exit
  --select              Interactively select from video files discovered in data/
  --width WIDTH         Camera frame width (default: 640 for webcams, 1280 for files)
  --height HEIGHT       Camera frame height (default: 480 for webcams, 720 for files)
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
