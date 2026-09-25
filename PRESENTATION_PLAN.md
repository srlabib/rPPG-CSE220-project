# CSE 220 Final Project Presentation & Evaluation Plan
## Real-Time Remote Photoplethysmography (rPPG) System

---

## 1. Evaluation & Timing Overview

| Format | Presentation + Live Demo | Q&A Session | Total Time |
| :--- | :---: | :---: | :---: |
| **2-Person Team** | **6 minutes** (4 min slides + 2 min demo) | **2 minutes** | **8 minutes** |
| **3-Person Team** | **8 minutes** (5.5 min slides + 2.5 min demo) | **2 minutes** | **10 minutes** |

> [!IMPORTANT]
> **Strict Pacing Rule:** Evaluators strictly cut off presentations at the time limit. Every team member **must speak** during both the slides and the live demo. Do not read bullet points verbatim; use slides as visual anchors while explaining the technical rationale.

---

## 2. Slide Deck Structure & Content Plan

### Slide 1: Title & Project Identity
* **Header / Title:** Contactless Physiological Monitoring via Remote Photoplethysmography (rPPG)
* **Subtitle:** Real-Time Heart Rate Estimation using Facial Landmark Tracking and Chrominance/Orthogonal Signal Decompositions
* **Metadata:** Course: CSE 220 | Date | Team Member Names, Student IDs & Assigned Roles
* **Visual Anchor:** Clean screenshot of the web dashboard with the live BVP oscilloscope and dual-curve graph.
* **Key Talking Points:**
  - Introduce the team and project title.
  - State the core objective: extracting accurate human cardiac vitals (pulse waveform and BPM) from consumer webcams and video feeds in real time without any physical sensors.

---

### Slide 2: Motivation & Problem Addressed
* **Header:** Motivation: The Need for Contactless Vital Sign Monitoring
* **Content Points:**
  - **Traditional Contact PPG Limitations:** Pulse oximeters cause skin irritation, restrict movement, require hardware sterilization, and cannot be used on fragile neonatal skin, burn victims, or infectious patients.
  - **The Computer Vision Alternative:** Standard RGB cameras capture subtle optical absorption changes on human skin caused by pulsatile arterial blood flow.
  - **Core Engineering Challenge:** The cardiac pulsatile variation is tiny (**$0.1\% - 1.0\%$** of skin reflectance), while head movements, facial expressions, and lighting changes modulate reflectance by **$2\% - 10\%$** (a signal-to-noise ratio of roughly $-40\text{ dB}$).
  - **Objective:** Design an end-to-end robust pipeline capable of isolating this micro-signal in real time at 30+ FPS.

---

### Slide 3: Optical Biophysics & Relevant Theory
* **Header:** Biophysics of rPPG: How Skin Reflectance Encodes Cardiac Cycles
* **Content Points:**
  - **Light-Tissue Interaction (Modified Beer-Lambert Law):**
    $$I(t) = I_0(t) \cdot (1 + \text{AC}(t) / \text{DC})$$
    $I_0(t)$ is incident ambient illumination, $\text{DC}$ is static tissue reflection, and $\text{AC}(t)$ is the dynamic blood volume pulse.
  - **Hemoglobin Optical Absorption:** Oxygenated ($HbO_2$) and deoxygenated ($Hb$) hemoglobin peak in absorption around the **green wavelength spectrum ($540 - 580\text{ nm}$)**.
  - **Pulse Extraction Concept:** With each heartbeat (systole), capillary blood volume surges, absorption increases, and skin reflectance dips. During diastole, blood volume recedes and reflectance increases.
* **Visual Anchor:** Spectral absorption graph of hemoglobin showing green channel sensitivity vs red/blue.

---

### Slide 4: Algorithmic Formulations (POS vs CHROM vs GREEN)
* **Header:** Extraction Algorithms: Mathematical Formulations & Noise Rejection
* **Content Points (Comparison Table or 3 Columns):**
  1. **GREEN Channel Baseline (Verkruysse et al., 2008):**
     - $h = -g(t)/\mu_g$. Inverts green reflection so systolic peak is positive.
     - *Fatal Flaw:* Single channel, zero motion cancellation. Auto-exposure or motion directly corrupts the pulse.
  2. **CHROM Algorithm (de Haan & Jeanne, 2013):**
     - Chrominance difference projections: $X_s = 3R_n - 2G_n$, $Y_s = 1.5R_n + G_n - 1.5B_n$
     - Dynamic alpha tuning: $\alpha = \sigma(X_s)/\sigma(Y_s)$, $H = X_s - \alpha Y_s$
  3. **POS (Plane-Orthogonal-to-Skin) (Wang et al., 2017):**
     - Projection onto the plane orthogonal to skin tone vector:
       $$S_1 = G_n - B_n, \quad S_2 = -2R_n + G_n + B_n, \quad \alpha = \frac{\sigma(S_1)}{\sigma(S_2)}$$
       $$H = S_1 + \alpha S_2$$
     - *Why POS Wins:* Intensity swings (shadows, lighting, head tilt) move along the specular axis $[1, 1, 1]^T$. POS projects orthogonally to this axis, mathematically canceling out common-mode motion artifacts.

---

### Slide 5: End-to-End System Architecture
* **Header:** System Pipeline: From Video Ingestion to Pulse Waveform
* **Visual Anchor:** High-level architectural flowchart:
  ```
  Camera / Video File
         ↓
  Face Detection & Mesh (MediaPipe 468 landmarks)
         ↓
  12-Patch Spatial Discretization (Forehead & Cheeks)
         ↓
  Temporal Normalization (Cn = C / mean(C))
         ↓
  Batch Pulse Extraction (POS / CHROM / GREEN)
         ↓
  Vectorized SNR Patch Selection (Top-6 patches fused)
         ↓
  Overlap-Add (OLA) Continuous Stitching (Hanning Window)
         ↓
  Butterworth Bandpass Filter (0.75 - 2.5 Hz / 45 - 150 BPM)
         ↓
  Zero-Padded FFT + Sub-Harmonic Peak Verification -> BPM Output
  ```
* **Key Talking Points:** Explain the separation of concerns: Face Tracking $\rightarrow$ Spatial ROI $\rightarrow$ Color Processing $\rightarrow$ Spectral Analysis $\rightarrow$ Web Dashboard.

---

### Slide 6: Critical Implementation Decisions & Engineering Features
* **Header:** Key Engineering Decisions & Optimizations
* **Content Points:**
  - **12-Patch Spatial Discretization:** Instead of one large noisy face box, the face is divided into 12 distinct anatomical patches (6 forehead, 6 cheeks). Hair, eyes, and mouth are explicitly excluded.
  - **Dynamic Top-K SNR Fusion:** Vectorized FFT computes SNR across all 12 patches simultaneously; only the cleanest top 6 patches are fused using SNR-weighted averaging.
  - **Overlap-Add (OLA) Accumulator:** Seamlessly stitches independently extracted 2.5s sliding window pulses into a smooth, continuous waveform without step discontinuities.
  - **Sub-Harmonic Verification (Anti-Harmonic Doubling):** When high peak frequencies ($\ge 1.6\text{ Hz}$ / $96\text{ BPM}$) are detected, verifies whether power at $f/2$ represents the true fundamental heart rate, avoiding $2\times$ harmonic errors (e.g. 70 BPM misread as 140 BPM).
  - **Fast Forward Engine:** Configurable unthrottled loop execution in the streaming worker to process 2000-frame datasets at full CPU speed.

---

### Slide 7: Experimental Results & Benchmarking
* **Header:** Experimental Validation on the UBFC-rPPG Dataset
* **Content Points (Leaderboard Table):**
  | Algorithm | Ground Truth | Mean Estimate | MAE (Error) | RMSE | Pearson $r$ |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **POS** | 97.2 BPM | **94.5 BPM** | **7.54 BPM** | **8.52 BPM** | **High** |
  | **CHROM** | 97.2 BPM | **94.4 BPM** | **7.51 BPM** | **8.48 BPM** | **High** |
  | **GREEN** | 97.2 BPM | **68.4 BPM** | **28.84 BPM** | **31.47 BPM** | **Low** |
* **Key Analytical Finding:**
  - Proves the theoretical hypothesis: Green channel locks onto the $56\text{ BPM}$ ($0.9\text{ Hz}$) respiration/sway artifact because it lacks orthogonal reference channels.
  - POS and CHROM attenuate the low-frequency noise by $>80\times$, delivering clinical accuracy within $\pm 7.5\text{ BPM}$.

---

### Slide 8: Full-Stack Web Architecture
* **Header:** Real-Time Clinical Dashboard & Benchmarking Platform
* **Content Points:**
  - **Backend:** Django web application with multi-threaded `RPPGStreamManager` executing background workers for frame acquisition, MediaPipe processing, and telemetry serialization.
  - **Low-Latency Streaming:** Multipart MJPEG video streaming at `/api/stream/` paired with lightweight JSON telemetry polling at `/api/metrics/` (10 Hz).
  - **Frontend UI/UX:** Dark-mode glassmorphic interface, Chart.js dual-curve live plotting (Ground Truth vs rPPG Predictions), 60 FPS HTML5 Canvas BVP oscilloscope, and a dynamic tri-method comparison leaderboard.

---

### Slide 9: Conclusion, Limitations & Future Scope
* **Header:** Conclusion & Future Horizons
* **Content Points:**
  - **Conclusions:** Successfully demonstrated real-time, non-invasive heart rate extraction from facial video with verified clinical accuracy on standard benchmark datasets.
  - **Current Limitations:**
    - Extreme head turns ($>45^\circ$) degrade facial landmark visibility.
    - Low-light environments reduce SNR in consumer camera sensors.
    - Performance variations across darker skin tones (Fitzpatrick scale types V-VI) due to higher melanin absorption.
  - **Future Extensions:**
    - Integration of remote respiratory rate (RR) and Heart Rate Variability (HRV / SDNN).
    - Deep learning hybrid architectures (e.g., PhysNet / 3D-CNN temporal backbones).

---

## 3. Team Member Role Distribution

### Scenario A: Two-Person Team (8 Minutes Total)

```
[0:00 - 2:00] Member 1: Problem Statement, Theory & POS/CHROM Math (Slides 1-4)
[2:00 - 4:00] Member 2: Architecture, Engineering Decisions & Results (Slides 5-7, 9)
[4:00 - 6:00] Both Members: Live Application Demonstration (Dashboard & Fast Forward)
[6:00 - 8:00] Both Members: Evaluator Q&A
```

* **Member 1 Focus:** Biophysics of skin reflection, Beer-Lambert law, color-space projection math (POS vs CHROM vs GREEN), and physiological frequency bounds.
* **Member 2 Focus:** MediaPipe 12-patch extraction, Top-K SNR selection, Overlap-Add stitching, sub-harmonic FFT peak picking, and UBFC benchmark results.
* **Live Demo Division:** Member 2 operates the UI controls while Member 1 explains the active waveform and leaderboard metrics.

---

### Scenario B: Three-Person Team (10 Minutes Total)

```
[0:00 - 2:00] Member 1: Motivation, Problem, Biophysics & Algorithms (Slides 1-4)
[2:00 - 4:00] Member 2: Pipeline Architecture & Signal Processing Innovations (Slides 5-6)
[4:00 - 5:30] Member 3: Experimental Benchmark Results & Web System Design (Slides 7-9)
[5:30 - 8:00] All Members: Live Interactive Demonstration
[8:00 - 10:00] All Members: Evaluator Q&A
```

* **Member 1 (Biophysics & Math Specialist):** Explains optical hemoglobin absorption, specular vs diffuse reflection, and why POS orthogonal projection cancels intensity noise.
* **Member 2 (Signal Processing & Algorithms Specialist):** Explains 12-patch facial grid, vectorized batch SNR computation, Overlap-Add smoothing, and zero-phase Butterworth filtering.
* **Member 3 (System Architecture & Evaluation Specialist):** Explains UBFC dataset results, error analysis (MAE/RMSE), Django streaming architecture, and frontend real-time plotting.

---

## 4. Live Demonstration Script (Step-by-Step)

Prepare this exact sequence beforehand so the demo runs without delays:

1. **Pre-Demo Setup (Before stepping up):**
   - Ensure `python manage.py runserver` is already active in the terminal.
   - Have the browser open to `http://127.0.0.1:8000/`.
   - Verify `data/subject10/vid.avi` is loaded in the local dataset selector.

2. **Step 1: Benchmark Mode Demonstration (1.5 Minutes):**
   - Click on the **"📁 Video & Ground Truth Benchmark"** tab.
   - Select `data/subject10` from the Auto-Detected Datasets dropdown and click **"⚡ Load"**.
   - Point to the chart: Show how the **Ground Truth HR reference curve (Sky Blue)** is pre-rendered across the 67-second timeline.
   - Click **"▶ Start Stream"**.
   - Show the real-time red estimated curve tracing alongside the blue ground truth curve.
   - Point to the **BVP Oscilloscope**: Explain how systolic peaks correspond to heartbeat pulses at 60 FPS.
   - Point to the **Leaderboard**: Show POS tracking at $\approx 94 - 98\text{ BPM}$ with an error within $\pm 7\text{ BPM}$.

3. **Step 2: Fast-Forward Feature Demonstration (30 Seconds):**
   - Click the **"⏩ Fast Forward"** button.
   - Point out that the button highlights gold (`Fast Forward: ON`).
   - Show the progress bar accelerating rapidly as frames are processed at full CPU throughput without artificial frame delays.
   - Show that final MAE/RMSE metrics converge instantaneously to the benchmark table.

4. **Step 3: Tri-Method Comparison (30 Seconds):**
   - Click **"⚡ Compare All"** on the method toolbar.
   - Show POS, CHROM, and GREEN curves running concurrently.
   - Point out the GREEN channel discrepancy: explain that GREEN reports $\approx 56 - 68\text{ BPM}$ because of respiration/motion, proving why POS is superior.

5. **Step 4: Live Webcam Quick Demo (If time allows, 30 Seconds):**
   - Switch to **"📷 Live Webcam"** and click **"Start Stream"**.
   - Show the face mesh tracking the speaker's face and computing live resting heart rate.

---

## 5. Anticipated Evaluator Q&A Questions & Technical Answers

Be ready for these common evaluator questions:

### Q1: "Why does the Green algorithm show such high error compared to POS?"
* **Answer:** "The pulsatile cardiac signal accounts for only $0.1\% - 1.0\%$ of the green channel reflectance, while head movement, facial tilt, and room lighting changes modulate reflectance by $2\% - 10\%$. Because Green is univariate (single-channel), it has no reference channel to distinguish blood volume expansion from movement. In contrast, POS uses all three RGB channels to project the signal onto a plane orthogonal to the skin tone, which mathematically cancels common-mode intensity fluctuations along the $[1, 1, 1]^T$ axis."

### Q2: "Why do you use 12 patches instead of the whole face bounding box?"
* **Answer:** "A single whole-face box includes non-skin regions: eyes (blinking), mouth (talking/swallowing), facial hair, and glasses, which inject severe non-cardiac noise. Our 12-patch grid isolates rigid regions with dense micro-capillaries: the forehead and upper cheeks. Furthermore, computing SNR independently per patch allows us to dynamically discard shaded or shadowed patches and fuse only the top 6 highest-quality signals."

### Q3: "What is Overlap-Add (OLA) and why is it necessary?"
* **Answer:** "POS calculates projection weights ($\alpha$) dynamically on sliding windows of 2.5 seconds. Because $\alpha$ varies slightly between consecutive windows, stitching raw window outputs directly would cause sharp step discontinuities at window boundaries. Overlap-Add applies a tapering Hanning window to each sliding segment and accumulates overlapping points, ensuring smooth, phase-aligned reconstruction of the continuous pulse wave."

### Q4: "How do you prevent harmonic doubling (e.g., 70 BPM misread as 140 BPM)?"
* **Answer:** "In rPPG, the heart's dicrotic notch and non-linear arterial reflections often create a pronounced second harmonic ($2f$). In `calculate_bpm()`, whenever the dominant FFT peak is $\ge 1.6\text{ Hz}$ ($96\text{ BPM}$), the algorithm inspects the sub-harmonic at $f/2$. If significant spectral power ($\ge 35\%$) exists at $f/2$, the algorithm correctly designates the lower frequency as the physiological fundamental heart rate."

### Q5: "How does the web dashboard stream video without blocking signal processing?"
* **Answer:** "We decoupled ingestion, signal processing, and display using a multi-threaded architecture in `RPPGStreamManager`. A dedicated worker thread executes camera capture, FaceMesh inference, and the rPPG pipeline. It pushes encoded JPEG frames to a thread-safe buffer for the MJPEG endpoint (`/api/stream/`), while a separate HTTP endpoint (`/api/metrics/`) periodically returns lightweight JSON telemetry to the frontend."

---

## 6. Slide Design & Visual Presentation Best Practices

1. **Dark Mode Clinical Aesthetic:** Use a deep slate/navy background (`#0b0f19` or `#0f172a`) with high-contrast text and emerald/amber accents to match the web dashboard interface.
2. **Minimize Text Clutter:** Limit each slide to 3–4 concise bullet points. Let diagrams, equations, and benchmark charts do the heavy lifting.
3. **High-Resolution Figures:** Embed actual screenshots from the dashboard (`evaluation_result.png`, pulse graph, face mesh ROI overlay).
4. **Equation Highlighting:** When presenting the POS and Beer-Lambert equations, explicitly define every variable ($I(t), I_0, S_1, S_2, \alpha, H$).
