/**
 * rPPG VitalSigns Studio - Frontend Controller
 * Handles Chart.js dual-curve plotting, 60fps canvas oscilloscope, file upload, and real-time metrics telemetry.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentMode = "webcam"; // "webcam" or "upload"
    let isRunning = false;
    let isPaused = false;
    let telemetryInterval = null;
    let uploadedVideoPath = null;
    let groundTruthData = null;
    let startupGraceCount = 0;

    // Pulse Wave Oscilloscope Canvas
    const canvas = document.getElementById("pulseCanvas");
    const ctx = canvas.getContext("2d");
    let pulseBuffer = [];

    // Resize canvas properly for retina displays
    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // Chart.js: BPM vs Time
    const chartCtx = document.getElementById("bpmChart").getContext("2d");
    const bpmChart = new Chart(chartCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Ground Truth (Reference)",
                    data: [],
                    borderColor: "#38bdf8", // Crisp Sky Blue
                    backgroundColor: "transparent",
                    borderWidth: 1.5, // Thin, sharp, standard line
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0, // Sharp line segments
                    hidden: true,
                },
                {
                    label: "Predicted rPPG (BPM)",
                    data: [],
                    borderColor: "#10b981", // Vibrant Emerald
                    backgroundColor: "transparent",
                    borderWidth: 1.5, // Thin, sharp, standard line
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0, // Sharp line segments
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            elements: {
                line: {
                    tension: 0 // Sharp, precise clinical line segments
                }
            },
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    title: { display: true, text: "Time (seconds)", color: "#9ca3af" },
                    grid: { color: "rgba(55, 65, 81, 0.35)" },
                    ticks: { color: "#9ca3af" }
                },
                y: {
                    grace: "5%",
                    suggestedMin: 60,
                    suggestedMax: 100,
                    title: { display: true, text: "Heart Rate (BPM)", color: "#9ca3af" },
                    grid: { color: "rgba(55, 65, 81, 0.35)" },
                    ticks: { color: "#9ca3af" }
                }
            },
            plugins: {
                legend: {
                    display: false // Clean header badge indicators used instead
                },
                tooltip: {
                    mode: "index",
                    intersect: false
                }
            }
        }
    });

    // Elements
    const videoFeed = document.getElementById("videoFeed");
    const videoPlaceholder = document.getElementById("videoPlaceholder");
    const videoProgressBar = document.getElementById("videoProgressBar");
    const bpmValue = document.getElementById("bpmValue");
    const heartIcon = document.getElementById("heartIcon");
    const sqiValue = document.getElementById("sqiValue");
    const statusText = document.getElementById("statusText");
    const statusDot = document.getElementById("statusDot");

    // Benchmark Metric Elements
    const metricGtBpm = document.getElementById("metricGtBpm");
    const metricError = document.getElementById("metricError");
    const metricMae = document.getElementById("metricMae");
    const metricRmse = document.getElementById("metricRmse");
    const metricPearson = document.getElementById("metricPearson");

    // Controls
    const btnStart = document.getElementById("btnStart");
    const btnPause = document.getElementById("btnPause");
    const btnStop = document.getElementById("btnStop");

    // Mode Switchers
    const tabWebcam = document.getElementById("tabWebcam");
    const tabUpload = document.getElementById("tabUpload");
    const uploadPanel = document.getElementById("uploadPanel");

    // Upload elements
    const videoFileInput = document.getElementById("videoFileInput");
    const gtFileInput = document.getElementById("gtFileInput");
    const videoFilePreview = document.getElementById("videoFilePreview");
    const gtFilePreview = document.getElementById("gtFilePreview");
    const btnSubmitUpload = document.getElementById("btnSubmitUpload");
    const uploadStatus = document.getElementById("uploadStatus");

    // Local Dataset Auto-Scan elements
    const localDatasetSelect = document.getElementById("localDatasetSelect");
    const btnLoadSelectedDataset = document.getElementById("btnLoadSelectedDataset");
    const btnRefreshDatasets = document.getElementById("btnRefreshDatasets");

    // Mode Switching
    tabWebcam.addEventListener("click", () => {
        if (isRunning) stopStream();
        currentMode = "webcam";
        tabWebcam.classList.add("active");
        tabUpload.classList.remove("active");
        uploadPanel.classList.remove("active");
        resetCharts();
        bpmChart.data.datasets[0].hidden = true; // hide GT in webcam mode
        const gtBadge = document.getElementById("gtLegendBadge");
        if (gtBadge) gtBadge.style.display = "none";
        bpmChart.update();
    });

    tabUpload.addEventListener("click", () => {
        if (isRunning) stopStream();
        currentMode = "upload";
        tabUpload.classList.add("active");
        tabWebcam.classList.remove("active");
        uploadPanel.classList.add("active");
        resetCharts();
        bpmChart.data.datasets[0].hidden = false;
        const gtBadge = document.getElementById("gtLegendBadge");
        if (gtBadge) gtBadge.style.display = "flex";
        bpmChart.update();
        fetchDatasetList(); // Auto scan /data folder
    });

    // File selection UI feedback
    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            videoFilePreview.textContent = e.target.files[0].name;
            uploadedVideoPath = null; // require new upload
            uploadStatus.textContent = "File selected. Ready to upload or start stream.";
        }
    });

    gtFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            gtFilePreview.textContent = e.target.files[0].name;
            groundTruthData = null;
        }
    });

    // Perform Upload Function
    async function performUpload() {
        if (!videoFileInput.files.length) {
            uploadStatus.textContent = "Please select a video file (.avi, .mp4) first.";
            return false;
        }

        const formData = new FormData();
        formData.append("video", videoFileInput.files[0]);
        if (gtFileInput.files.length) {
            formData.append("ground_truth", gtFileInput.files[0]);
        }

        uploadStatus.textContent = "Uploading video & parsing ground truth... Please wait.";
        btnSubmitUpload.disabled = true;
        btnStart.disabled = true;

        try {
            const resp = await fetch("/api/upload/", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            btnSubmitUpload.disabled = false;
            btnStart.disabled = false;

            if (!resp.ok || !data.success) {
                uploadStatus.textContent = "Upload error: " + (data.error || "Unknown error");
                return false;
            }

            uploadedVideoPath = data.video_path;
            groundTruthData = data.gt_data;

            uploadStatus.textContent = `✅ Ready! Video: ${data.video_name}` + 
                (data.has_ground_truth ? ` | GT Loaded (${data.gt_data.sample_count} pts, Mean: ${data.gt_data.mean_hr.toFixed(1)} BPM)` : "");

            // Pre-populate Ground Truth Curve in Chart!
            if (data.has_ground_truth && groundTruthData) {
                prepopulateGroundTruth(groundTruthData);
            }
            return true;

        } catch (err) {
            btnSubmitUpload.disabled = false;
            btnStart.disabled = false;
            uploadStatus.textContent = "Upload failed: " + err.message;
            return false;
        }
    }

    btnSubmitUpload.addEventListener("click", performUpload);

    // Auto-Scan Dataset Lister & Instant Loader (0s load, no browser upload)
    async function fetchDatasetList() {
        if (!localDatasetSelect) return;
        localDatasetSelect.innerHTML = '<option value="">Scanning data/ folder...</option>';
        try {
            const resp = await fetch("/api/datasets/");
            const data = await resp.json();
            if (resp.ok && data.success && data.datasets.length > 0) {
                localDatasetSelect.innerHTML = '<option value="">-- Choose local dataset to load (0s) --</option>';
                data.datasets.forEach(d => {
                    const opt = document.createElement("option");
                    opt.value = d.id;
                    opt.textContent = `📁 ${d.display_name}`;
                    localDatasetSelect.appendChild(opt);
                });
            } else {
                localDatasetSelect.innerHTML = '<option value="">No videos found in data/ folder</option>';
            }
        } catch (err) {
            localDatasetSelect.innerHTML = '<option value="">Error scanning data/ folder</option>';
        }
    }

    async function loadSelectedDataset() {
        const datasetId = localDatasetSelect.value;
        if (!datasetId) {
            uploadStatus.textContent = "Please select a dataset from the dropdown first.";
            return;
        }

        uploadStatus.textContent = `Loading ${datasetId} instantly from disk...`;
        if (btnLoadSelectedDataset) btnLoadSelectedDataset.disabled = true;

        try {
            const resp = await fetch("/api/control/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "load_dataset", dataset_id: datasetId })
            });
            const data = await resp.json();
            if (btnLoadSelectedDataset) btnLoadSelectedDataset.disabled = false;

            if (!resp.ok || !data.success) {
                uploadStatus.textContent = "Failed to load: " + (data.error || "Unknown error");
                return;
            }

            uploadedVideoPath = data.video_path;
            groundTruthData = data.gt_data;
            videoFilePreview.textContent = data.video_name;
            gtFilePreview.textContent = data.has_ground_truth ? "Local Ground Truth Loaded" : "None";

            uploadStatus.textContent = `✅ Ready! Loaded ${data.video_name}` + 
                (data.has_ground_truth ? ` | Duration: ${data.gt_data.duration.toFixed(1)}s, ${data.gt_data.sample_count} GT pts (Mean: ${data.gt_data.mean_hr.toFixed(1)} BPM). Click 'Start Stream'!` : ". Click 'Start Stream'!");

            if (data.has_ground_truth && groundTruthData) {
                prepopulateGroundTruth(groundTruthData);
            }
        } catch (err) {
            if (btnLoadSelectedDataset) btnLoadSelectedDataset.disabled = false;
            uploadStatus.textContent = "Error loading dataset: " + err.message;
        }
    }

    if (btnRefreshDatasets) {
        btnRefreshDatasets.addEventListener("click", fetchDatasetList);
    }
    if (btnLoadSelectedDataset) {
        btnLoadSelectedDataset.addEventListener("click", loadSelectedDataset);
    }
    if (localDatasetSelect) {
        localDatasetSelect.addEventListener("change", () => {
            if (localDatasetSelect.value) {
                loadSelectedDataset();
            }
        });
    }
    // Also auto-fetch immediately on initial page load
    fetchDatasetList();

    function prepopulateGroundTruth(gt) {
        resetCharts();
        const gtPoints = [];
        for (let i = 0; i < gt.timestamps.length; i++) {
            gtPoints.push({ x: gt.timestamps[i], y: gt.hr[i] });
        }
        bpmChart.data.datasets[0].data = gtPoints;
        bpmChart.data.datasets[0].hidden = false;
        const gtBadge = document.getElementById("gtLegendBadge");
        if (gtBadge) gtBadge.style.display = "flex";
        
        // Auto scale x axis
        bpmChart.options.scales.x.min = 0;
        bpmChart.options.scales.x.max = Math.ceil(gt.duration || 60);

        // Dynamically adjust Y scale bounds to fit data tightly so curves are not vertically squeezed
        if (gt.min_hr !== undefined && gt.max_hr !== undefined && gt.min_hr > 0) {
            const span = gt.max_hr - gt.min_hr;
            const margin = Math.max(4, span * 0.15);
            bpmChart.options.scales.y.suggestedMin = Math.max(30, Math.floor(gt.min_hr - margin));
            bpmChart.options.scales.y.suggestedMax = Math.min(220, Math.ceil(gt.max_hr + margin));
        }

        bpmChart.update();
    }

    function resetCharts() {
        bpmChart.data.datasets[0].data = [];
        bpmChart.data.datasets[1].data = [];
        bpmChart.options.scales.x.min = 0;
        bpmChart.options.scales.x.max = undefined;
        bpmChart.options.scales.y.suggestedMin = 60;
        bpmChart.options.scales.y.suggestedMax = 100;
        bpmChart.update();
        pulseBuffer = [];
        metricGtBpm.textContent = "--";
        metricError.textContent = "--";
        metricMae.textContent = "--";
        metricRmse.textContent = "--";
        metricPearson.textContent = "--";
        bpmValue.textContent = "--";
        videoProgressBar.style.width = "0%";
    }

    // Stream Controls
    btnStart.addEventListener("click", async () => {
        if (currentMode === "webcam") {
            statusText.textContent = "Starting camera...";
            const resp = await fetch("/api/control/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "start_webcam", camera_index: 0 })
            });
            if (resp.ok) {
                startDisplay();
            } else {
                alert("Failed to start webcam. Please verify camera is connected.");
            }
        } else {
            // If user selected file in input but didn't click separate upload button, auto upload!
            if (!uploadedVideoPath) {
                if (videoFileInput.files.length > 0) {
                    const ok = await performUpload();
                    if (!ok) return;
                } else {
                    alert("Please select a video file to upload, or click '⚡ Load Sample (UBFC Subject 10)'.");
                    return;
                }
            }

            statusText.textContent = "Starting video playback...";
            const payload = {
                action: "start_video",
                video_path: uploadedVideoPath,
                gt_times: groundTruthData ? groundTruthData.raw_times : [],
                gt_hr: groundTruthData ? groundTruthData.raw_hr : []
            };
            const resp = await fetch("/api/control/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (resp.ok) {
                startDisplay();
            } else {
                const err = await resp.json();
                alert("Failed to start video: " + (err.error || "File error"));
            }
        }
    });

    btnPause.addEventListener("click", async () => {
        await fetch("/api/control/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "pause" })
        });
        isPaused = !isPaused;
        btnPause.textContent = isPaused ? "▶ Resume" : "⏸ Pause";
    });

    btnStop.addEventListener("click", () => {
        stopStream();
    });

    function startDisplay() {
        isRunning = true;
        isPaused = false;
        startupGraceCount = 20; // 2 seconds grace period for camera / video opening
        btnStart.style.display = "none";
        btnPause.style.display = "inline-flex";
        btnStop.style.display = "inline-flex";
        btnPause.textContent = "⏸ Pause";

        // Show stream
        videoPlaceholder.style.display = "none";
        videoFeed.style.display = "block";
        videoFeed.src = "/api/stream/?t=" + new Date().getTime();

        statusDot.className = "status-dot";

        // Clear prediction dataset, keep pre-loaded ground truth if present
        bpmChart.data.datasets[1].data = [];
        lastTimePoint = -1;
        bpmChart.update();

        // Start polling telemetry
        if (telemetryInterval) clearInterval(telemetryInterval);
        telemetryInterval = setInterval(fetchTelemetry, 100);
    }

    async function stopStream() {
        isRunning = false;
        if (telemetryInterval) {
            clearInterval(telemetryInterval);
            telemetryInterval = null;
        }

        await fetch("/api/control/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "stop" })
        });

        btnStart.style.display = "inline-flex";
        btnPause.style.display = "none";
        btnStop.style.display = "none";

        videoFeed.src = "";
        videoFeed.style.display = "none";
        videoPlaceholder.style.display = "block";
        statusDot.className = "status-dot idle";
        statusText.textContent = "Stopped";
    }

    // Telemetry consumer
    let lastTimePoint = -1;
    async function fetchTelemetry() {
        try {
            const resp = await fetch("/api/metrics/");
            const data = await resp.json();

            // Status updates
            statusText.textContent = data.status_text || "Processing...";
            if (data.face_detected) {
                statusDot.className = "status-dot";
            } else {
                statusDot.className = "status-dot warning";
            }

            // BPM readout
            if (data.current_bpm) {
                bpmValue.textContent = data.current_bpm.toFixed(1);
                // Adjust heart pulse animation speed to match BPM
                const beatDuration = (60.0 / data.current_bpm).toFixed(2);
                heartIcon.style.animationDuration = `${beatDuration}s`;
            } else {
                bpmValue.textContent = "--";
            }

            // SQI
            sqiValue.textContent = `${data.sqi}%`;

            // Progress
            if (data.progress !== undefined) {
                videoProgressBar.style.width = `${data.progress}%`;
            }

            // Pulse wave buffer for oscilloscope
            if (data.pulse_history && data.pulse_history.length > 0) {
                pulseBuffer = data.pulse_history;
            }

            // Append live predicted point to Chart.js
            if (data.current_bpm && data.timestamp > lastTimePoint + 0.3) {
                lastTimePoint = data.timestamp;
                bpmChart.data.datasets[1].data.push({
                    x: data.timestamp,
                    y: data.current_bpm
                });

                // Auto slide window in webcam mode
                if (currentMode === "webcam" && data.timestamp > 30) {
                    bpmChart.options.scales.x.min = data.timestamp - 30;
                    bpmChart.options.scales.x.max = data.timestamp;
                }
                bpmChart.update("none");
            }

            // Performance Benchmark Metrics (in Upload Mode with GT)
            if (data.metrics) {
                const m = data.metrics;
                metricGtBpm.textContent = m.latest_gt !== null ? `${m.latest_gt} BPM` : "--";
                metricError.textContent = m.current_error !== null ? `±${m.current_error} BPM` : "--";
                metricMae.textContent = m.mae !== null ? `${m.mae} BPM` : "--";
                metricRmse.textContent = m.rmse !== null ? `${m.rmse} BPM` : "--";
                metricPearson.textContent = m.pearson_r !== null ? m.pearson_r : "--";
            }

            // Grace period check: don't stop prematurely on startup
            if (startupGraceCount > 0) {
                startupGraceCount--;
            } else if (!data.running && isRunning) {
                stopStream();
                statusText.textContent = (currentMode === "upload") ? "Video Completed" : "Stream Stopped";
            }

        } catch (err) {
            // Network glitch
        }
    }

    // 60 FPS Oscilloscope Canvas Drawing Loop
    function drawOscilloscope() {
        requestAnimationFrame(drawOscilloscope);

        const rect = canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        ctx.clearRect(0, 0, w, h);

        // Draw grid
        ctx.strokeStyle = "rgba(55, 65, 81, 0.25)";
        ctx.lineWidth = 1;
        const step = 30;
        for (let x = 0; x < w; x += step) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += step) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Center zero-line
        ctx.strokeStyle = "rgba(16, 185, 129, 0.15)";
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        if (pulseBuffer.length < 2) {
            ctx.fillStyle = "rgba(156, 163, 175, 0.45)";
            ctx.font = "12px Inter, system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(isRunning ? "Extracting Butterworth BVP pulse..." : "BVP Oscilloscope Idle", w / 2, h / 2 - 8);
            return;
        }

        // Draw pulse waveform with glow
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2.2;
        ctx.shadowColor = "rgba(16, 185, 129, 0.6)";
        ctx.shadowBlur = 8;
        ctx.beginPath();

        const n = pulseBuffer.length;
        const dx = w / (n - 1);
        const centerY = h / 2;
        const scaleY = h * 0.38;

        for (let i = 0; i < n; i++) {
            const x = i * dx;
            const y = centerY - (pulseBuffer[i] * scaleY);
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Subtle gradient under the pulse wave
        ctx.lineTo(w, centerY);
        ctx.lineTo(0, centerY);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, centerY - scaleY, 0, centerY + scaleY);
        grad.addColorStop(0, "rgba(16, 185, 129, 0.12)");
        grad.addColorStop(1, "rgba(16, 185, 129, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();
    }

    requestAnimationFrame(drawOscilloscope);

    // Modal format toggles
    const formatInfoBtn = document.getElementById("formatInfoBtn");
    const formatModal = document.getElementById("formatModal");
    const modalClose = document.getElementById("modalClose");

    formatInfoBtn.addEventListener("click", () => formatModal.classList.add("active"));
    modalClose.addEventListener("click", () => formatModal.classList.remove("active"));
    formatModal.addEventListener("click", (e) => {
        if (e.target === formatModal) formatModal.classList.remove("active");
    });
});
