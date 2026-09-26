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
    let rawPulseBuffer = [];

    let currentMethod = "pos"; // "pos", "chrom", "omit", or "all"

    // Resize canvas properly for retina displays
    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // Chart.js: BPM vs Time (Supports GT, POS, CHROM, OMIT)
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
                    borderWidth: 1.8,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    hidden: true,
                },
                {
                    label: "POS (Wang et al., 2017)",
                    data: [],
                    borderColor: "#eb7a02", // Warm Amber/Orange
                    backgroundColor: "transparent",
                    borderWidth: 1.6,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    hidden: false,
                },
                {
                    label: "CHROM (de Haan et al., 2013)",
                    data: [],
                    borderColor: "#10b981", // Emerald Neon
                    backgroundColor: "transparent",
                    borderWidth: 1.6,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    hidden: true,
                },
                {
                    label: "OMIT (Álvarez et al., 2023)",
                    data: [],
                    borderColor: "#a855f7", // Electric Violet
                    backgroundColor: "transparent",
                    borderWidth: 1.6,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    hidden: true,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            elements: {
                line: {
                    tension: 0
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
                    display: false // Interactive header badge indicators used instead
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

    // Algorithm Switcher & Badges
    const btnMethodPos = document.getElementById("btnMethodPos");
    const btnMethodChrom = document.getElementById("btnMethodChrom");
    const btnMethodOmit = document.getElementById("btnMethodOmit") || document.getElementById("btnMethodGreen");
    const btnMethodAll = document.getElementById("btnMethodAll");
    const algoButtons = [btnMethodPos, btnMethodChrom, btnMethodOmit, btnMethodAll].filter(Boolean);

    const videoMethodBadge = document.getElementById("videoMethodBadge");
    const oscMethodBadge = document.getElementById("oscMethodBadge");
    const vitalMethodLabel = document.getElementById("vitalMethodLabel");
    const metricsMethodLabel = document.getElementById("metricsMethodLabel");

    // Layout Containers
    const singleVitalsGrid = document.getElementById("singleVitalsGrid");
    const compareVitalsGrid = document.getElementById("compareVitalsGrid");
    const singleMetricsContainer = document.getElementById("singleMetricsContainer");
    const compareMetricsContainer = document.getElementById("compareMetricsContainer");

    // Compare All Vitals Elements
    const posBpmValue = document.getElementById("posBpmValue");
    const chromBpmValue = document.getElementById("chromBpmValue");
    const omitBpmValue = document.getElementById("omitBpmValue") || document.getElementById("greenBpmValue");
    const compareSqiValue = document.getElementById("compareSqiValue");
    const compareRefGtBpm = document.getElementById("compareRefGtBpm");

    // Comparison Table Elements
    const tblPosBpm = document.getElementById("tblPosBpm");
    const tblPosError = document.getElementById("tblPosError");
    const tblPosMae = document.getElementById("tblPosMae");
    const tblPosRmse = document.getElementById("tblPosRmse");
    const tblPosPearson = document.getElementById("tblPosPearson");
    const tblPosBadge = document.getElementById("tblPosBadge");
    const rowMetricPos = document.getElementById("rowMetricPos");

    const tblChromBpm = document.getElementById("tblChromBpm");
    const tblChromError = document.getElementById("tblChromError");
    const tblChromMae = document.getElementById("tblChromMae");
    const tblChromRmse = document.getElementById("tblChromRmse");
    const tblChromPearson = document.getElementById("tblChromPearson");
    const tblChromBadge = document.getElementById("tblChromBadge");
    const rowMetricChrom = document.getElementById("rowMetricChrom");

    const tblOmitBpm = document.getElementById("tblOmitBpm") || document.getElementById("tblGreenBpm");
    const tblOmitError = document.getElementById("tblOmitError") || document.getElementById("tblGreenError");
    const tblOmitMae = document.getElementById("tblOmitMae") || document.getElementById("tblGreenMae");
    const tblOmitRmse = document.getElementById("tblOmitRmse") || document.getElementById("tblGreenRmse");
    const tblOmitPearson = document.getElementById("tblOmitPearson") || document.getElementById("tblGreenPearson");
    const tblOmitBadge = document.getElementById("tblOmitBadge") || document.getElementById("tblGreenBadge");
    const rowMetricOmit = document.getElementById("rowMetricOmit") || document.getElementById("rowMetricGreen");

    // Chart Legend Badges
    const gtLegendBadge = document.getElementById("gtLegendBadge");
    const posLegendBadge = document.getElementById("posLegendBadge");
    const chromLegendBadge = document.getElementById("chromLegendBadge");
    const omitLegendBadge = document.getElementById("omitLegendBadge") || document.getElementById("greenLegendBadge");

    // Benchmark Metric Elements (Single Mode)
    const metricGtBpm = document.getElementById("metricGtBpm");
    const metricError = document.getElementById("metricError");
    const metricMae = document.getElementById("metricMae");
    const metricRmse = document.getElementById("metricRmse");
    const metricPearson = document.getElementById("metricPearson");

    // Method Switcher Controller
    function setMethod(method) {
        currentMethod = method;
        algoButtons.forEach(btn => {
            if (btn.dataset.method === method) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        const upper = method.toUpperCase();
        if (videoMethodBadge) {
            videoMethodBadge.textContent = (method === "all") ? "ALL 3" : upper;
            videoMethodBadge.className = `algo-tag ${method}`;
        }
        if (oscMethodBadge) {
            oscMethodBadge.textContent = (method === "all") ? "POS (Main)" : upper;
            oscMethodBadge.className = `algo-tag ${method}`;
        }

        if (method === "all") {
            if (singleVitalsGrid) singleVitalsGrid.style.display = "none";
            if (compareVitalsGrid) compareVitalsGrid.style.display = "grid";
            if (singleMetricsContainer) singleMetricsContainer.style.display = "none";
            if (compareMetricsContainer) compareMetricsContainer.style.display = "block";

            // Show all 3 rPPG legend badges
            if (posLegendBadge) posLegendBadge.style.display = "flex";
            if (chromLegendBadge) chromLegendBadge.style.display = "flex";
            if (omitLegendBadge) omitLegendBadge.style.display = "flex";

            // Unhide all 3 rPPG curves
            bpmChart.data.datasets[1].hidden = false;
            bpmChart.data.datasets[2].hidden = false;
            bpmChart.data.datasets[3].hidden = false;
        } else {
            if (singleVitalsGrid) singleVitalsGrid.style.display = "grid";
            if (compareVitalsGrid) compareVitalsGrid.style.display = "none";
            if (singleMetricsContainer) singleMetricsContainer.style.display = "block";
            if (compareMetricsContainer) compareMetricsContainer.style.display = "none";

            if (vitalMethodLabel) {
                vitalMethodLabel.textContent = `Heart Rate (rPPG ${upper})`;
            }
            if (metricsMethodLabel) {
                metricsMethodLabel.textContent = upper;
            }

            // Show only the selected algorithm's legend badge
            if (posLegendBadge) posLegendBadge.style.display = (method === "pos") ? "flex" : "none";
            if (chromLegendBadge) chromLegendBadge.style.display = (method === "chrom") ? "flex" : "none";
            if (omitLegendBadge) omitLegendBadge.style.display = (method === "omit" || method === "green") ? "flex" : "none";

            // Only show the selected curve in the graph
            bpmChart.data.datasets[1].hidden = (method !== "pos");
            bpmChart.data.datasets[2].hidden = (method !== "chrom");
            bpmChart.data.datasets[3].hidden = (method !== "omit" && method !== "green");
        }

        // Keep Ground Truth visible if present and in upload mode
        if (groundTruthData && currentMode === "upload") {
            bpmChart.data.datasets[0].hidden = false;
            if (gtLegendBadge) gtLegendBadge.style.display = "flex";
        }

        bpmChart.update();

        // Inform backend stream manager of dynamic switch
        fetch("/api/control/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "set_method", method: method })
        }).catch(() => {});
    }

    algoButtons.forEach(btn => {
        btn.addEventListener("click", () => setMethod(btn.dataset.method));
    });

    // Interactive Legend Badges to toggle individual curves on click
    function setupLegendToggle(badgeEl, datasetIdx) {
        if (!badgeEl) return;
        badgeEl.addEventListener("click", () => {
            const isHidden = !bpmChart.data.datasets[datasetIdx].hidden;
            bpmChart.data.datasets[datasetIdx].hidden = isHidden;
            if (isHidden) {
                badgeEl.classList.add("strike");
            } else {
                badgeEl.classList.remove("strike");
            }
            bpmChart.update();
        });
    }
    setupLegendToggle(gtLegendBadge, 0);
    setupLegendToggle(posLegendBadge, 1);
    setupLegendToggle(chromLegendBadge, 2);
    setupLegendToggle(omitLegendBadge, 3);

    // Controls
    const btnStart = document.getElementById("btnStart");
    const btnPause = document.getElementById("btnPause");
    const btnStop = document.getElementById("btnStop");
    const btnFastForward = document.getElementById("btnFastForward");
    let isFastForward = false;

    function updateFastForwardButtonVisibility() {
        if (!btnFastForward) return;
        if (currentMode === "upload" && isRunning) {
            btnFastForward.style.display = "inline-flex";
        } else {
            btnFastForward.style.display = "none";
        }
    }

    function updateFastForwardUI(active) {
        isFastForward = !!active;
        if (!btnFastForward) return;
        if (isFastForward) {
            btnFastForward.classList.add("active");
            btnFastForward.innerHTML = '<span>⏩</span> Fast Forward: ON';
        } else {
            btnFastForward.classList.remove("active");
            btnFastForward.innerHTML = '<span>⏩</span> Fast Forward';
        }
    }

    if (btnFastForward) {
        btnFastForward.addEventListener("click", async () => {
            try {
                const resp = await fetch("/api/control/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "toggle_fast_forward" })
                });
                const data = await resp.json();
                if (resp.ok && data.fast_forward !== undefined) {
                    updateFastForwardUI(data.fast_forward);
                }
            } catch (err) {
                console.error("Fast forward toggle error:", err);
            }
        });
    }

    // Mode Switchers
    const tabWebcam = document.getElementById("tabWebcam");
    const tabUpload = document.getElementById("tabUpload");
    const uploadPanel = document.getElementById("uploadPanel");
    const webcamOptions = document.getElementById("webcamOptions");
    const chkCameraSettings = document.getElementById("chkCameraSettings");
    const btnCameraSettings = document.getElementById("btnCameraSettings");
    const btnManualCameraSettings = document.getElementById("btnManualCameraSettings");

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

    // Helper to trigger camera settings
    async function triggerCameraSettings() {
        try {
            const resp = await fetch("/api/control/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "open_camera_settings" })
            });
            const data = await resp.json();
            if (!data.success && !isRunning && currentMode === "webcam") {
                // If stream is not running yet, start webcam which automatically opens settings
                btnStart.click();
            }
        } catch (err) {
            console.error("Camera settings error:", err);
        }
    }

    if (btnCameraSettings) {
        btnCameraSettings.addEventListener("click", triggerCameraSettings);
    }
    if (btnManualCameraSettings) {
        btnManualCameraSettings.addEventListener("click", triggerCameraSettings);
    }

    // Mode Switching
    tabWebcam.addEventListener("click", () => {
        if (isRunning) stopStream();
        currentMode = "webcam";
        tabWebcam.classList.add("active");
        tabUpload.classList.remove("active");
        uploadPanel.classList.remove("active");
        if (webcamOptions) webcamOptions.style.display = "flex";
        updateFastForwardButtonVisibility();
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
        if (webcamOptions) webcamOptions.style.display = "none";
        updateFastForwardButtonVisibility();
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
        bpmChart.data.datasets[2].data = [];
        bpmChart.data.datasets[3].data = [];
        bpmChart.options.scales.x.min = 0;
        bpmChart.options.scales.x.max = undefined;
        bpmChart.options.scales.y.suggestedMin = 60;
        bpmChart.options.scales.y.suggestedMax = 100;
        bpmChart.update();
        pulseBuffer = [];
        rawPulseBuffer = [];
        metricGtBpm.textContent = "--";
        metricError.textContent = "--";
        metricMae.textContent = "--";
        metricRmse.textContent = "--";
        metricPearson.textContent = "--";
        bpmValue.textContent = "--";
        if (posBpmValue) posBpmValue.textContent = "--";
        if (chromBpmValue) chromBpmValue.textContent = "--";
        if (omitBpmValue) omitBpmValue.textContent = "--";
        if (compareRefGtBpm) compareRefGtBpm.textContent = "--";
        resetCompareTable();
        videoProgressBar.style.width = "0%";
    }

    function resetCompareTable() {
        const rows = [rowMetricPos, rowMetricChrom, rowMetricOmit];
        rows.forEach(r => { if (r) r.classList.remove("highlight-best"); });
        const badges = [tblPosBadge, tblChromBadge, tblOmitBadge];
        badges.forEach(b => { if (b) b.innerHTML = '<span class="badge-status">Evaluating</span>'; });
        const cells = [
            tblPosBpm, tblPosError, tblPosMae, tblPosRmse, tblPosPearson,
            tblChromBpm, tblChromError, tblChromMae, tblChromRmse, tblChromPearson,
            tblOmitBpm, tblOmitError, tblOmitMae, tblOmitRmse, tblOmitPearson
        ];
        cells.forEach(c => { if (c) c.textContent = "--"; });
    }

    // Stream Controls
    btnStart.addEventListener("click", async () => {
        if (currentMode === "webcam") {
            const openSettings = chkCameraSettings ? chkCameraSettings.checked : true;
            statusText.textContent = openSettings ? "Starting camera & settings dialog..." : "Starting camera...";
            const resp = await fetch("/api/control/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    action: "start_webcam",
                    camera_index: 0,
                    method: currentMethod,
                    camera_settings: openSettings
                })
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
                    alert("Please select a video file to upload, or choose a local dataset from /data.");
                    return;
                }
            }

            statusText.textContent = "Starting video playback...";
            const payload = {
                action: "start_video",
                video_path: uploadedVideoPath,
                gt_times: groundTruthData ? groundTruthData.raw_times : [],
                gt_hr: groundTruthData ? groundTruthData.raw_hr : [],
                method: currentMethod
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
        if (currentMode === "webcam") {
            if (btnCameraSettings) btnCameraSettings.style.display = "inline-flex";
        } else {
            if (btnCameraSettings) btnCameraSettings.style.display = "none";
        }
        updateFastForwardButtonVisibility();
        updateFastForwardUI(false);

        // Show stream
        videoPlaceholder.style.display = "none";
        videoFeed.style.display = "block";
        videoFeed.src = "/api/stream/?t=" + new Date().getTime();

        statusDot.className = "status-dot";

        // Clear prediction datasets, keep pre-loaded ground truth if present
        bpmChart.data.datasets[1].data = [];
        bpmChart.data.datasets[2].data = [];
        bpmChart.data.datasets[3].data = [];
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
        if (btnCameraSettings) btnCameraSettings.style.display = "none";
        updateFastForwardButtonVisibility();
        updateFastForwardUI(false);

        videoFeed.src = "";
        videoFeed.style.display = "none";
        videoPlaceholder.style.display = "block";
        statusDot.className = "status-dot idle";
        statusText.textContent = "Stopped";
    }

    function updateCompareTableRow(m, bpm, elBpm, elErr, elMae, elRmse, elPearson) {
        if (elBpm) elBpm.textContent = bpm ? bpm.toFixed(1) : "--";
        if (!m) return;
        if (elErr) elErr.textContent = m.current_error !== null ? `±${m.current_error}` : "--";
        if (elMae) elMae.textContent = m.mae !== null ? m.mae : "--";
        if (elRmse) elRmse.textContent = m.rmse !== null ? m.rmse : "--";
        if (elPearson) elPearson.textContent = m.pearson_r !== null ? m.pearson_r : "--";
    }

    function rankMethods(allM) {
        const list = [];
        ["pos", "chrom", "omit"].forEach(k => {
            const m = allM[k] || (k === "omit" ? allM["green"] : null);
            if (m && m.mae !== null && m.mae > 0) {
                list.push({ key: k, mae: m.mae });
            }
        });

        // Reset highlight and badge texts
        [rowMetricPos, rowMetricChrom, rowMetricOmit].forEach(r => { if (r) r.classList.remove("highlight-best"); });
        [tblPosBadge, tblChromBadge, tblOmitBadge].forEach(b => {
            if (b) b.innerHTML = '<span class="badge-status">Evaluating</span>';
        });

        if (list.length === 0) return;

        list.sort((a, b) => a.mae - b.mae);

        const badgeEls = { pos: tblPosBadge, chrom: tblChromBadge, omit: tblOmitBadge };
        const rowEls = { pos: rowMetricPos, chrom: rowMetricChrom, omit: rowMetricOmit };

        list.forEach((item, idx) => {
            const b = badgeEls[item.key];
            const r = rowEls[item.key];
            if (idx === 0) {
                if (b) b.innerHTML = '<span class="badge-status best">🏆 1st (Best)</span>';
                if (r) r.classList.add("highlight-best");
            } else if (idx === 1) {
                if (b) b.innerHTML = '<span class="badge-status rank2">🥈 2nd</span>';
            } else if (idx === 2) {
                if (b) b.innerHTML = '<span class="badge-status rank3">🥉 3rd</span>';
            }
        });
    }

    // Telemetry consumer
    let lastTimePoint = -1;
    async function fetchTelemetry() {
        try {
            const resp = await fetch("/api/metrics/");
            const data = await resp.json();

            // Status updates
            statusText.textContent = data.status_text || "Processing...";
            if (data.fast_forward !== undefined && data.fast_forward !== isFastForward) {
                updateFastForwardUI(data.fast_forward);
            }
            if (data.face_detected) {
                statusDot.className = "status-dot";
            } else {
                statusDot.className = "status-dot warning";
            }

            const methodBpms = data.method_bpms || {};
            const posBpm = methodBpms.pos;
            const chromBpm = methodBpms.chrom;
            const omitBpm = methodBpms.omit !== undefined ? methodBpms.omit : methodBpms.green;

            // Single Vitals Readout
            if (data.current_bpm) {
                bpmValue.textContent = data.current_bpm.toFixed(1);
                const beatDuration = (60.0 / data.current_bpm).toFixed(2);
                heartIcon.style.animationDuration = `${beatDuration}s`;
            } else {
                bpmValue.textContent = "--";
            }

            // Compare All 3 Vitals Readouts
            if (posBpmValue) posBpmValue.textContent = posBpm ? posBpm.toFixed(1) : "--";
            if (chromBpmValue) chromBpmValue.textContent = chromBpm ? chromBpm.toFixed(1) : "--";
            if (omitBpmValue) omitBpmValue.textContent = omitBpm ? omitBpm.toFixed(1) : "--";
            if (compareSqiValue) compareSqiValue.textContent = `${data.sqi}%`;

            // SQI
            sqiValue.textContent = `${data.sqi}%`;

            // Progress
            if (data.progress !== undefined) {
                videoProgressBar.style.width = `${data.progress}%`;
            }

            // Pulse wave buffer for oscilloscope
            if (data.method_pulses) {
                const activeKey = (currentMethod in data.method_pulses) ? currentMethod : "pos";
                if (data.method_pulses[activeKey] && data.method_pulses[activeKey].length > 0) {
                    pulseBuffer = data.method_pulses[activeKey];
                }
            } else if (data.pulse_history && data.pulse_history.length > 0) {
                pulseBuffer = data.pulse_history;
            }

            if (data.method_raw_pulses) {
                const activeRawKey = (currentMethod in data.method_raw_pulses) ? currentMethod : "pos";
                if (data.method_raw_pulses[activeRawKey] && data.method_raw_pulses[activeRawKey].length > 0) {
                    rawPulseBuffer = data.method_raw_pulses[activeRawKey];
                }
            } else if (data.raw_pulse_history && data.raw_pulse_history.length > 0) {
                rawPulseBuffer = data.raw_pulse_history;
            }

            // Append live predicted points to Chart.js
            if (data.timestamp > lastTimePoint + 0.3) {
                lastTimePoint = data.timestamp;
                let chartUpdated = false;

                if (currentMethod === "all") {
                    if (posBpm) {
                        bpmChart.data.datasets[1].data.push({ x: data.timestamp, y: posBpm });
                        chartUpdated = true;
                    }
                    if (chromBpm) {
                        bpmChart.data.datasets[2].data.push({ x: data.timestamp, y: chromBpm });
                        chartUpdated = true;
                    }
                    if (omitBpm) {
                        bpmChart.data.datasets[3].data.push({ x: data.timestamp, y: omitBpm });
                        chartUpdated = true;
                    }
                } else {
                    const dsIdx = (currentMethod === "chrom") ? 2 : ((currentMethod === "omit" || currentMethod === "green") ? 3 : 1);
                    if (data.current_bpm) {
                        bpmChart.data.datasets[dsIdx].data.push({ x: data.timestamp, y: data.current_bpm });
                        chartUpdated = true;
                    }
                }

                if (chartUpdated) {
                    // Auto slide window in webcam mode
                    if (currentMode === "webcam" && data.timestamp > 30) {
                        bpmChart.options.scales.x.min = data.timestamp - 30;
                        bpmChart.options.scales.x.max = data.timestamp;
                    }
                    bpmChart.update("none");
                }
            }

            // Performance Benchmark Metrics (in Upload Mode with GT)
            if (data.metrics) {
                const m = data.metrics;
                metricGtBpm.textContent = m.latest_gt !== null ? `${m.latest_gt} BPM` : "--";
                metricError.textContent = m.current_error !== null ? `±${m.current_error} BPM` : "--";
                metricMae.textContent = m.mae !== null ? `${m.mae} BPM` : "--";
                metricRmse.textContent = m.rmse !== null ? `${m.rmse} BPM` : "--";
                metricPearson.textContent = m.pearson_r !== null ? m.pearson_r : "--";
                if (compareRefGtBpm) {
                    compareRefGtBpm.textContent = m.latest_gt !== null ? `${m.latest_gt} BPM` : "--";
                }
            }

            // Comparative Leaderboard Table
            if (data.all_metrics) {
                const allM = data.all_metrics;
                const omitMetrics = allM.omit || allM.green;
                updateCompareTableRow(allM.pos, posBpm, tblPosBpm, tblPosError, tblPosMae, tblPosRmse, tblPosPearson);
                updateCompareTableRow(allM.chrom, chromBpm, tblChromBpm, tblChromError, tblChromMae, tblChromRmse, tblChromPearson);
                updateCompareTableRow(omitMetrics, omitBpm, tblOmitBpm, tblOmitError, tblOmitMae, tblOmitRmse, tblOmitPearson);
                rankMethods(allM);
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

        if (pulseBuffer.length < 2 && rawPulseBuffer.length < 2) {
            ctx.fillStyle = "rgba(156, 163, 175, 0.45)";
            ctx.font = "12px Inter, system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(isRunning ? "Extracting Butterworth BVP pulse..." : "BVP Oscilloscope Idle", w / 2, h / 2 - 8);
            return;
        }

        const centerY = h / 2;
        const scaleY = h * 0.38;

        // Draw RAW (pre-filter) pulse waveform first (behind filtered)
        if (rawPulseBuffer.length >= 2) {
            ctx.strokeStyle = "rgba(156, 163, 175, 0.4)";
            ctx.lineWidth = 1.4;
            ctx.shadowColor = "transparent";
            ctx.shadowBlur = 0;
            ctx.beginPath();

            const nRaw = rawPulseBuffer.length;
            const dxRaw = w / (nRaw - 1);

            for (let i = 0; i < nRaw; i++) {
                const x = i * dxRaw;
                const y = centerY - (rawPulseBuffer[i] * scaleY);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }

        // Draw FILTERED pulse waveform with dynamic algorithm color & glow (on top)
        const isOmit = (currentMethod === "omit" || currentMethod === "green");
        const strokeColor = (currentMethod === "pos" || currentMethod === "all") ? "#eb7a02" : (isOmit ? "#c084fc" : "#10b981");
        const glowColor = (currentMethod === "pos" || currentMethod === "all") ? "rgba(235, 122, 2, 0.6)" : (isOmit ? "rgba(192, 132, 252, 0.6)" : "rgba(16, 185, 129, 0.6)");
        const gradColor = (currentMethod === "pos" || currentMethod === "all") ? "rgba(235, 122, 2, 0.12)" : (isOmit ? "rgba(168, 85, 247, 0.12)" : "rgba(16, 185, 129, 0.12)");

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.2;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 8;
        ctx.beginPath();

        const n = pulseBuffer.length;
        const dx = w / (n - 1);

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
        grad.addColorStop(0, gradColor);
        grad.addColorStop(1, "rgba(0, 0, 0, 0.0)");
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
