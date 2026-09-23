/**
 * PulseVision rPPG Studio - Main Application Logic & WebSocket Telemetry Controller.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const connectionPill = document.getElementById('connection-pill');
  const connectionText = document.getElementById('connection-text');
  const activeSourceName = document.getElementById('active-source-name');
  const headerFps = document.getElementById('header-fps');
  const sessionTime = document.getElementById('session-time');

  // Video & Controls
  const videoStream = document.getElementById('video-stream');
  const standbyOverlay = document.getElementById('standby-overlay');
  const btnStandbyBrowse = document.getElementById('btn-standby-browse');
  const btnStandbyWebcam = document.getElementById('btn-standby-webcam');
  const toggleOverlay = document.getElementById('toggle-overlay');
  const faceBadge = document.getElementById('face-detection-badge');
  const hudStatus = document.getElementById('hud-status');
  const btnPlayPause = document.getElementById('btn-play-pause');
  const playPauseIcon = document.getElementById('play-pause-icon');
  const playPauseText = document.getElementById('play-pause-text');
  const btnReset = document.getElementById('btn-reset');
  const btnRecord = document.getElementById('btn-record');
  const recordBtnText = document.getElementById('record-btn-text');
  const btnExport = document.getElementById('btn-export');
  const sourceSelect = document.getElementById('source-select');
  const btnBrowseData = document.getElementById('btn-browse-data');
  const btnRefreshSources = document.getElementById('btn-refresh-sources');
  const btnUploadModal = document.getElementById('btn-upload-modal');

  // Data Browser Modal Elements
  const dataBrowserModal = document.getElementById('data-browser-modal');
  const btnCloseDataBrowser = document.getElementById('btn-close-data-browser');
  const dataSearchInput = document.getElementById('data-search-input');
  const btnRescanData = document.getElementById('btn-rescan-data');
  const dataFolderPills = document.getElementById('data-folder-pills');
  const dataFilesList = document.getElementById('data-files-list');
  const manualPathInput = document.getElementById('manual-path-input');
  const btnLoadManualPath = document.getElementById('btn-load-manual-path');
  const manualPathStatus = document.getElementById('manual-path-status');

  // Vitals Readouts
  const bpmValue = document.getElementById('bpm-value');
  const bpmStateBadge = document.getElementById('bpm-state-badge');
  const bpmZone = document.getElementById('bpm-zone');
  const bpmMarker = document.getElementById('bpm-marker');
  const heartIconContainer = document.getElementById('heart-icon-container');

  // Quality & Patches
  const snrValue = document.getElementById('snr-value');
  const snrRating = document.getElementById('snr-rating');
  const snrBarFill = document.getElementById('snr-bar-fill');
  const activePatchesCount = document.getElementById('active-patches-count');
  const patchesGrid = document.getElementById('patches-grid');

  // Session Stats
  const statMinBpm = document.getElementById('stat-min-bpm');
  const statAvgBpm = document.getElementById('stat-avg-bpm');
  const statMaxBpm = document.getElementById('stat-max-bpm');
  const recordedCount = document.getElementById('recorded-count');

  // Oscilloscope & Tabs
  const liveAmplitude = document.getElementById('live-amplitude');
  const pulseCanvas = document.getElementById('pulse-canvas');
  const trendCanvas = document.getElementById('trend-canvas');
  const tabPulse = document.getElementById('tab-pulse');
  const tabTrend = document.getElementById('tab-trend');

  // Modals
  const settingsModal = document.getElementById('settings-modal');
  const btnOpenSettings = document.getElementById('btn-open-settings');
  const btnCloseSettings = document.getElementById('btn-close-settings');
  const btnSaveSettings = document.getElementById('btn-save-settings');
  const settingRoiAlpha = document.getElementById('setting-roi-alpha');
  const valRoiAlpha = document.getElementById('val-roi-alpha');
  const settingQuality = document.getElementById('setting-quality');
  const valQuality = document.getElementById('val-quality');
  const settingLoop = document.getElementById('setting-loop');
  const settingHud = document.getElementById('setting-hud');

  const uploadModal = document.getElementById('upload-modal');
  const btnCloseUpload = document.getElementById('btn-close-upload');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const uploadProgressBox = document.getElementById('upload-progress-box');
  const uploadProgressFill = document.getElementById('upload-progress-fill');
  const uploadStatusText = document.getElementById('upload-status-text');

  // Initialize Canvas Visualizers
  const pulseRenderer = new PulseWaveformRenderer('pulse-canvas');
  const trendRenderer = new BpmTrendRenderer('trend-canvas');

  // Spatial Patch Labels
  const PATCH_LABELS = [
    { id: 0, name: 'L-Lo-FH' },
    { id: 1, name: 'C-Lo-FH' },
    { id: 2, name: 'R-Lo-FH' },
    { id: 3, name: 'L-Up-FH' },
    { id: 4, name: 'C-Up-FH' },
    { id: 5, name: 'R-Up-FH' },
    { id: 6, name: 'L-Up-CK' },
    { id: 7, name: 'L-Lo-CK' },
    { id: 8, name: 'L-Out-CK' },
    { id: 9, name: 'R-Up-CK' },
    { id: 10, name: 'R-Lo-CK' },
    { id: 11, name: 'R-Out-CK' },
  ];

  // Initialize 12-Patch Matrix DOM
  function initPatchesGrid() {
    patchesGrid.innerHTML = '';
    PATCH_LABELS.forEach((patch) => {
      const cell = document.createElement('div');
      cell.className = 'patch-cell';
      cell.id = `patch-${patch.id}`;
      cell.innerHTML = `
        <span class="patch-id">P${patch.id}</span>
        <span class="patch-name">${patch.name}</span>
      `;
      patchesGrid.appendChild(cell);
    });
  }
  initPatchesGrid();

  // State
  let ws = null;
  let isRecording = false;
  let isPaused = false;
  let lastBpm = 0;

  // Format Elapsed Seconds as MM:SS
  function formatSeconds(sec) {
    const m = Math.floor(sec / 60).toString().padStart(2, '0');
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  // Determine Physiological Heart Rate Zone
  function getBpmZone(bpm) {
    if (bpm <= 0) return { label: 'Awaiting Signal', class: '' };
    if (bpm < 60) return { label: 'Bradycardia / Athletic (<60)', class: 'text-cyan' };
    if (bpm <= 100) return { label: 'Normal Resting (60-100)', class: 'text-emerald' };
    return { label: 'Elevated / Tachycardia (>100)', class: 'text-pulse' };
  }

  // Update Range Bar Marker Position (40 - 150 BPM)
  function updateBpmMarker(bpm) {
    if (bpm <= 0) {
      bpmMarker.style.left = '50%';
      return;
    }
    const clamped = Math.max(40, Math.min(150, bpm));
    const percent = ((clamped - 40) / (150 - 40)) * 100;
    bpmMarker.style.left = `${percent}%`;
  }

  // Update SNR Gauge
  function updateSnr(snr) {
    snrValue.textContent = snr.toFixed(2);
    // Linear power ratio: typically 1.0 to 12.0+
    const clamped = Math.max(0, Math.min(12, snr));
    const percent = (clamped / 12) * 100;
    snrBarFill.style.width = `${percent}%`;

    if (snr < 1.5) {
      snrRating.textContent = 'NOISY';
      snrRating.style.color = '#ff2a5f';
      snrRating.style.backgroundColor = 'rgba(255, 42, 95, 0.15)';
    } else if (snr < 3.5) {
      snrRating.textContent = 'FAIR';
      snrRating.style.color = '#ffb300';
      snrRating.style.backgroundColor = 'rgba(255, 179, 0, 0.15)';
    } else {
      snrRating.textContent = 'OPTIMAL';
      snrRating.style.color = '#00e676';
      snrRating.style.backgroundColor = 'rgba(0, 230, 118, 0.15)';
    }
  }

  // Update Active 12-Patches Grid
  function updatePatches(activeIndices) {
    const activeSet = new Set(activeIndices || []);
    activePatchesCount.textContent = `${activeSet.size}/6 Active`;

    PATCH_LABELS.forEach((patch) => {
      const el = document.getElementById(`patch-${patch.id}`);
      if (el) {
        if (activeSet.has(patch.id)) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      }
    });
  }

  // Synchronize CSS Heartbeat Pulse Duration to Measured BPM
  function setHeartPulseDuration(bpm) {
    if (bpm && bpm > 40 && bpm < 200) {
      const durationSec = (60.0 / bpm).toFixed(2);
      document.documentElement.style.setProperty('--heart-pulse-duration', `${durationSec}s`);
    } else {
      document.documentElement.style.setProperty('--heart-pulse-duration', '0.85s');
    }
  }

  // ==========================================================================
  // WebSocket Telemetry Connection
  // ==========================================================================
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      connectionPill.classList.remove('disconnected');
      connectionText.textContent = 'LIVE STREAMING';
      hudStatus.textContent = 'Engine Connected & Active';
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleTelemetry(data);
      } catch (err) {
        console.error('Failed to parse telemetry message:', err);
      }
    };

    ws.onclose = () => {
      connectionPill.classList.add('disconnected');
      connectionText.textContent = 'RECONNECTING...';
      hudStatus.textContent = 'Reconnecting to stream...';
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.warn('WebSocket encountered error:', err);
    };
  }

  // Ingest incoming telemetry payload from Python backend
  function handleTelemetry(data) {
    const isStandby = !data.is_running || !data.source;

    if (isStandby) {
      if (standbyOverlay) standbyOverlay.style.display = 'flex';
      connectionPill.className = 'stream-status-pill standby';
      connectionText.textContent = 'STANDBY (AWAITING SELECTION)';
      activeSourceName.textContent = 'No source selected';
      faceBadge.textContent = 'STANDBY';
      faceBadge.className = 'badge';
      bpmValue.textContent = '--';
      bpmStateBadge.textContent = 'STANDBY';
      bpmStateBadge.className = 'confidence-pill';
      bpmZone.textContent = 'Select a video from data/ or choose a webcam to begin';
      headerFps.textContent = '--';
      return;
    }

    if (standbyOverlay) standbyOverlay.style.display = 'none';
    connectionPill.className = 'stream-status-pill';
    connectionText.textContent = 'LIVE STREAMING';

    // 1. Header & System Stats
    headerFps.textContent = data.fps ? data.fps.toFixed(1) : '--';
    sessionTime.textContent = formatSeconds(data.session_duration_sec || 0);

    // 2. Face Detection Status
    if (data.face_detected) {
      faceBadge.textContent = 'FACE TRACKED';
      faceBadge.className = 'badge detected';
    } else {
      faceBadge.textContent = 'NO FACE DETECTED';
      faceBadge.className = 'badge lost';
    }

    // 3. Heart Rate (BPM)
    if (data.is_warming_up || !data.bpm || data.bpm <= 0) {
      bpmValue.textContent = '--';
      bpmStateBadge.textContent = 'CALIBRATING';
      bpmStateBadge.className = 'confidence-pill';
      bpmZone.textContent = data.face_detected ? 'Accumulating POS signal...' : 'Position face in frame';
      updateBpmMarker(0);
      setHeartPulseDuration(0);
    } else {
      bpmValue.textContent = data.bpm.toFixed(1);
      bpmStateBadge.textContent = 'STABLE READING';
      bpmStateBadge.className = 'confidence-pill stable';
      const zone = getBpmZone(data.bpm);
      bpmZone.textContent = zone.label;
      updateBpmMarker(data.bpm);
      setHeartPulseDuration(data.bpm);

      if (Math.abs(data.bpm - lastBpm) > 0.4) {
        trendRenderer.addPoint(data.bpm);
        lastBpm = data.bpm;
      }
    }

    // 4. Signal Quality & Spatial Patches
    updateSnr(data.snr || 0);
    updatePatches(data.active_patches || []);

    // 5. Pulse Waveform Stream
    if (data.pulse_samples && data.pulse_samples.length > 0) {
      pulseRenderer.setSamples(data.pulse_samples);
    }
    if (data.latest_pulse !== undefined) {
      const sign = data.latest_pulse >= 0 ? '+' : '';
      liveAmplitude.textContent = `Amp: ${sign}${data.latest_pulse.toFixed(4)}`;
    }

    // 6. Session Aggregate Stats
    statMinBpm.textContent = data.min_bpm > 0 ? data.min_bpm.toFixed(1) : '--';
    statAvgBpm.textContent = data.avg_bpm > 0 ? data.avg_bpm.toFixed(1) : '--';
    statMaxBpm.textContent = data.max_bpm > 0 ? data.max_bpm.toFixed(1) : '--';
    recordedCount.textContent = `${data.recorded_samples || 0} samples`;

    // 7. Playback & Recording state sync
    isPaused = !!data.is_paused;
    playPauseIcon.textContent = isPaused ? '▶' : '⏸';
    playPauseText.textContent = isPaused ? 'Resume' : 'Pause';

    isRecording = !!data.is_recording;
    if (isRecording) {
      btnRecord.classList.add('recording');
      recordBtnText.textContent = 'Stop Record';
    } else {
      btnRecord.classList.remove('recording');
      recordBtnText.textContent = 'Record';
    }
  }

  // ==========================================================================
  // Sources & Data Folder Management
  // ==========================================================================
  let dataFolderFiles = [];
  let currentActiveFolder = 'all';
  let currentSearchQuery = '';
  let currentSourcePath = '';

  async function switchSource(sourcePath, displayName) {
    try {
      await fetch('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourcePath }),
      });
      currentSourcePath = sourcePath;
      pulseRenderer.clear();
      trendRenderer.clear();
      if (displayName) {
        activeSourceName.textContent = displayName;
      }
      reloadVideoStream();
      await loadSources();
      renderDataFilesList();
    } catch (err) {
      console.error('Error switching source:', err);
    }
  }

  async function loadSources() {
    try {
      const res = await fetch('/api/sources');
      const data = await res.json();
      currentSourcePath = data.current_source || '';

      sourceSelect.innerHTML = '';

      // Default prompt option
      const defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.textContent = '-- Select Video from data/ or Webcam --';
      defaultOpt.disabled = true;
      if (!currentSourcePath) {
        defaultOpt.selected = true;
      }
      sourceSelect.appendChild(defaultOpt);

      // 1. Cameras group
      const camGroup = document.createElement('optgroup');
      camGroup.label = 'Webcam Devices';
      data.cameras.forEach((cam) => {
        const opt = document.createElement('option');
        opt.value = cam.id;
        opt.textContent = `📷 ${cam.name}`;
        if (String(cam.id) === String(data.current_source)) {
          opt.selected = true;
          activeSourceName.textContent = cam.name;
        }
        camGroup.appendChild(opt);
      });
      sourceSelect.appendChild(camGroup);

      // Separate regular data/ videos from uploads
      const dataVideos = (data.files || []).filter((f) => !f.is_upload);
      const uploadVideos = (data.files || []).filter((f) => f.is_upload);

      // 2. Videos in data/ folder (all subfolders)
      if (dataVideos.length > 0) {
        const dataGroup = document.createElement('optgroup');
        dataGroup.label = `Videos in data/ folder (${dataVideos.length})`;
        dataVideos.forEach((file) => {
          const opt = document.createElement('option');
          opt.value = file.path;
          const subInfo = file.subfolder !== 'Root' ? `[${file.subfolder}] ` : '';
          opt.textContent = `📁 ${subInfo}${file.name} (${file.size_mb} MB)`;
          if (file.path === data.current_source || file.data_path === data.current_source) {
            opt.selected = true;
            activeSourceName.textContent = `${subInfo}${file.name}`;
          }
          dataGroup.appendChild(opt);
        });
        sourceSelect.appendChild(dataGroup);
      }

      // 3. Uploaded Videos
      if (uploadVideos.length > 0) {
        const uploadGroup = document.createElement('optgroup');
        uploadGroup.label = `Uploaded Videos (${uploadVideos.length})`;
        uploadVideos.forEach((file) => {
          const opt = document.createElement('option');
          opt.value = file.path;
          opt.textContent = `📹 ${file.name} (${file.size_mb} MB)`;
          if (file.path === data.current_source) {
            opt.selected = true;
            activeSourceName.textContent = file.name;
          }
          uploadGroup.appendChild(opt);
        });
        sourceSelect.appendChild(uploadGroup);
      }
    } catch (err) {
      console.error('Failed to load video sources:', err);
    }
  }

  sourceSelect.addEventListener('change', async (e) => {
    const newSource = e.target.value;
    const selectedText = e.target.options[e.target.selectedIndex].text;
    await switchSource(newSource, selectedText);
  });

  // Data Folder Explorer Modal
  async function openDataBrowser() {
    dataBrowserModal.style.display = 'flex';
    manualPathStatus.textContent = '';
    manualPathStatus.className = 'manual-path-status font-mono';
    await fetchAndRenderDataFiles();
  }

  async function fetchAndRenderDataFiles() {
    dataFilesList.innerHTML = '<div class="loading-placeholder">Scanning data/ directory for video files...</div>';
    try {
      const res = await fetch('/api/data/browse');
      const data = await res.json();
      dataFolderFiles = data.files || [];
      currentSourcePath = data.current_source || '';

      // Populate subfolder filter pills
      dataFolderPills.innerHTML = '';
      const allPill = document.createElement('span');
      allPill.className = `folder-pill ${currentActiveFolder === 'all' ? 'active' : ''}`;
      allPill.dataset.folder = 'all';
      allPill.textContent = `All Videos (${dataFolderFiles.length})`;
      allPill.addEventListener('click', () => {
        currentActiveFolder = 'all';
        updateFolderPillsActive();
        renderDataFilesList();
      });
      dataFolderPills.appendChild(allPill);

      (data.subfolders || []).forEach((folder) => {
        const pill = document.createElement('span');
        pill.className = `folder-pill ${currentActiveFolder === folder ? 'active' : ''}`;
        pill.dataset.folder = folder;
        const count = dataFolderFiles.filter((f) => f.subfolder === folder).length;
        pill.textContent = `${folder} (${count})`;
        pill.addEventListener('click', () => {
          currentActiveFolder = folder;
          updateFolderPillsActive();
          renderDataFilesList();
        });
        dataFolderPills.appendChild(pill);
      });

      renderDataFilesList();
    } catch (err) {
      dataFilesList.innerHTML = `<div class="empty-placeholder text-pulse">Error scanning data folder: ${err.message}</div>`;
    }
  }

  function updateFolderPillsActive() {
    Array.from(dataFolderPills.children).forEach((pill) => {
      if (pill.dataset.folder === currentActiveFolder) {
        pill.classList.add('active');
      } else {
        pill.classList.remove('active');
      }
    });
  }

  function renderDataFilesList() {
    const q = currentSearchQuery.toLowerCase().trim();
    const filtered = dataFolderFiles.filter((file) => {
      const matchesFolder = currentActiveFolder === 'all' || file.subfolder === currentActiveFolder;
      const matchesSearch =
        !q ||
        file.name.toLowerCase().includes(q) ||
        file.path.toLowerCase().includes(q) ||
        file.subfolder.toLowerCase().includes(q);
      return matchesFolder && matchesSearch;
    });

    if (filtered.length === 0) {
      dataFilesList.innerHTML = `
        <div class="empty-placeholder">
          <p>No video files found matching current filters.</p>
          <p class="text-muted" style="font-size: 0.75rem; margin-top: 6px;">
            Copy videos into <code>data/</code> or any subfolder (e.g. <code>data/subject10/vid.avi</code>) and click Rescan.
          </p>
        </div>
      `;
      return;
    }

    dataFilesList.innerHTML = '';
    filtered.forEach((file) => {
      const isActive = file.path === currentSourcePath || file.data_path === currentSourcePath;
      const card = document.createElement('div');
      card.className = `video-file-card ${isActive ? 'active' : ''}`;

      const icon = file.is_upload ? '📹' : '📁';
      const subTagClass = file.is_upload ? 'file-subfolder-tag upload' : 'file-subfolder-tag';

      card.innerHTML = `
        <div class="file-info-group">
          <span class="file-icon">${icon}</span>
          <div class="file-meta-col">
            <div class="file-name-row">
              <span class="file-name" title="${file.path}">${file.name}</span>
              <span class="${subTagClass}">${file.subfolder}</span>
              ${isActive ? '<span class="active-tag">PLAYING NOW</span>' : ''}
            </div>
            <div class="file-details-row">
              <span>Path: <code class="font-mono">${file.path}</code></span>
              <span>•</span>
              <span>Size: ${file.size_mb} MB</span>
              <span>•</span>
              <span>Modified: ${file.modified}</span>
            </div>
          </div>
        </div>
        <button class="btn ${isActive ? 'btn-primary' : 'btn-outline'} btn-sm play-file-btn">
          ${isActive ? 'Active' : 'Play Video'}
        </button>
      `;

      card.querySelector('.play-file-btn').addEventListener('click', async () => {
        await switchSource(file.path, file.name);
        setTimeout(() => {
          dataBrowserModal.style.display = 'none';
        }, 300);
      });

      dataFilesList.appendChild(card);
    });
  }

  // Quick Action Buttons
  btnBrowseData.addEventListener('click', openDataBrowser);
  if (btnStandbyBrowse) btnStandbyBrowse.addEventListener('click', openDataBrowser);
  if (btnStandbyWebcam) btnStandbyWebcam.addEventListener('click', async () => {
    await switchSource('0', 'Camera 0 (Default Integrated / USB)');
  });
  btnCloseDataBrowser.addEventListener('click', () => {
    dataBrowserModal.style.display = 'none';
  });

  btnRefreshSources.addEventListener('click', async () => {
    btnRefreshSources.disabled = true;
    await loadSources();
    setTimeout(() => {
      btnRefreshSources.disabled = false;
    }, 500);
  });

  btnRescanData.addEventListener('click', async () => {
    btnRescanData.disabled = true;
    await fetchAndRenderDataFiles();
    await loadSources();
    setTimeout(() => {
      btnRescanData.disabled = false;
    }, 500);
  });

  dataSearchInput.addEventListener('input', (e) => {
    currentSearchQuery = e.target.value;
    renderDataFilesList();
  });

  // Direct Custom Video Path Verification & Loading
  btnLoadManualPath.addEventListener('click', async () => {
    const enteredPath = manualPathInput.value.trim();
    if (!enteredPath) {
      manualPathStatus.textContent = 'Please enter a video file path.';
      manualPathStatus.className = 'manual-path-status error font-mono';
      return;
    }

    manualPathStatus.textContent = 'Verifying video file with OpenCV...';
    manualPathStatus.className = 'manual-path-status font-mono';

    try {
      const res = await fetch('/api/data/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: enteredPath }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Could not load video.');
      }

      manualPathStatus.textContent = `Valid video! ${data.resolution} @ ${data.fps} FPS (${data.duration_sec}s). Loading...`;
      manualPathStatus.className = 'manual-path-status success font-mono';

      await switchSource(data.path, enteredPath);
      setTimeout(() => {
        dataBrowserModal.style.display = 'none';
      }, 700);
    } catch (err) {
      manualPathStatus.textContent = `Error: ${err.message}`;
      manualPathStatus.className = 'manual-path-status error font-mono';
    }
  });

  manualPathInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      btnLoadManualPath.click();
    }
  });

  function reloadVideoStream() {
    videoStream.src = `/api/video_feed?t=${Date.now()}`;
  }

  // ==========================================================================
  // Controls Handlers
  // ==========================================================================
  btnPlayPause.addEventListener('click', async () => {
    try {
      await fetch('/api/session/pause', { method: 'POST' });
    } catch (err) {
      console.error('Error pausing session:', err);
    }
  });

  btnReset.addEventListener('click', async () => {
    try {
      await fetch('/api/session/reset', { method: 'POST' });
      pulseRenderer.clear();
      trendRenderer.clear();
    } catch (err) {
      console.error('Error resetting signal:', err);
    }
  });

  btnRecord.addEventListener('click', async () => {
    try {
      await fetch('/api/session/record/toggle', { method: 'POST' });
    } catch (err) {
      console.error('Error toggling record:', err);
    }
  });

  btnExport.addEventListener('click', () => {
    window.location.href = '/api/session/export';
  });

  toggleOverlay.addEventListener('change', async (e) => {
    try {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_overlay: e.target.checked }),
      });
    } catch (err) {
      console.error('Error updating overlay:', err);
    }
  });

  // Tab View Switcher (Pulse Wave vs BPM Trend)
  tabPulse.addEventListener('click', () => {
    tabPulse.classList.add('active');
    tabTrend.classList.remove('active');
    pulseCanvas.style.display = 'block';
    trendCanvas.style.display = 'none';
  });

  tabTrend.addEventListener('click', () => {
    tabTrend.classList.add('active');
    tabPulse.classList.remove('active');
    pulseCanvas.style.display = 'none';
    trendCanvas.style.display = 'block';
    trendRenderer.draw();
  });

  // ==========================================================================
  // Settings Modal Handlers
  // ==========================================================================
  btnOpenSettings.addEventListener('click', () => {
    settingsModal.style.display = 'flex';
  });

  btnCloseSettings.addEventListener('click', () => {
    settingsModal.style.display = 'none';
  });

  settingRoiAlpha.addEventListener('input', (e) => {
    valRoiAlpha.textContent = e.target.value;
  });

  settingQuality.addEventListener('input', (e) => {
    valQuality.textContent = e.target.value;
  });

  btnSaveSettings.addEventListener('click', async () => {
    const payload = {
      roi_alpha: parseFloat(settingRoiAlpha.value),
      jpeg_quality: parseInt(settingQuality.value),
      loop_video: settingLoop.checked,
      show_hud: settingHud.checked,
    };
    try {
      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      settingsModal.style.display = 'none';
    } catch (err) {
      console.error('Failed to save settings:', err);
    }
  });

  // ==========================================================================
  // Upload Video Modal & Drag-and-Drop
  // ==========================================================================
  btnUploadModal.addEventListener('click', () => {
    uploadModal.style.display = 'flex';
    uploadProgressBox.style.display = 'none';
  });

  btnCloseUpload.addEventListener('click', () => {
    uploadModal.style.display = 'none';
  });

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  function handleFileUpload(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    uploadProgressBox.style.display = 'flex';
    uploadProgressFill.style.width = '20%';
    uploadStatusText.textContent = `Uploading ${file.name}...`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        uploadProgressFill.style.width = `${percent}%`;
        uploadStatusText.textContent = `Uploading: ${percent}%`;
      }
    };

    xhr.onload = async () => {
      if (xhr.status === 200) {
        const resp = JSON.parse(xhr.responseText);
        uploadStatusText.textContent = 'Upload complete! Starting playback...';
        await loadSources();
        // Immediately start uploaded video
        await fetch('/api/session/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: resp.path }),
        });
        reloadVideoStream();
        setTimeout(() => {
          uploadModal.style.display = 'none';
        }, 800);
      } else {
        uploadStatusText.textContent = 'Upload failed. Check format.';
      }
    };

    xhr.onerror = () => {
      uploadStatusText.textContent = 'Network error during upload.';
    };

    xhr.send(formData);
  }

  // Close modals on backdrop click
  window.addEventListener('click', (e) => {
    if (e.target === settingsModal) settingsModal.style.display = 'none';
    if (e.target === uploadModal) uploadModal.style.display = 'none';
    if (e.target === dataBrowserModal) dataBrowserModal.style.display = 'none';
  });

  // Kick off
  loadSources();
  connectWebSocket();
});
