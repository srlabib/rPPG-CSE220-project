/**
 * High-Performance Canvas Oscilloscope & Trend Visualizer for rPPG Signals.
 */

class PulseWaveformRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    // Buffer capacity
    this.maxSamples = 240;
    this.samples = [];

    // Scaling
    this.maxAmplitude = 0.05;
    this.targetAmplitude = 0.05;

    // View dimensions
    this.width = this.canvas.width;
    this.height = this.canvas.height;

    // Handle high-DPI scaling
    this._resize();
    window.addEventListener('resize', () => this._resize());

    // Start 60 FPS render loop
    this.isRunning = true;
    this._renderLoop();
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.width = this.canvas.width;
    this.height = this.canvas.height;
    this.ctx.scale(dpr, dpr);
    this.displayWidth = rect.width;
    this.displayHeight = rect.height;
  }

  pushSample(val) {
    if (typeof val !== 'number' || isNaN(val)) return;
    this.samples.push(val);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
  }

  setSamples(sampleArray) {
    if (!Array.isArray(sampleArray)) return;
    this.samples = sampleArray.slice(-this.maxSamples);
  }

  clear() {
    this.samples = [];
  }

  _renderLoop() {
    if (!this.isRunning) return;
    this._draw();
    requestAnimationFrame(() => this._renderLoop());
  }

  _draw() {
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;

    if (!w || !h) return;

    // Clear background
    ctx.clearRect(0, 0, w, h);

    // 1. Draw subtle background oscilloscope grid
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    const gridSpacing = 30;

    ctx.beginPath();
    for (let x = 0; x < w; x += gridSpacing) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += gridSpacing) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    // 2. Draw Zero Baseline
    const zeroY = h / 2;
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.22)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    if (this.samples.length < 2) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.font = '13px Inter, sans-serif';
      ctx.fillText('Accumulating physiological pulse samples...', 24, zeroY - 12);
      return;
    }

    // 3. Dynamic Auto-scaling with smooth damping
    let localMax = 0.001;
    for (let i = 0; i < this.samples.length; i++) {
      const absVal = Math.abs(this.samples[i]);
      if (absVal > localMax) localMax = absVal;
    }

    this.targetAmplitude = Math.max(localMax * 1.15, 0.005);
    this.maxAmplitude = this.maxAmplitude * 0.92 + this.targetAmplitude * 0.08;

    // 4. Draw Oscilloscope Glowing Pulse Wave
    const step = w / (this.maxSamples - 1);
    const startIndex = Math.max(0, this.maxSamples - this.samples.length);

    // Glowing cyan-to-emerald gradient stroke
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.3)');
    grad.addColorStop(0.7, '#00f0ff');
    grad.addColorStop(1, '#00e676');

    ctx.save();
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 12;
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    let lastX = 0;
    let lastY = zeroY;

    for (let i = 0; i < this.samples.length; i++) {
      const x = (startIndex + i) * step;
      const normalized = this.samples[i] / this.maxAmplitude;
      const y = zeroY - normalized * (h * 0.42);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        // Smooth quadratic Bezier curve
        const prevX = (startIndex + i - 1) * step;
        const prevY = zeroY - (this.samples[i - 1] / this.maxAmplitude) * (h * 0.42);
        const midX = (prevX + x) / 2;
        const midY = (prevY + y) / 2;
        ctx.quadraticCurveTo(prevX, prevY, midX, midY);
      }
      lastX = x;
      lastY = y;
    }
    ctx.lineTo(lastX, lastY);
    ctx.stroke();

    // 5. Draw Phosphor Dot at the latest point
    ctx.shadowColor = '#00e676';
    ctx.shadowBlur = 16;
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }
}


class BpmTrendRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.history = [];
    this.maxPoints = 120;
    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayWidth = rect.width;
    this.displayHeight = rect.height;
    this.draw();
  }

  addPoint(bpm) {
    if (typeof bpm === 'number' && bpm > 0) {
      this.history.push(bpm);
      if (this.history.length > this.maxPoints) {
        this.history.shift();
      }
      this.draw();
    }
  }

  clear() {
    this.history = [];
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;
    if (!w || !h) return;

    ctx.clearRect(0, 0, w, h);

    if (this.history.length < 2) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '13px Inter, sans-serif';
      ctx.fillText('Accumulating heart rate trend history...', 24, h / 2);
      return;
    }

    const minBpm = Math.min(...this.history, 45);
    const maxBpm = Math.max(...this.history, 140);
    const range = Math.max(maxBpm - minBpm, 20);

    const step = w / (this.maxPoints - 1);
    const startIdx = this.maxPoints - this.history.length;

    // Gradient fill under curve
    const areaGrad = ctx.createLinearGradient(0, 0, 0, h);
    areaGrad.addColorStop(0, 'rgba(255, 42, 95, 0.25)');
    areaGrad.addColorStop(1, 'rgba(255, 42, 95, 0.0)');

    ctx.beginPath();
    for (let i = 0; i < this.history.length; i++) {
      const x = (startIdx + i) * step;
      const y = h - ((this.history[i] - minBpm) / range) * (h * 0.75) - h * 0.12;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    // Fill area
    ctx.save();
    const lastX = (startIdx + this.history.length - 1) * step;
    ctx.lineTo(lastX, h);
    ctx.lineTo(startIdx * step, h);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();
    ctx.restore();

    // Stroke line
    ctx.save();
    ctx.strokeStyle = '#ff2a5f';
    ctx.lineWidth = 2.2;
    ctx.shadowColor = '#ff2a5f';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i < this.history.length; i++) {
      const x = (startIdx + i) * step;
      const y = h - ((this.history[i] - minBpm) / range) * (h * 0.75) - h * 0.12;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }
}
