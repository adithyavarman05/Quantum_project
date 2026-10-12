/**
 * Quantum Demand Forecasting - Frontend Application Logic
 * Vanilla ES Module handling API calls, Chart.js benchmarks, and count-up animations.
 */

// Global chart registry to allow proper recreation if needed
let chartFitInstance = null;
let chartLossInstance = null;
let chartCmpInstance = null;

/**
 * Universal JSON fetch helper with status error handling.
 */
async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let message = `Server responded with ${res.status}`;
    try {
      const err = await res.json();
      if (err?.detail) {
        if (Array.isArray(err.detail)) {
          message = err.detail.map(d => d.msg || 'Validation error').join('; ');
        } else {
          message = err.detail;
        }
      }
    } catch (_) {}
    throw new Error(message);
  }
  return await res.json();
}

/**
 * Formats numbers into "12,345.6 MW" string.
 */
function formatMW(num) {
  return (
    new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(num) + ' MW'
  );
}

/**
 * Smooth ease-out cubic count-up animation for numeric results.
 */
function animateCountUp(element, targetValue, durationMs = 700) {
  const startTime = performance.now();
  const startValue = 0;

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / durationMs, 1);
    // Ease-out cubic curve
    const ease = 1 - Math.pow(1 - progress, 3);
    const currentVal = startValue + (targetValue - startValue) * ease;

    element.textContent = formatMW(currentVal);

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = formatMW(targetValue);
    }
  }

  requestAnimationFrame(update);
}

/**
 * Initialize scroll reveal observer.
 */
function initRevealObserver() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -50px 0px' }
  );

  document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
}

/**
 * Shared Chart.js styling options for Solar Flare Plasma aesthetic.
 */
function getSharedChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 900, easing: 'easeOutQuart' },
    plugins: {
      legend: {
        labels: {
          color: '#FFF4E6',
          font: { family: "'IBM Plex Mono', monospace", size: 12 },
          usePointStyle: true,
          boxWidth: 8,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(28, 10, 18, 0.95)',
        titleColor: '#FFB703',
        bodyColor: '#FFF4E6',
        borderColor: 'rgba(255, 183, 3, 0.25)',
        borderWidth: 1,
        padding: 10,
        titleFont: { family: "'IBM Plex Mono', monospace" },
        bodyFont: { family: "'IBM Plex Mono', monospace" },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(184, 154, 154, 0.15)' },
        ticks: { color: '#B89A9A', font: { family: "'IBM Plex Mono', monospace", size: 11 } },
      },
      y: {
        grid: { color: 'rgba(184, 154, 154, 0.15)' },
        ticks: { color: '#B89A9A', font: { family: "'IBM Plex Mono', monospace", size: 11 } },
      },
    },
  };
}

/**
 * Render the 3 Chart.js graphs using API benchmark data.
 */
function renderCharts(results) {
  const shared = getSharedChartOptions();

  // 1. Chart: Test Series Fit (first 150 points)
  const fitCanvas = document.getElementById('chart-fit');
  if (fitCanvas) {
    const k = Math.min(150, results.actual.length);
    const labels = Array.from({ length: k }, (_, i) => `+${i + 1}h`);
    if (chartFitInstance) chartFitInstance.destroy();

    chartFitInstance = new Chart(fitCanvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Actual Demand (MW)',
            data: results.actual.slice(0, k),
            borderColor: '#FFF4E6',
            backgroundColor: 'rgba(255, 244, 230, 0.05)',
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.15,
          },
          {
            label: 'VQC Predicted (MW)',
            data: results.vqc_pred.slice(0, k),
            borderColor: '#FFB703',
            borderDash: [4, 4],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
        ],
      },
      options: {
        ...shared,
        scales: {
          ...shared.scales,
          y: {
            ...shared.scales.y,
            title: { display: true, text: 'Demand (MW)', color: '#B89A9A', font: { family: "'IBM Plex Mono', monospace" } },
          },
        },
      },
    });
  }

  // 2. Chart: Training Loss Curve
  const lossCanvas = document.getElementById('chart-loss');
  if (lossCanvas) {
    const lossLabels = results.loss_curve.map((_, i) => `Ep ${i + 1}`);
    if (chartLossInstance) chartLossInstance.destroy();

    chartLossInstance = new Chart(lossCanvas, {
      type: 'line',
      data: {
        labels: lossLabels,
        datasets: [
          {
            label: 'MSE Loss',
            data: results.loss_curve,
            borderColor: '#FF5D3A',
            backgroundColor: 'rgba(255, 93, 58, 0.1)',
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2,
          },
        ],
      },
      options: {
        ...shared,
        scales: {
          ...shared.scales,
          x: {
            ...shared.scales.x,
            ticks: {
              ...shared.scales.x.ticks,
              maxTicksLimit: 10,
            },
          },
          y: {
            ...shared.scales.y,
            title: { display: true, text: 'MSE (Scaled [-1, 1])', color: '#B89A9A', font: { family: "'IBM Plex Mono', monospace" } },
          },
        },
      },
    });
  }

  // 3. Chart: Model Error Comparison
  const cmpCanvas = document.getElementById('chart-cmp');
  if (cmpCanvas) {
    if (chartCmpInstance) chartCmpInstance.destroy();

    chartCmpInstance = new Chart(cmpCanvas, {
      type: 'bar',
      data: {
        labels: ['MAE (MW)', 'RMSE (MW)'],
        datasets: [
          {
            label: 'VQC (Quantum)',
            data: [results.metrics.vqc.mae, results.metrics.vqc.rmse],
            backgroundColor: 'rgba(255, 183, 3, 0.85)',
            borderColor: '#FFB703',
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: 'Linear Regression',
            data: [results.metrics.linear.mae, results.metrics.linear.rmse],
            backgroundColor: 'rgba(200, 255, 46, 0.85)',
            borderColor: '#C8FF2E',
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        ...shared,
        scales: {
          ...shared.scales,
          y: {
            ...shared.scales.y,
            title: { display: true, text: 'Megawatts (MW)', color: '#B89A9A', font: { family: "'IBM Plex Mono', monospace" } },
          },
        },
      },
    });
  }
}

/**
 * Populate metrics table with accurate benchmark values.
 */
function renderMetricsTable(metrics) {
  const tbody = document.getElementById('metrics-body');
  if (!tbody) return;

  const rows = [
    { name: 'MAE', vqc: `${metrics.vqc.mae.toFixed(2)} MW`, lin: `${metrics.linear.mae.toFixed(2)} MW` },
    { name: 'RMSE', vqc: `${metrics.vqc.rmse.toFixed(2)} MW`, lin: `${metrics.linear.rmse.toFixed(2)} MW` },
    { name: 'R² Score', vqc: metrics.vqc.r2.toFixed(4), lin: metrics.linear.r2.toFixed(4) },
    {
      name: 'Training Time',
      vqc: `${metrics.vqc.train_time_s.toFixed(2)} s`,
      lin: `${(metrics.linear.train_time_s * 1000).toFixed(1)} ms`,
    },
    { name: 'Parameters', vqc: `${metrics.vqc.params}`, lin: `${metrics.linear.params}` },
  ];

  tbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td>${r.name}</td>
        <td class="mono-num vqc-accent">${r.vqc}</td>
        <td class="mono-num lin-accent">${r.lin}</td>
      </tr>
    `
    )
    .join('');
}

/**
 * Load initial sample values and populate inputs.
 */
async function loadSampleValues() {
  try {
    const data = await api('/api/sample');
    if (Array.isArray(data.values) && data.values.length === 4) {
      data.values.forEach((val, idx) => {
        const inp = document.getElementById(`v${idx}`);
        if (inp) inp.value = Math.round(val);
      });
    }
    if (Array.isArray(data.labels) && data.labels.length === 4) {
      data.labels.forEach((lbl, idx) => {
        const stepEl = document.getElementById(`lbl-step-${idx}`);
        if (stepEl) stepEl.textContent = lbl;
      });
    }
  } catch (err) {
    const feedback = document.getElementById('predict-feedback');
    if (feedback) {
      feedback.textContent = `Could not load sample data: ${err.message}`;
      feedback.className = 'feedback-msg error';
    }
  }
}

/**
 * Load benchmark results and construct table and charts.
 */
async function loadResults() {
  try {
    const data = await api('/api/results');
    renderMetricsTable(data.metrics);
    renderCharts(data);
  } catch (err) {
    const feedback = document.getElementById('predict-feedback');
    if (feedback) {
      feedback.textContent = `Could not load results: ${err.message}`;
      feedback.className = 'feedback-msg error';
    }
  }
}

/**
 * Setup predict button click handler and validation.
 */
function setupPredictAction() {
  const btn = document.getElementById('go');
  const feedback = document.getElementById('predict-feedback');
  const outVqc = document.getElementById('out-vqc');
  const outLin = document.getElementById('out-lin');

  if (!btn) return;

  btn.addEventListener('click', async () => {
    feedback.textContent = '';
    feedback.className = 'feedback-msg';

    // 1. Validate 4 inputs
    const values = [];
    for (let i = 0; i < 4; i++) {
      const inp = document.getElementById(`v${i}`);
      const val = parseFloat(inp?.value);
      if (isNaN(val) || !isFinite(val) || val < 0 || val > 100000) {
        feedback.textContent = `Please enter a valid electricity load (0 to 100,000 MW) for t-${4 - i}h.`;
        feedback.className = 'feedback-msg error';
        if (inp) inp.focus();
        return;
      }
      values.push(val);
    }

    // 2. Set loading state
    btn.classList.add('loading');
    btn.disabled = true;

    try {
      const data = await api('/api/predict', {
        method: 'POST',
        body: JSON.stringify({ values }),
      });

      // 3. Count-up animation
      if (outVqc) animateCountUp(outVqc, data.vqc_mw, 700);
      if (outLin) animateCountUp(outLin, data.linear_mw, 700);

      feedback.textContent = 'Predicted next hour demand computed with PennyLane simulation.';
      feedback.className = 'feedback-msg success';

      // 4. Trigger 3D Bloch visualization sequence if scene is loaded
      if (typeof window.playQuantumSequence === 'function') {
        window.playQuantumSequence(data);
      }
    } catch (err) {
      feedback.textContent = `Prediction failed: ${err.message}`;
      feedback.className = 'feedback-msg error';
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  });
}

// Initial bootstrap on module load
document.addEventListener('DOMContentLoaded', () => {
  initRevealObserver();
  loadSampleValues();
  loadResults();
  setupPredictAction();
});
