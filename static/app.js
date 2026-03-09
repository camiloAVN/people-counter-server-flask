/**
 * Contador de Personas — Dashboard frontend
 *
 * Responsabilidades:
 *  - Conectar al servidor vía WebSocket y recibir estadísticas en vivo.
 *  - Actualizar todos los indicadores del dashboard.
 *  - Renderizar el gráfico de barras con Chart.js.
 *  - Reconectar automáticamente si se pierde la conexión.
 *  - Manejar la descarga del CSV y el reset de contadores.
 */

'use strict';

// ── Constantes ──────────────────────────────────────────────────────────────
const RECONNECT_DELAY_MS = 3000;
const MODE_LABELS = {
  line: 'Modo: cruce de línea',
  roi:  'Modo: zona rectangular',
  fov:  'Modo: campo de visión',
};

// ── Estado local ────────────────────────────────────────────────────────────
let chart        = null;
let ws           = null;
let sessionStart = null;   // hora de la primera estadística recibida

// Modo activo en el panel (puede diferir del servidor hasta aplicar)
let selectedMode = 'line';
// Flag para evitar que el primer WS sync sobreescriba una selección activa del usuario
let configSyncedOnce = false;

// ── Inicialización ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  initEventListeners();
  initConfigPanel();
  connectWebSocket();
  startClock();
  monitorVideoFeed();
});

// ── Chart.js ────────────────────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('hourlyChart').getContext('2d');

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels:   [],
      datasets: [{
        label:           'Visitas',
        data:            [],
        backgroundColor: 'rgba(0, 232, 122, 0.55)',
        borderColor:     'rgba(0, 232, 122, 0.9)',
        borderWidth:     1,
        borderRadius:    4,
        borderSkipped:   false,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.raw} visitas`,
          },
        },
      },
      scales: {
        x: {
          ticks:  { color: '#7070a0', font: { size: 11 } },
          grid:   { color: '#1e1e3a' },
        },
        y: {
          beginAtZero: true,
          ticks: {
            color:    '#7070a0',
            font:     { size: 11 },
            stepSize: 1,
            precision: 0,
          },
          grid: { color: '#1e1e3a' },
        },
      },
    },
  });
}

function updateChart(hourlyData) {
  if (!chart) return;

  const hours  = Object.keys(hourlyData).map(Number).sort((a, b) => a - b);
  const labels = hours.map((h) => `${h}:00`);
  const values = hours.map((h) => hourlyData[String(h)] ?? 0);

  // Marcar la barra de hora pico con un color diferente
  const maxVal     = values.length ? Math.max(...values) : 0;
  const colors     = values.map((v) =>
    v === maxVal && maxVal > 0
      ? 'rgba(255, 209, 102, 0.8)'
      : 'rgba(0, 232, 122, 0.55)',
  );
  const borderColors = values.map((v) =>
    v === maxVal && maxVal > 0
      ? 'rgba(255, 209, 102, 1)'
      : 'rgba(0, 232, 122, 0.9)',
  );

  chart.data.labels                                    = labels;
  chart.data.datasets[0].data                          = values;
  chart.data.datasets[0].backgroundColor               = colors;
  chart.data.datasets[0].borderColor                   = borderColors;
  chart.update('none');   // sin animación para actualizaciones frecuentes
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus(true);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (e) {
      console.warn('Mensaje WS inválido:', e);
    }
  };

  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWebSocket, RECONNECT_DELAY_MS);
  };

  ws.onerror = () => {
    setStatus(false);
    ws.close();
  };
}

// ── Actualización del dashboard ──────────────────────────────────────────────
function updateDashboard(data) {
  // Registrar hora de inicio al recibir el primer mensaje
  if (sessionStart === null) {
    sessionStart = new Date();
    setText('sessionStart', sessionStart.toLocaleTimeString('es', {
      hour: '2-digit', minute: '2-digit',
    }));
  }

  // Contadores principales
  setText('inCount',  data.in_count  ?? data.fov_count ?? 0);
  setText('outCount', data.out_count ?? 0);
  setText('inFrame',  data.persons_in_frame ?? 0);
  setText('inFrameOverlayVal', data.persons_in_frame ?? 0);

  // Flujo neto
  const net    = data.net_flow ?? 0;
  const netEl  = document.getElementById('netFlow');
  if (netEl) {
    netEl.textContent = net >= 0 ? `+${net}` : String(net);
    netEl.style.color = net >= 0 ? 'var(--blue)' : 'var(--red)';
  }

  // FPS
  const fpsBadge = document.getElementById('fpsBadge');
  if (fpsBadge) fpsBadge.textContent = `${data.fps ?? '--'} fps`;

  // Modo de conteo
  const modeBadge = document.getElementById('modeBadge');
  if (modeBadge) modeBadge.textContent = MODE_LABELS[data.counting_mode] ?? data.counting_mode;

  // Tarjetas de analítica
  if (data.peak_hour !== null && data.peak_hour !== undefined) {
    setText('peakHour', `${data.peak_hour}:00  (${data.peak_count})`);
  }
  setText('hourlyAvg',   data.hourly_average ?? 0);
  setText('totalVisits', data.total_today    ?? 0);

  // Gráfico
  if (data.hourly_entries) {
    updateChart(data.hourly_entries);
  }

  // Sincronizar panel de configuración con el estado del servidor (solo primera vez)
  syncConfigPanel(data);

  // Adaptar etiquetas según el modo
  if (data.counting_mode === 'fov') {
    setLabelText('inCount',  'Personas vistas');
    setLabelText('outCount', 'N/A');
    setText('outCount', '—');
  } else {
    setLabelText('inCount',  'Entradas hoy');
    setLabelText('outCount', 'Salidas hoy');
  }
}

// ── Indicador de estado ──────────────────────────────────────────────────────
function setStatus(online) {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  if (dot)  { dot.className  = `status-dot ${online ? 'online' : 'offline'}`; }
  if (text) { text.textContent = online ? 'En línea' : 'Sin conexión — reconectando…'; }
}

// ── Reloj en tiempo real ─────────────────────────────────────────────────────
function startClock() {
  const tick = () => {
    const el = document.getElementById('currentTime');
    if (el) {
      el.textContent = new Date().toLocaleTimeString('es', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
    }
  };
  tick();
  setInterval(tick, 1000);
}

// ── Monitor del feed de video ────────────────────────────────────────────────
/**
 * Recarga el stream MJPEG si la imagen deja de cargar
 * (p.ej. al reconectar el servidor).
 */
function monitorVideoFeed() {
  const img = document.getElementById('videoFeed');
  if (!img) return;

  img.addEventListener('error', () => {
    setTimeout(() => {
      img.src = `/video_feed?t=${Date.now()}`;
    }, 2000);
  });
}

// ── Eventos de botones ───────────────────────────────────────────────────────
function initEventListeners() {
  document.getElementById('btnDownload')?.addEventListener('click', handleDownload);
  document.getElementById('btnReset')?.addEventListener('click', handleReset);
}

// ── Panel de configuración del modo ─────────────────────────────────────────
function initConfigPanel() {
  // Botones de modo
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      selectedMode = btn.dataset.mode;
      updateModeBtnUI(selectedMode);
      showModeParams(selectedMode);
    });
  });

  // Sliders de línea
  linkSlider('sliderHpos',      'valHpos',      (v) => `${v}%`);
  linkSlider('sliderVpos',      'valVpos',      (v) => `${v}%`);
  linkSlider('sliderRoiX1',     'valRoiX1',     (v) => `${v}%`);
  linkSlider('sliderRoiY1',     'valRoiY1',     (v) => `${v}%`);
  linkSlider('sliderRoiX2',     'valRoiX2',     (v) => `${v}%`);
  linkSlider('sliderRoiY2',     'valRoiY2',     (v) => `${v}%`);
  linkSlider('sliderConfidence','valConfidence', (v) => `${v}%`);

  // Checkboxes de línea: mostrar/ocultar slider asociado cuando está deshabilitado
  document.getElementById('useHorizontal')?.addEventListener('change', (e) => {
    const row = document.getElementById('rowHpos');
    if (row) row.style.opacity = e.target.checked ? '1' : '0.4';
  });
  document.getElementById('useVertical')?.addEventListener('change', (e) => {
    const row = document.getElementById('rowVpos');
    if (row) row.style.opacity = e.target.checked ? '1' : '0.4';
  });

  // Botón aplicar
  document.getElementById('btnApplyConfig')?.addEventListener('click', handleApplyConfig);
}

function linkSlider(sliderId, valueId, format) {
  const slider = document.getElementById(sliderId);
  if (!slider) return;
  slider.addEventListener('input', () => {
    setText(valueId, format(slider.value));
  });
}

function updateModeBtnUI(mode) {
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
}

function showModeParams(mode) {
  const panels = { line: 'paramsLine', roi: 'paramsRoi', fov: 'paramsFov' };
  Object.entries(panels).forEach(([m, id]) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden', m !== mode);
  });
}

function syncConfigPanel(data) {
  // Solo sincronizar con el servidor la primera vez (evitar pisar cambios del usuario)
  if (configSyncedOnce) return;
  configSyncedOnce = true;

  const mode = data.counting_mode ?? 'line';
  selectedMode = mode;
  updateModeBtnUI(mode);
  showModeParams(mode);

  // Línea horizontal
  const useH = data.use_horizontal_line ?? true;
  const chkH = document.getElementById('useHorizontal');
  if (chkH) {
    chkH.checked = useH;
    const row = document.getElementById('rowHpos');
    if (row) row.style.opacity = useH ? '1' : '0.4';
  }
  setSlider('sliderHpos', 'valHpos', Math.round((data.line_position ?? 0.5) * 100));

  // Línea vertical
  const useV = data.use_vertical_line ?? false;
  const chkV = document.getElementById('useVertical');
  if (chkV) {
    chkV.checked = useV;
    const row = document.getElementById('rowVpos');
    if (row) row.style.opacity = useV ? '1' : '0.4';
  }
  setSlider('sliderVpos', 'valVpos', Math.round((data.line_position_vertical ?? 0.5) * 100));

  // ROI
  setSlider('sliderRoiX1', 'valRoiX1', Math.round((data.roi_x1 ?? 0) * 100));
  setSlider('sliderRoiY1', 'valRoiY1', Math.round((data.roi_y1 ?? 0) * 100));
  setSlider('sliderRoiX2', 'valRoiX2', Math.round((data.roi_x2 ?? 1) * 100));
  setSlider('sliderRoiY2', 'valRoiY2', Math.round((data.roi_y2 ?? 1) * 100));

  // Confianza
  setSlider('sliderConfidence', 'valConfidence', Math.round((data.confidence ?? 0.3) * 100));
}

function setSlider(sliderId, valueId, intValue) {
  const slider = document.getElementById(sliderId);
  if (slider) slider.value = intValue;
  setText(valueId, `${intValue}%`);
}

async function handleApplyConfig() {
  const btn = document.getElementById('btnApplyConfig');
  if (btn) { btn.disabled = true; btn.textContent = 'Aplicando…'; }

  const payload = { counting_mode: selectedMode };

  if (selectedMode === 'line') {
    payload.use_horizontal_line = document.getElementById('useHorizontal')?.checked ?? true;
    payload.use_vertical_line   = document.getElementById('useVertical')?.checked ?? false;
    payload.line_position           = (parseInt(document.getElementById('sliderHpos')?.value ?? 50) / 100);
    payload.line_position_vertical  = (parseInt(document.getElementById('sliderVpos')?.value ?? 50) / 100);
  } else if (selectedMode === 'roi') {
    payload.roi_x1 = parseInt(document.getElementById('sliderRoiX1')?.value ?? 0)   / 100;
    payload.roi_y1 = parseInt(document.getElementById('sliderRoiY1')?.value ?? 0)   / 100;
    payload.roi_x2 = parseInt(document.getElementById('sliderRoiX2')?.value ?? 100) / 100;
    payload.roi_y2 = parseInt(document.getElementById('sliderRoiY2')?.value ?? 100) / 100;
  }

  payload.confidence = parseInt(document.getElementById('sliderConfidence')?.value ?? 30) / 100;

  try {
    const resp = await fetch('/api/config', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    if (resp.ok) {
      if (btn) { btn.textContent = '✓ Aplicado'; }
      setTimeout(() => {
        if (btn) { btn.textContent = '✓ Aplicar configuración'; btn.disabled = false; }
      }, 1500);
    } else {
      if (btn) { btn.textContent = '✗ Error al aplicar'; btn.disabled = false; }
    }
  } catch (e) {
    if (btn) { btn.textContent = `✗ Error: ${e.message}`; btn.disabled = false; }
  }
}

function handleDownload() {
  window.location.href = '/api/export/csv';
}

async function handleReset() {
  const confirmed = confirm('¿Reiniciar todos los contadores a cero?\nEsta acción no se puede deshacer.');
  if (!confirmed) return;

  try {
    const resp = await fetch('/api/reset', { method: 'POST' });
    if (resp.ok) {
      sessionStart = null;
      setText('sessionStart', '--:--');
      updateChart({});
    } else {
      alert('Error al reiniciar los contadores.');
    }
  } catch (e) {
    alert(`Error de red: ${e.message}`);
  }
}

// ── Utilidades DOM ───────────────────────────────────────────────────────────
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setLabelText(valueId, labelText) {
  const valueEl = document.getElementById(valueId);
  if (!valueEl) return;
  const card     = valueEl.closest('.metric-card');
  const labelEl  = card?.querySelector('.metric-label');
  if (labelEl) labelEl.textContent = labelText;
}