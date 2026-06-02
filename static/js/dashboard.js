let shapChart = null;
let shapChartFull = null;
let lastShapData = null;
let telemetryLogs = [];
let selectedAlertId = null;
let autoRefreshEnabled = true;
let refreshInterval = null;

const RISK_STYLES = {
  HIGH: {
    badge: "risk-badge-HIGH",
    row: "risk-HIGH",
    color: "#dc3545",
  },
  MEDIUM: {
    badge: "risk-badge-MEDIUM",
    row: "risk-MEDIUM",
    color: "#ffc107",
  },
  LOW: {
    badge: "risk-badge-LOW",
    row: "risk-LOW",
    color: "#6c757d",
  },
};

function _createToast(text, isError=false) {
  const toast = document.createElement('div');
  toast.innerText = text;
  toast.style.position = 'fixed';
  toast.style.right = '18px';
  toast.style.bottom = '18px';
  toast.style.padding = '10px 14px';
  toast.style.background = isError ? '#6b1d1d' : 'var(--surface2)';
  toast.style.color = 'var(--text)';
  toast.style.border = '1px solid var(--border)';
  toast.style.borderRadius = '6px';
  toast.style.zIndex = 80;
  document.body.appendChild(toast);
  setTimeout(() => { toast.remove(); }, 4200);
}

function addLogControls() {
  const logPanel = document.querySelector(".log-feed-panel");
  if (!logPanel) return;
  
  const header = logPanel.querySelector(".panel-header");
  if (header) {
    const controls = document.createElement("div");
    controls.style.display = "flex";
    controls.style.gap = "8px";
    controls.style.marginLeft = "auto";
    
    const downloadBtn = document.createElement("button");
    downloadBtn.className = "btn-icon-sm";
    downloadBtn.textContent = "⬇ Download";
    downloadBtn.type = "button";
    downloadBtn.addEventListener("click", downloadLogs);
    
    const viewBtn = document.createElement("button");
    viewBtn.className = "btn-icon-sm";
    viewBtn.textContent = "📄 View Logs";
    viewBtn.type = "button";
    viewBtn.addEventListener("click", () => {
      window.location.href = "/dashboard/logs";
    });

    controls.appendChild(downloadBtn);
    controls.appendChild(viewBtn);
    header.appendChild(controls);
  }
}

async function downloadLogs() {
  try {
    // Download processed logs by default; let user pass ?type=raw in URL if needed
    const response = await fetch("/api/logs/download?type=processed");
    if (!response.ok) throw new Error("Download failed");
    const data = await response.json();
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sentinel-logs-${data.type || 'processed'}-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (error) {
    _createToast("Failed to download logs", true);
  }
}

async function clearLogs() {
  try {
    const response = await fetch("/api/logs/clear", { method: "POST" });
    if (!response.ok) throw new Error("Clear failed");
    telemetryLogs = [];
    document.getElementById("logFeed").innerHTML = "";
    _createToast("Logs cleared successfully");
  } catch (error) {
    _createToast("Failed to clear logs", true);
  }
}

async function initDashboard() {
  addLogControls();
  initUploadForm();
  await applyDashboardSettings();
  await refreshDashboard();
}

async function applyDashboardSettings() {
  try {
    const response = await fetch('/api/settings', { cache: 'no-store' });
    if (!response.ok) throw new Error('Settings unavailable');
    const settings = await response.json();
    const intervalSeconds = Number(settings.auto_refresh) || 18;
    const intervalMs = Math.max(5, Math.min(intervalSeconds, 120)) * 1000;
    scheduleDashboardRefresh(intervalMs);
  } catch (error) {
    scheduleDashboardRefresh(3000);
  }
}

function scheduleDashboardRefresh(intervalMs) {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
  refreshInterval = setInterval(refreshDashboard, intervalMs);
}

async function refreshDashboard() {
  if (!autoRefreshEnabled) return;
  await Promise.all([fetchTelemetry(), fetchStats()]);
}

async function fetchTelemetry() {
  try {
    const response = await fetch("/api/telemetry", { cache: "no-store" });
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }

    const data = await response.json();
    renderAlerts(data.alerts || []);
    telemetryLogs = [...telemetryLogs, ...(data.logs || [])].slice(-250);
    renderLogs(telemetryLogs);

    if (!selectedAlertId && data.alerts && data.alerts.length > 0) {
      selectAlert(data.alerts[0]);
    }

  } catch (error) {
    _createToast("Telemetry fetch failed", true);
  }
}

async function fetchStats() {
  try {
    const response = await fetch("/api/stats", { cache: "no-store" });
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }

    const stats = await response.json();
    setText("activeAlerts", stats.active_alerts || 0);
    setText("highRisk", stats.high_risk || 0);
    setText("mediumRisk", stats.medium_risk || 0);
    setText("resolvedCount", stats.resolved || 0);
  } catch (error) {
    _createToast("Stats fetch failed", true);
  }
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.innerText = value;
  }
}

function renderAlerts(alerts) {
  const container = document.getElementById("alertContainer");
  if (!container) return;

  container.innerHTML = "";

  if (!alerts.length) {
    container.innerHTML = `<div class="empty-state">No active threats</div>`;
    return;
  }

  alerts.forEach(alert => {
    const card = document.createElement("div");
    card.className = `alert-card ${alert.risk ? RISK_STYLES[alert.risk]?.row : ""}`.trim();
    card.style.borderLeft = `4px solid ${RISK_STYLES[alert.risk]?.color || "#6c757d"}`;
    card.dataset.uid = alert.user_id;

    if (selectedAlertId === alert.user_id) {
      card.classList.add("selected-alert");
    }

    const isLocked = Boolean(alert.locked);
    const lockLabel = isLocked ? `<span style="font-size:0.75rem;color:#dc3545;font-weight:700;margin-right:6px;">LOCKED</span>` : ``;
    const actionLabel = isLocked ? '🔓 Unlock Account' : '🔒 Lock Account';
    const actionType = isLocked ? 'unlock_account' : 'lock_account';

    card.innerHTML = `
      <div class="alert-top">
        <span class="ac-uid">${alert.user_id}</span>
        <span class="ac-risk-badge ${RISK_STYLES[alert.risk]?.badge || "risk-badge-LOW"}">
          ${alert.risk}
        </span>
      </div>
      <div class="ac-dept">${alert.department}</div>
      <div class="ac-field">${lockLabel}${alert.threat_category.replace(/\b\w/g, m => m.toUpperCase())}</div>
      <div class="ac-field">Score: ${alert.score.toFixed(4)}</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;">
        <button class="btn-secondary" type="button">🔍 View Details</button>
        <button class="btn-resolve action-button" type="button" data-action="${actionType}">${actionLabel}</button>
      </div>
    `;

    card.addEventListener("click", () => selectAlert(alert));
    const detailsButton = card.querySelector(".btn-secondary");
    if (detailsButton) {
      detailsButton.addEventListener("click", async event => {
        event.stopPropagation();
        selectAlert(alert);
      });
    }

    const actionButton = card.querySelector(".action-button");
    if (actionButton) {
      actionButton.addEventListener("click", async event => {
        event.stopPropagation();
        await resolveAlert(alert.user_id, actionButton.dataset.action || 'resolve');
      });
    }

    container.appendChild(card);
  });
}

function renderLogs(logs) {
  const feed = document.getElementById("logFeed");
  if (!feed) return;

  feed.innerHTML = "";
  const latest = [...logs].slice(-50).reverse();

  if (!latest.length) {
    feed.innerHTML = `<div class="empty-state">No log events yet</div>`;
    return;
  }

  latest.forEach(log => {
    const row = document.createElement("div");
    row.className = `log-row ${RISK_STYLES[log.risk]?.row || ""}`.trim();
    row.innerHTML = `
      <div class="col col-ts">${log.time_short}</div>
      <div class="col col-eid">${log.event_id}</div>
      <div class="col col-user">${log.user_id}</div>
      <div class="col col-dept">${log.category}</div>
      <div class="col col-ip">${log.source}</div>
      <div class="col col-host ${log.risk ? `risk-col-${log.risk}` : ""}">${log.risk}</div>
    `;
    feed.appendChild(row);
  });
}

async function selectAlert(alert) {
  selectedAlertId = alert.user_id;
  updateAlertSelection();
  renderInvestigationPanel(alert);
  renderShapChart(alert);
}

function updateAlertSelection() {
  document.querySelectorAll("#alertContainer .alert-card").forEach(card => {
    if (card.dataset.uid === selectedAlertId) {
      card.classList.add("selected-alert");
    } else {
      card.classList.remove("selected-alert");
    }
  });
}

async function renderInvestigationPanel(alert) {
  const container = document.getElementById("investigationPanel");
  if (!container) return;

  container.innerHTML = `<div class="empty-state"><div class="empty-state">Loading alert details...</div></div>`;

  const details = await fetchUserDetails(alert.user_id);
  const burnRate = details.burn_rate ?? alert.burn_rate;
  const recentEvents = details.recent_events || [];

  container.innerHTML = `
    <div class="inv-content">
      <div class="inv-avatar">${alert.user_id.charAt(0) || "U"}</div>
      <div class="inv-name">${alert.user_id}</div>
      <div class="inv-dept">${alert.department} · ${alert.role}</div>
      <div class="inv-score-box">
        <div class="inv-score-label">Current Risk Score</div>
        <div class="inv-score-value risk-${alert.risk}">${alert.score.toFixed(4)}</div>
      </div>
      <div class="burn-bar-container">
        <div class="burn-bar-label">
          <span>Burn Rate</span>
          <span>${burnRate} anomalies</span>
        </div>
        <div class="burn-bar-track">
          <div class="burn-bar-fill" style="width: ${Math.min(100, burnRate * 5)}%;"></div>
        </div>
      </div>
      <div class="inv-field-row"><span class="inv-field-label">Email</span><span class="inv-field-value">${alert.email}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Department</span><span class="inv-field-value">${details.department || alert.department}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Supervisor</span><span class="inv-field-value">${details.supervisor || "Unknown"}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Team</span><span class="inv-field-value">${details.team || "N/A"}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Business Unit</span><span class="inv-field-value">${details.business_unit || "N/A"}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Host</span><span class="inv-field-value">${alert.hostname}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">IP</span><span class="inv-field-value">${alert.source_ip}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Threat</span><span class="inv-field-value">${alert.threat_category}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Actual Status</span><span class="inv-field-value">${alert.actual_label || 'Unknown'}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Event</span><span class="inv-field-value">${alert.event_type}</span></div>
      <div class="inv-field-row"><span class="inv-field-label">Recent Events</span><span class="inv-field-value">${recentEvents.length}</span></div>
      <div style="display:flex;gap:8px;margin-top:16px;flex-wrap:wrap;">
        <button class="btn-primary" type="button" onclick="investigateAlert('${alert.user_id}')">📋 Create Investigation Case</button>
        <button class="btn-resolve" type="button" onclick="resolveAlert('${alert.user_id}', 'resolve')">✓ Resolve Alert</button>
        <button class="btn-resolve" type="button" style="border-color:#dc3545;color:#dc3545;" onclick="resolveAlert('${alert.user_id}', '${alert.locked ? 'unlock_account' : 'lock_account'}')">${alert.locked ? '🔓 Unlock Account' : '🔒 Lock Account'}</button>
      </div>
    </div>
  `;
}

async function fetchUserDetails(uid) {
  try {
    const response = await fetch(`/api/user/${encodeURIComponent(uid)}`, { cache: "no-store" });
    if (!response.ok) return {};
    return await response.json();
  } catch (error) {
    // failed to load user details — return empty
    return {};
  }
}

async function investigateAlert(uid) {
  try {
    const response = await fetch("/api/case/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        user_id: uid,
        reason: "Alert investigation from dashboard"
      }),
    });

    if (response.ok) {
      const data = await response.json();
      window.location.href = `/dashboard/investigation?case=${data.case_id}`;
    } else {
      _createToast("Failed to create investigation case", true);
    }
  } catch (error) {
    _createToast("Error creating case", true);
  }
}

async function resolveAlert(uid, action = "resolve") {
  try {
    const response = await fetch("/api/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        user_id: uid,
        action: action
      }),
    });

    if (response.ok) {
      selectedAlertId = null;
      await refreshDashboard();
      if (action === "lock_account") {
        _createToast(`Account ${uid} has been locked`);
      }
    }
  } catch (error) {
    _createToast("Resolve request failed", true);
  }
}

function renderShapChart(data) {
  const canvas = document.getElementById("shapChart");
  if (!canvas) return;

  lastShapData = data;

  if (shapChart) {
    shapChart.destroy();
    shapChart = null;
  }

  if (shapChartFull) {
    shapChartFull.destroy();
    shapChartFull = null;
  }

  const shapPanel = canvas.closest(".shap-panel");
  if (shapPanel && !shapPanel.querySelector(".shap-controls")) {
    const controls = document.createElement("div");
    controls.className = "shap-controls";
    controls.innerHTML = `
      <span class="shap-target">Target: ${data.user_id}</span>
      <button class="btn-fullscreen" type="button" onclick="expandShapChart()">⛶ Fullscreen</button>
    `;
    const header = shapPanel.querySelector(".panel-header");
    if (header) {
      header.appendChild(controls);
    }
  }

  const labelColor = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#8b949e";
  const labels = Array.isArray(data.feature_labels) && data.feature_labels.length ? data.feature_labels : (Array.isArray(data.shap_values) ? data.shap_values.map((_, index) => `Feature ${index + 1}`) : []);
  const values = Array.isArray(data.shap_values) ? data.shap_values : [];
  const backgroundColor = values.map(value => (value >= 0 ? "rgba(220,53,69,0.8)" : "rgba(40,167,69,0.8)"));

  shapChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Feature Impact",
          data: values,
          backgroundColor,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const value = ctx.parsed.y;
              const direction = value >= 0 ? 'Increases risk' : 'Reduces risk';
              return `${ctx.label}: ${value.toFixed(3)} (${direction})`;
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: labelColor }, grid: { color: "rgba(255,255,255,0.08)" } },
        y: { beginAtZero: true, ticks: { color: labelColor }, grid: { color: "rgba(255,255,255,0.08)" } },
      },
    },
  });
}

async function expandShapChart() {
  const modal = document.getElementById("shapModal") || createShapModal();
  if (!lastShapData) return;

  modal.style.display = "flex";
  await new Promise(requestAnimationFrame);

  const fullCanvas = document.getElementById("shapChartFull");
  if (!fullCanvas) return;

  if (shapChartFull) {
    shapChartFull.destroy();
    shapChartFull = null;
  }

  const labelColor = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() || "#8b949e";
  const labels = Array.isArray(lastShapData.feature_labels) && lastShapData.feature_labels.length ? lastShapData.feature_labels : (Array.isArray(lastShapData.shap_values) ? lastShapData.shap_values.map((_, index) => `Feature ${index + 1}`) : []);
  const values = Array.isArray(lastShapData.shap_values) ? lastShapData.shap_values : [];
  const backgroundColor = values.map(value => (value >= 0 ? "rgba(220,53,69,0.7)" : "rgba(108,117,125,0.7)"));

  shapChartFull = new Chart(fullCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Feature Impact",
          data: values,
          backgroundColor,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: labelColor }, grid: { color: "rgba(255,255,255,0.08)" } },
        y: { beginAtZero: true, ticks: { color: labelColor }, grid: { color: "rgba(255,255,255,0.08)" } },
      },
    },
  });
}

function closeShapModal() {
  const modal = document.getElementById("shapModal");
  if (!modal) return;
  modal.style.display = "none";
  if (shapChartFull) {
    shapChartFull.destroy();
    shapChartFull = null;
  }
}

function createShapModal() {
  const modal = document.createElement("div");
  modal.id = "shapModal";
  modal.className = "modal-overlay";
  modal.style.display = "none";
  modal.style.flexDirection = "column";
  modal.innerHTML = `
    <div class="modal-box" style="height: 90vh; max-height: 90vh;">
      <div class="modal-header">
        <h5>SHAP Feature Contribution Analysis</h5>
        <button style="background:transparent;border:none;color:var(--text);font-size:1.2rem;cursor:pointer;" onclick="closeShapModal()">✕</button>
      </div>
      <div class="modal-body">
        <canvas id="shapChartFull"></canvas>
      </div>
    </div>
  `;

  modal.addEventListener("click", event => {
    if (event.target === modal) {
      closeShapModal();
    }
  });

  document.body.appendChild(modal);
  return modal;
}

function initUploadForm() {
  const uploadZone = document.getElementById("overviewUploadZone");
  const fileInput = document.getElementById("overviewFileInput");
  const uploadStatus = document.getElementById("overviewUploadStatus");
  const statusMessage = document.getElementById("overviewStatusMessage");
  const progressBar = document.getElementById("overviewProgressBar");
  const resultsContainer = document.getElementById("overviewResultsContainer");

  if (!uploadZone || !fileInput || !uploadStatus || !statusMessage || !progressBar || !resultsContainer) {
    return;
  }

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadZone.classList.remove("dragover");
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        uploadOverviewFile(files[0]);
      }
    });

    fileInput.addEventListener("change", () => uploadOverviewFile());

    async function uploadOverviewFile(droppedFile) {
      const file = droppedFile || fileInput.files[0];
      if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    uploadStatus.style.display = "block";
    statusMessage.innerText = "Processing upload...";
    progressBar.style.width = "30%";

    try {
      const response = await fetch("/api/logs/upload", { method: "POST", body: formData });
      progressBar.style.width = "60%";
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Upload failed");
      }
      const data = await response.json();
      progressBar.style.width = "100%";
      statusMessage.innerText = `Processed ${data.count} logs.`;
      renderUploadResults(resultsContainer, data.results || []);
      fileInput.value = "";
    } catch (error) {
      statusMessage.innerText = `Upload error: ${error.message}`;
      progressBar.style.width = "0%";
    }
  }

  function renderUploadResults(container, results) {
    container.innerHTML = "";
    if (!results.length) {
      container.innerHTML = `<div class="empty-state">No upload results available</div>`;
      return;
    }

    const table = document.createElement("table");
    table.className = "user-table";
    table.innerHTML = `
      <thead>
        <tr>
          <th>User</th>
          <th>Risk</th>
          <th>Score</th>
          <th>Threat</th>
          <th>Dept</th>
          <th>Email</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector("tbody");
    results.slice(0, 20).forEach(item => {
      const row = document.createElement("tr");
      const riskColor = {
        HIGH: "var(--risk-high)",
        MEDIUM: "var(--risk-medium)",
        LOW: "var(--risk-low)",
      }[item.risk] || "var(--muted)";
      row.innerHTML = `
        <td>${item.user_id || "-"}</td>
        <td style="color:${riskColor};font-weight:600;">${item.risk || "-"}</td>
        <td>${item.score != null ? item.score.toFixed(4) : "-"}</td>
        <td>${item.threat_category || "-"}</td>
        <td>${item.department || "-"}</td>
        <td>${item.email || "-"}</td>
      `;
      tbody.appendChild(row);
    });

    container.appendChild(table);
  }
}

window.addEventListener("DOMContentLoaded", initDashboard);

