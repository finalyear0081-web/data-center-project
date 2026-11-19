/**
 * Dashboard Controller and API Integration Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    initAuthStatus();
    loadDashboardSummary();
    loadForecastData();
    loadTelemetry(24);
    loadCoolingHistory();
    loadDatasetMetadataAndPreview();
    loadModelBenchmarks();
    loadAlertsFeed();

    // Auto-refresh summary and forecast every 30 seconds
    setInterval(() => {
        loadDashboardSummary();
        loadCoolingHistory();
        loadAlertsFeed();
    }, 30000);

    setupEventListeners();
});

function initAuthStatus() {
    const userStr = localStorage.getItem('user');
    if (userStr) {
        try {
            const user = JSON.parse(userStr);
            document.getElementById('navUserName').textContent = user.display_name || user.email.split('@')[0];
            document.getElementById('navUserRole').textContent = user.role || 'operator';
            document.getElementById('navUserEmail').textContent = user.email;
            
            // If operator, disable admin-only controls
            if (user.role !== 'admin') {
                const btnSaveTh = document.getElementById('btnSaveThresholds');
                if (btnSaveTh) {
                    btnSaveTh.disabled = true;
                    btnSaveTh.title = 'Administrator privileges required to change thresholds.';
                }
            }
        } catch (e) {}
    }
}

function handleLogout() {
    fetch('/api/auth/logout', { method: 'POST' }).finally(() => {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user');
        window.location.href = '/login';
    });
}

function switchToTab(tabId) {
    const triggerEl = document.querySelector(`button[data-bs-target="#${tabId}"]`);
    if (triggerEl) {
        bootstrap.Tab.getOrCreateInstance(triggerEl).show();
    }
}

async function refreshDashboardData() {
    await Promise.all([
        loadDashboardSummary(),
        loadForecastData(),
        loadTelemetry(24),
        loadCoolingHistory(),
        loadModelBenchmarks(),
        loadAlertsFeed()
    ]);
}

/**
 * Fetch and Render Overview Summary KPIs
 */
async function loadDashboardSummary() {
    try {
        const res = await fetch('/api/dashboard/summary');
        const json = await res.json();
        if (!json.success) return;
        const d = json.data;

        document.getElementById('kpi-inlet-temp').textContent = `${d.current_inlet_temperature.toFixed(2)} °C`;
        document.getElementById('kpi-outlet-temp').textContent = `${d.current_outlet_temperature.toFixed(2)} °C`;
        document.getElementById('kpi-predicted-max').textContent = `${d.predicted_max_temperature.toFixed(2)} °C`;
        document.getElementById('kpi-predicted-avg').textContent = `${d.predicted_avg_temperature.toFixed(2)} °C`;
        
        document.getElementById('kpi-it-power').textContent = `${d.it_power_consumption.toFixed(1)} kW`;
        document.getElementById('kpi-cpu-usage').textContent = `${d.cpu_utilization.toFixed(1)} %`;
        document.getElementById('kpi-cooling-power').textContent = `${d.cooling_power_consumption.toFixed(1)} kW`;

        // Chiller status pill
        const pill = document.getElementById('kpi-chiller-status-pill');
        if (d.chiller_status === 'ON') {
            pill.className = 'status-pill status-on';
            pill.innerHTML = '<span class="status-dot dot-orange"></span><span>CHILLER ACTIVE (ON)</span>';
        } else {
            pill.className = 'status-pill status-off';
            pill.innerHTML = '<span class="status-dot dot-green"></span><span>CHILLER IDLE (OFF)</span>';
        }

        document.getElementById('kpi-activation-time').textContent = d.expected_activation_time || 'None in next 24h';
        document.getElementById('summary-cooling-hours').textContent = `${d.total_cooling_hours_next_24h} hrs / 24h`;
        document.getElementById('summary-on-threshold').textContent = `${d.on_threshold.toFixed(2)} °C`;
        document.getElementById('summary-off-threshold').textContent = `${d.off_threshold.toFixed(2)} °C`;

        // Sync input fields in cooling tab
        const inOn = document.getElementById('inputOnThreshold');
        const inOff = document.getElementById('inputOffThreshold');
        if (inOn && document.activeElement !== inOn) inOn.value = d.on_threshold;
        if (inOff && document.activeElement !== inOff) inOff.value = d.off_threshold;

        // Banner if temperature high
        const banner = document.getElementById('systemAlertBanner');
        if (d.predicted_max_temperature >= d.on_threshold) {
            document.getElementById('bannerText').textContent = `Thermal Alert: Projected temperature ${d.predicted_max_temperature.toFixed(2)}°C will trigger chiller activation at ${d.expected_activation_time || 'peak hours'}.`;
            banner.className = 'alert alert-warning py-2 px-3 small d-flex justify-content-between align-items-center mb-3';
            banner.classList.remove('d-none');
        } else {
            banner.classList.add('d-none');
        }
    } catch (err) {
        console.error('Error loading dashboard summary:', err);
    }
}

/**
 * Fetch and Render 24-Hour Forecast
 */
async function loadForecastData() {
    try {
        const res = await fetch('/api/dashboard/forecast');
        const json = await res.json();
        if (!json.success) return;

        const fc = json.forecast;
        const cooling = json.cooling_evaluation;
        const th = json.thresholds;

        renderOverviewForecastChart(fc.forecast_timestamps, fc.forecast_temperatures, th.on_threshold, th.off_threshold);
        renderDetailedForecastChart(fc.forecast_timestamps, fc.forecast_temperatures, th.on_threshold, th.off_threshold);

        document.getElementById('pred-peak-val').textContent = `${fc.max_temperature.toFixed(2)} °C`;
        document.getElementById('pred-min-val').textContent = `${fc.min_temperature.toFixed(2)} °C`;
        document.getElementById('pred-act-time').textContent = cooling.expected_activation_time || 'None Projected';
        document.getElementById('pred-runtime-val').textContent = `${cooling.total_on_hours} Hours`;
    } catch (err) {
        console.error('Error loading forecast:', err);
    }
}

/**
 * Fetch and Render Historical Telemetry
 */
async function loadTelemetry(hours = 24) {
    try {
        // Toggle active button style
        document.querySelectorAll('#tab-telemetry .btn-group button').forEach(b => b.classList.remove('active'));
        const activeBtn = Array.from(document.querySelectorAll('#tab-telemetry .btn-group button')).find(b => b.textContent.includes(`${hours}`));
        if (activeBtn) activeBtn.classList.add('active');

        const res = await fetch(`/api/dashboard/temperature?hours=${hours}`);
        const json = await res.json();
        if (!json.success) return;

        const data = json.data;
        renderTelemetrySensorsChart(data);
        renderTelemetryPowerChart(data);
        renderTelemetryCpuHumidityChart(data);
    } catch (err) {
        console.error('Error loading telemetry:', err);
    }
}

/**
 * Fetch Cooling Event History
 */
async function loadCoolingHistory() {
    try {
        const res = await fetch('/api/cooling/history?limit=30');
        const json = await res.json();
        if (!json.success) return;

        const tbody = document.getElementById('coolingEventsTableBody');
        if (!json.events || json.events.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-secondary py-3">No state transitions recorded yet.</td></tr>';
            return;
        }

        tbody.innerHTML = json.events.map(ev => `
            <tr>
                <td class="small">${ev.timestamp}</td>
                <td><span class="badge ${ev.previous_state === 'ON' ? 'bg-warning text-dark' : 'bg-secondary'}">${ev.previous_state}</span></td>
                <td><span class="badge ${ev.current_state === 'ON' ? 'bg-warning text-dark' : 'bg-success'}">${ev.current_state}</span></td>
                <td class="fw-semibold">${ev.trigger_temperature ? ev.trigger_temperature.toFixed(2) + ' °C' : '--'}</td>
                <td class="text-secondary small">${ev.threshold ? ev.threshold.toFixed(2) + ' °C' : '--'}</td>
            </tr>
        `).join('');
    } catch (err) {
        console.error('Error loading cooling events:', err);
    }
}

/**
 * Fetch Dataset Metadata & Preview Records
 */
async function loadDatasetMetadataAndPreview() {
    try {
        const infoRes = await fetch('/api/dataset/info');
        const infoJson = await infoRes.json();
        if (infoJson.success) {
            const meta = infoJson.data;
            document.getElementById('datasetMetaContainer').innerHTML = `
                <div class="mb-2"><span class="text-secondary">Total Records:</span> <strong class="text-dark">${meta.total_records.toLocaleString()}</strong></div>
                <div class="mb-2"><span class="text-secondary">Sampling Resolution:</span> <strong>${meta.sampling_interval_minutes} Minutes</strong></div>
                <div class="mb-2"><span class="text-secondary">Time Horizon:</span> <strong>${meta.start_time ? meta.start_time.split(' ')[0] : ''} to ${meta.end_time ? meta.end_time.split(' ')[0] : ''}</strong></div>
                <div class="mb-2"><span class="text-secondary">Duplicate Timestamps:</span> <strong class="text-success">${meta.duplicate_timestamps}</strong></div>
                <div class="mb-2"><span class="text-secondary">Dataset Type:</span> <span class="badge bg-info text-dark">Synthetic Simulation</span></div>
                <hr class="my-2">
                <div class="small text-secondary mb-1">Monitored Sensor Columns (10):</div>
                <div class="d-flex flex-wrap gap-1">
                    ${meta.columns.map(c => `<span class="badge bg-light text-secondary border">${c}</span>`).join('')}
                </div>
            `;
        }

        const prevRes = await fetch('/api/dataset/preview?limit=15');
        const prevJson = await prevRes.json();
        if (prevJson.success && prevJson.records.length > 0) {
            const cols = Object.keys(prevJson.records[0]);
            document.getElementById('previewTableHead').innerHTML = `<tr>${cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('')}</tr>`;
            document.getElementById('previewTableBody').innerHTML = prevJson.records.map(row => `
                <tr>${cols.map(c => `<td>${row[c]}</td>`).join('')}</tr>
            `).join('');
        }
    } catch (err) {
        console.error('Error loading dataset metadata:', err);
    }
}

/**
 * Fetch Model Benchmarks & Metrics
 */
async function loadModelBenchmarks() {
    try {
        const res = await fetch('/api/model/metrics');
        const json = await res.json();
        if (!json.success) return;

        const evalData = json.evaluation;
        const history = json.training_history;

        if (evalData && evalData.models) {
            const lstm = evalData.models.LSTM;
            if (lstm) {
                document.getElementById('metric-mae').textContent = `${lstm.mae.toFixed(4)} °C`;
                document.getElementById('metric-rmse').textContent = `${lstm.rmse.toFixed(4)} °C`;
                document.getElementById('metric-r2').textContent = `${lstm.r2.toFixed(4)}`;
            }

            renderModelComparisonChart(evalData.models);

            const tbody = document.getElementById('modelBenchmarkTableBody');
            tbody.innerHTML = Object.entries(evalData.models).map(([k, m]) => `
                <tr ${k === 'LSTM' ? 'class="table-primary fw-semibold"' : ''}>
                    <td>${m.name}</td>
                    <td>${m.mae.toFixed(4)}</td>
                    <td>${m.rmse.toFixed(4)}</td>
                    <td>${m.r2.toFixed(4)}</td>
                </tr>
            `).join('');
        }

        if (history) {
            renderTrainingLossChart(history);
        }

        if (evalData && evalData.sample_forecast) {
            renderActualVsPredictedChart(evalData.sample_forecast);
        }
    } catch (err) {
        console.error('Error loading model metrics:', err);
    }
}

/**
 * Fetch and Render Alerts Feed
 */
async function loadAlertsFeed() {
    try {
        const res = await fetch('/api/dashboard/alerts?limit=25');
        const json = await res.json();
        if (!json.success) return;

        const alerts = json.alerts || [];
        document.getElementById('activeAlertsBadge').textContent = alerts.length;
        document.getElementById('totalAlertsCount').textContent = `${alerts.length} Events`;

        const renderAlertHtml = (a) => `
            <div class="alert-item alert-${a.severity}">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <strong class="small">${a.title}</strong>
                    <span class="text-secondary" style="font-size: 0.72rem;">${a.timestamp}</span>
                </div>
                <div class="small text-secondary">${a.message}</div>
            </div>
        `;

        const ovContainer = document.getElementById('overviewAlertsContainer');
        const fullContainer = document.getElementById('fullAlertsContainer');

        if (alerts.length === 0) {
            const emptyMsg = '<div class="text-secondary small text-center py-3">No active incident alerts registered.</div>';
            ovContainer.innerHTML = emptyMsg;
            fullContainer.innerHTML = emptyMsg;
            return;
        }

        ovContainer.innerHTML = alerts.slice(0, 4).map(renderAlertHtml).join('');
        fullContainer.innerHTML = alerts.map(renderAlertHtml).join('');
    } catch (err) {
        console.error('Error loading alerts:', err);
    }
}

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
    // Threshold form submit
    const thForm = document.getElementById('thresholdForm');
    if (thForm) {
        thForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const feedback = document.getElementById('thresholdFeedback');
            feedback.classList.add('d-none');

            const on_threshold = parseFloat(document.getElementById('inputOnThreshold').value);
            const off_threshold = parseFloat(document.getElementById('inputOffThreshold').value);

            const token = localStorage.getItem('auth_token') || 'demo-token-admin';
            try {
                const res = await fetch('/api/cooling/thresholds', {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({ on_threshold, off_threshold })
                });
                const data = await res.json();
                if (data.success) {
                    feedback.className = 'alert alert-success py-2 small mb-3';
                    feedback.textContent = data.message;
                    feedback.classList.remove('d-none');
                    refreshDashboardData();
                } else {
                    feedback.className = 'alert alert-danger py-2 small mb-3';
                    feedback.textContent = data.error || 'Failed to update thresholds.';
                    feedback.classList.remove('d-none');
                }
            } catch (err) {
                feedback.className = 'alert alert-danger py-2 small mb-3';
                feedback.textContent = 'Server error: ' + err.message;
                feedback.classList.remove('d-none');
            }
        });
    }

    // Dataset upload form
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const feedback = document.getElementById('uploadFeedback');
            feedback.classList.add('d-none');

            const fileInput = document.getElementById('datasetFileInput');
            if (!fileInput.files || fileInput.files.length === 0) return;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const token = localStorage.getItem('auth_token') || 'demo-token-admin';
            try {
                feedback.className = 'alert alert-info py-2 small mb-3';
                feedback.textContent = 'Uploading and validating dataset...';
                feedback.classList.remove('d-none');

                const res = await fetch('/api/dataset/upload', {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}` },
                    body: formData
                });
                const data = await res.json();
                if (data.success) {
                    feedback.className = 'alert alert-success py-2 small mb-3';
                    feedback.textContent = 'Dataset uploaded and validated successfully!';
                    loadDatasetMetadataAndPreview();
                } else {
                    feedback.className = 'alert alert-danger py-2 small mb-3';
                    feedback.textContent = data.error || 'Upload failed.';
                }
            } catch (err) {
                feedback.className = 'alert alert-danger py-2 small mb-3';
                feedback.textContent = 'Upload error: ' + err.message;
            }
        });
    }
}
