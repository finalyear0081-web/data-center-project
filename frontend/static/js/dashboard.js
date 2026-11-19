/**
 * Overview Dashboard Controller
 * Multi-schema dynamic rendering for HazardNet and AAU Experimental Data
 */

document.addEventListener('DOMContentLoaded', () => {
    const initialSvg = document.getElementById('overviewChillerFanSvg');
    if (initialSvg && initialSvg.pauseAnimations) {
        try { initialSvg.pauseAnimations(); } catch (e) {}
    }
    loadDashboardOverview();
    setInterval(loadDashboardOverview, 25000);
});

async function refreshDashboard() {
    await loadDashboardOverview();
}

async function loadDashboardOverview() {
    try {
        const [sumRes, fcRes, telemRes, alertRes] = await Promise.all([
            fetch('/api/dashboard/summary'),
            fetch('/api/dashboard/forecast'),
            fetch('/api/dashboard/temperature?hours=24'),
            fetch('/api/dashboard/alerts?limit=6')
        ]);

        const sumJson = await sumRes.json();
        const fcJson = await fcRes.json();
        const telemJson = await telemRes.json();
        const alertJson = await alertRes.json();

        // 1. Update KPI Cards & Ribbon
        if (sumJson.success && sumJson.data) {
            const d = sumJson.data;
            const isAAU = (d.schema_type === 'AAU_EXPERIMENTAL');

            // Dynamic Metadata Ribbon
            const badge = document.getElementById('ribbonDatasetBadge');
            const title = document.getElementById('ribbonDatasetTitle');
            const details = document.getElementById('ribbonDetails');

            const horizonHours = d.forecast_horizon_hours || 24;
            const horizonSteps = d.forecast_horizon_steps || (d.schema_type === 'AAU_EXPERIMENTAL' ? 30 : 720);
            const intervalMins = d.sampling_interval_minutes || 2;
            const srcName = d.dataset_source || 'Active Telemetry Dataset';
            const schemaLabel = d.schema_name || (isAAU ? 'AAU Experimental Benchmark' : 'Data Center Telemetry');

            if (badge) {
                if (isAAU) {
                    badge.className = 'badge bg-primary text-white';
                    badge.textContent = 'EXPERIMENTAL BENCHMARK';
                } else if (srcName.toLowerCase().includes('synthetic')) {
                    badge.className = 'badge bg-info text-dark';
                    badge.textContent = 'SYNTHETIC BENCHMARK';
                } else if (srcName.toLowerCase().includes('hazardnet')) {
                    badge.className = 'badge bg-success text-white';
                    badge.textContent = 'HAZARDNET HPC';
                } else {
                    badge.className = 'badge bg-success text-white';
                    badge.textContent = 'OPERATIONAL TELEMETRY';
                }
            }

            if (title) {
                title.textContent = `${schemaLabel} — ${srcName}`;
            }

            if (details) {
                const nodeStr = d.node_id ? `Node / Exp: ${d.node_id}` : 'Cluster Mean Aggregate';
                const timeStr = d.last_reading_time ? `Last: ${d.last_reading_time}` : 'Live Telemetry';
                details.innerHTML = `
                    <span><i data-lucide="server" class="me-1" style="width: 13px; height: 13px;"></i>${nodeStr}</span>
                    <span>&bull;</span>
                    <span><i data-lucide="clock" class="me-1" style="width: 13px; height: 13px;"></i>${intervalMins}-Min Sampling (${horizonSteps} Steps / ${horizonHours}h)</span>
                    <span>&bull;</span>
                    <span><i data-lucide="calendar" class="me-1" style="width: 13px; height: 13px;"></i>${timeStr}</span>
                `;
            }

            document.getElementById('kpi-peak-label').textContent = `Predicted Peak Inlet (${horizonHours}h Horizon)`;
            const horizonBadge = document.getElementById('forecastHorizonBadge');
            if (horizonBadge) horizonBadge.textContent = `${horizonHours}-Hour Forecast (${horizonSteps} Steps)`;

            document.getElementById('kpi-inlet-temp').textContent = (d.current_inlet_temperature != null) ? `${d.current_inlet_temperature.toFixed(2)} °C` : '-- °C';
            document.getElementById('kpi-outlet-temp').textContent = (d.current_outlet_temperature != null) ? `${d.current_outlet_temperature.toFixed(2)} °C` : '-- °C';
            
            // Temperature Status Badge
            const statusBadge = document.getElementById('kpi-inlet-status-badge');
            if (statusBadge) {
                const statusText = d.temperature_status || (d.current_inlet_temperature >= (d.on_threshold || 28.0) ? 'HIGH TEMPERATURE' : (d.current_inlet_temperature <= (d.off_threshold || 26.0) ? 'NORMAL / COOLING NOT REQUIRED' : 'NORMAL / RETAIN CURRENT STATE'));
                statusBadge.textContent = statusText;
                if (statusText === 'HIGH TEMPERATURE') {
                    statusBadge.className = 'badge bg-danger';
                } else if (statusText.includes('COOLING NOT REQUIRED')) {
                    statusBadge.className = 'badge bg-success';
                } else {
                    statusBadge.className = 'badge bg-info text-dark';
                }
            }
            
            if (d.current_outlet_temperature != null && d.current_inlet_temperature != null) {
                document.getElementById('kpi-delta-temp').textContent = `${(d.current_outlet_temperature - d.current_inlet_temperature).toFixed(2)} °C`;
            } else {
                document.getElementById('kpi-delta-temp').textContent = '-- °C';
            }

            document.getElementById('kpi-predicted-max').textContent = (d.predicted_max_temperature != null) ? `${d.predicted_max_temperature.toFixed(2)} °C` : '-- °C';
            document.getElementById('kpi-predicted-avg').textContent = (d.predicted_avg_temperature != null) ? `${d.predicted_avg_temperature.toFixed(2)} °C` : '-- °C';
            
            document.getElementById('kpi-it-power').textContent = (d.it_power_consumption != null) ? `${d.it_power_consumption.toFixed(2)} kW` : '-- kW';
            if (isAAU && d.facility_power != null) {
                document.getElementById('kpi-cpu-usage').textContent = `${d.facility_power.toFixed(2)} kW`;
            } else if (d.cpu_utilization != null) {
                document.getElementById('kpi-cpu-usage').textContent = `${d.cpu_utilization.toFixed(1)} %`;
            } else {
                document.getElementById('kpi-cpu-usage').textContent = 'N/A';
            }

            // Chiller state in KPI
            const pill = document.getElementById('kpi-chiller-status');
            const text = document.getElementById('kpi-chiller-text');
            const mode = document.getElementById('kpi-cooling-mode');
            const cabinet = document.getElementById('overviewChillerCabinet');
            const chTitle = document.getElementById('overviewChillerTitle');

            const isRunning = Boolean(d.is_running || d.chiller_status === 'RUNNING' || d.chiller_status === 'ON' || d.chiller_state === 'ON' || d.chiller_state === 'RUNNING');
            const fanSvg = document.getElementById('overviewChillerFanSvg') || (cabinet ? cabinet.querySelector('svg') : null);

            if (isRunning) {
                pill.className = 'status-pill-lg running';
                const dot = pill.querySelector('.dot-indicator');
                if (dot) dot.className = 'dot-indicator green';
                text.textContent = 'ON';
                if (chTitle) chTitle.textContent = 'CHILLER ACTIVE (ON)';
                if (cabinet) cabinet.className = 'chiller-cabinet running';

                if (fanSvg && fanSvg.unpauseAnimations) {
                    try { fanSvg.unpauseAnimations(); } catch (e) {}
                }
            } else {
                pill.className = 'status-pill-lg stopped';
                const dot = pill.querySelector('.dot-indicator');
                if (dot) dot.className = 'dot-indicator gray';
                text.textContent = 'OFF';
                if (chTitle) chTitle.textContent = 'CHILLER STOPPED (OFF)';
                if (cabinet) cabinet.className = 'chiller-cabinet';

                if (fanSvg && fanSvg.pauseAnimations) {
                    try { fanSvg.pauseAnimations(); } catch (e) {}
                }
            }
            window._lastOverviewRunning = isRunning;

            mode.textContent = d.operating_mode;
            document.getElementById('kpi-activation-time').textContent = d.expected_activation_time || 'None in forecast';
            const onThDisplay = (d.on_threshold != null ? d.on_threshold : 28.5);
            const offThDisplay = (d.off_threshold != null ? d.off_threshold : 26.5);
            document.getElementById('overviewOnTh').textContent = `${onThDisplay.toFixed(1)}°C`;
            document.getElementById('overviewOffTh').textContent = `${offThDisplay.toFixed(1)}°C`;

            // Thermal Alert Banner
            const banner = document.getElementById('dashboardAlertBanner');
            if (d.predicted_max_temperature >= onThDisplay) {
                document.getElementById('dashboardAlertBannerText').textContent = 
                    `Thermal Excursion: Peak temperature ${d.predicted_max_temperature.toFixed(2)}°C reaches/exceeds cooling activation threshold (${onThDisplay.toFixed(1)}°C). Chiller status: RUNNING.`;
                banner.classList.remove('d-none');
            } else {
                banner.classList.add('d-none');
            }
        }

        // 2. Render Overview Forecast Chart
        if (fcJson.success && fcJson.data && fcJson.data.prediction) {
            const pred = fcJson.data.prediction;
            const cooling = fcJson.data.cooling || {};
            const chartOnTh = cooling.on_threshold != null ? cooling.on_threshold : 28.5;
            const chartOffTh = cooling.off_threshold != null ? cooling.off_threshold : 26.5;
            renderOverviewForecastChart(
                pred.forecast_timestamps,
                pred.forecast_temperatures,
                chartOnTh,
                chartOffTh
            );

            if (document.getElementById('overviewChillerReason')) {
                document.getElementById('overviewChillerReason').textContent = cooling.activation_reason || 'Cooling state stabilized.';
            }
        }

        // 3. Render Overview Telemetry Chart
        if (telemJson.success && telemJson.data) {
            renderOverviewTelemetryChart(telemJson.data);
            const badge = document.getElementById('telemetryHistoryBadge');
            if (badge) {
                badge.textContent = `${telemJson.data.total_points || 0} Telemetry Points`;
            }
        }

        if (window.lucide) {
            lucide.createIcons();
        }
    } catch (err) {
        console.error('Error loading dashboard overview:', err);
    }
}
