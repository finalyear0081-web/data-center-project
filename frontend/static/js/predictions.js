/**
 * Detailed 24-Hour Prediction Chart and Paginated Table Controller
 */

let allPredictionRecords = [];
let filteredRecords = [];
let currentPage = 1;
const pageSize = 25;

document.addEventListener('DOMContentLoaded', () => {
    loadPredictionData();
    setupTableControls();
});

async function loadPredictionData() {
    try {
        const res = await fetch('/api/prediction/results');
        const json = await res.json();
        if (!json.success || !json.data || !json.data.prediction) return;

        const pred = json.data.prediction;
        const cooling = json.data.cooling || {};

        const horizonHours = pred.forecast_horizon_hours || 24;
        const durLabel = document.getElementById('coolingDurationLabel');
        if (durLabel) durLabel.textContent = `COOLING DURATION (${horizonHours}H)`;

        document.getElementById('forecastPeakVal').textContent = `${pred.max_temperature.toFixed(2)} °C`;
        document.getElementById('forecastMinVal').textContent = `${pred.min_temperature.toFixed(2)} °C`;
        document.getElementById('forecastActivationTime').textContent = cooling.expected_activation_time || `None in ${horizonHours}h`;
        document.getElementById('forecastDurationVal').textContent = `${cooling.predicted_running_hours || 0} Hours`;

        const onTh = cooling.on_threshold != null ? Number(cooling.on_threshold) : 28.5;
        const offTh = cooling.off_threshold != null ? Number(cooling.off_threshold) : 26.5;

        const bOn = document.getElementById('badgeOnThreshold');
        const bOff = document.getElementById('badgeOffThreshold');
        if (bOn) bOn.textContent = `${onTh.toFixed(1)}°C ON Threshold`;
        if (bOff) bOff.textContent = `${offTh.toFixed(1)}°C OFF Threshold`;

        renderDetailedForecastChart(
            pred.forecast_timestamps,
            pred.forecast_temperatures,
            onTh,
            offTh
        );

        // Prioritize 15-minute presentation forecast (96 records)
        const raw15 = json.data.forecast_15min || pred.forecast_15min;
        if (Array.isArray(raw15) && raw15.length === 96) {
            allPredictionRecords = raw15.map((r, i) => ({
                index: r.serial_number || r.index || (i + 1),
                timestamp: r.forecast_timestamp || r.timestamp,
                temperature: Number(r.predicted_inlet_temperature_celsius != null ? r.predicted_inlet_temperature_celsius : r.temperature),
                chiller_state: r.chiller_state || (r.is_running ? 'ON' : 'OFF'),
                condition: r.temperature_status || r.condition || '--'
            }));
        } else {
            // Resample client-side to 96 records (15-min intervals)
            const temps = pred.forecast_temperatures || [];
            const times = pred.forecast_timestamps || [];
            const numRecords = 96;
            let simState = 'OFF';

            allPredictionRecords = [];
            for (let k = 0; k < numRecords; k++) {
                const idx = 7.5 * k;
                const iLow = Math.floor(idx);
                const iHigh = Math.min(iLow + 1, temps.length - 1);
                const alpha = idx - iLow;

                let tempK = 24.0;
                if (temps.length > 0) {
                    if (iLow >= temps.length) {
                        tempK = Number(temps[temps.length - 1]);
                    } else if (iLow === iHigh || alpha === 0) {
                        tempK = Number(temps[iLow]);
                    } else {
                        tempK = (1.0 - alpha) * Number(temps[iLow]) + alpha * Number(temps[iHigh]);
                    }
                }
                tempK = Math.round(tempK * 100) / 100;

                const tsK = (times.length > 0) ? (times[Math.min(Math.round(idx), times.length - 1)] || `+${k * 15}m`) : `+${k * 15}m`;

                let cond = 'NORMAL / RETAIN CURRENT STATE';
                if (tempK >= onTh) {
                    simState = 'ON';
                    cond = 'HIGH TEMPERATURE';
                } else if (tempK <= offTh) {
                    simState = 'OFF';
                    cond = 'NORMAL / COOLING NOT REQUIRED';
                }

                allPredictionRecords.push({
                    index: k + 1,
                    timestamp: tsK,
                    temperature: tempK,
                    chiller_state: simState,
                    condition: cond
                });
            }
        }

        filteredRecords = [...allPredictionRecords];
        document.getElementById('totalTableRecords').textContent = `${allPredictionRecords.length} Records (15-min intervals)`;
        currentPage = 1;
        renderTablePage();
    } catch (err) {
        console.error('Error loading prediction details:', err);
    }
}

function setupTableControls() {
    const searchInput = document.getElementById('tableSearchInput');
    const sortSelect = document.getElementById('sortSelect');

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            filteredRecords = allPredictionRecords.filter(r => r.timestamp.toLowerCase().includes(query));
            currentPage = 1;
            renderTablePage();
        });
    }

    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            const val = e.target.value;
            if (val === 'time_asc') {
                filteredRecords.sort((a, b) => a.index - b.index);
            } else if (val === 'time_desc') {
                filteredRecords.sort((a, b) => b.index - a.index);
            } else if (val === 'temp_desc') {
                filteredRecords.sort((a, b) => b.temperature - a.temperature);
            } else if (val === 'temp_asc') {
                filteredRecords.sort((a, b) => a.temperature - b.temperature);
            }
            currentPage = 1;
            renderTablePage();
        });
    }
}

function renderTablePage() {
    const tbody = document.getElementById('predictionsTableBody');
    const total = filteredRecords.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));

    if (currentPage > totalPages) currentPage = totalPages;

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, total);
    const pageRecords = filteredRecords.slice(startIdx, endIdx);

    if (pageRecords.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">No matching forecast records found.</td></tr>';
        document.getElementById('paginationInfo').textContent = 'Showing 0 records';
        document.getElementById('paginationButtons').innerHTML = '';
        return;
    }

    tbody.innerHTML = pageRecords.map(r => {
        let condBadge = '<span class="badge bg-info text-dark">NORMAL / RETAIN CURRENT STATE</span>';
        let rowClass = '';
        if (r.condition === 'HIGH TEMPERATURE') {
            condBadge = '<span class="badge bg-danger">HIGH TEMPERATURE</span>';
            rowClass = 'table-danger';
        } else if (r.condition && r.condition.includes('COOLING NOT REQUIRED')) {
            condBadge = '<span class="badge bg-success">NORMAL / COOLING NOT REQUIRED</span>';
        }

        const isRunning = (r.chiller_state === 'ON' || r.chiller_state === 'RUNNING');
        const chillerBadge = isRunning
            ? '<span class="status-pill-lg running" style="font-size: 0.75rem;"><span class="dot-indicator green"></span>ON</span>'
            : '<span class="status-pill-lg stopped" style="font-size: 0.75rem;"><span class="dot-indicator gray"></span>OFF</span>';

        return `
            <tr class="${rowClass}">
                <td class="text-muted small">${r.index}</td>
                <td class="fw-semibold">${r.timestamp}</td>
                <td><strong>${r.temperature.toFixed(2)} °C</strong></td>
                <td>${chillerBadge}</td>
                <td>${condBadge}</td>
            </tr>
        `;
    }).join('');

    document.getElementById('paginationInfo').textContent = `Showing ${startIdx + 1} to ${endIdx} of ${total} records`;

    // Render pagination buttons
    const btnContainer = document.getElementById('paginationButtons');
    let btnsHtml = `
        <button class="btn btn-outline-secondary" onclick="changePage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>&larr; Prev</button>
    `;

    for (let p = 1; p <= totalPages; p++) {
        if (p === 1 || p === totalPages || (p >= currentPage - 2 && p <= currentPage + 2)) {
            btnsHtml += `
                <button class="btn ${p === currentPage ? 'btn-primary' : 'btn-outline-secondary'}" onclick="changePage(${p})">${p}</button>
            `;
        } else if (p === currentPage - 3 || p === currentPage + 3) {
            btnsHtml += `<span class="btn btn-outline-secondary disabled">...</span>`;
        }
    }

    btnsHtml += `
        <button class="btn btn-outline-secondary" onclick="changePage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>Next &rarr;</button>
    `;

    btnContainer.innerHTML = btnsHtml;
}

function changePage(newPage) {
    currentPage = newPage;
    renderTablePage();
}

async function runLiveInference() {
    try {
        const res = await fetch('/api/prediction/forecast', { method: 'POST' });
        const json = await res.json();
        if (json.success) {
            loadPredictionData();
            alert('Live 24-hour forecast updated successfully!');
        } else {
            alert('Prediction error: ' + json.error);
        }
    } catch (err) {
        alert('Server connection error: ' + err.message);
    }
}
