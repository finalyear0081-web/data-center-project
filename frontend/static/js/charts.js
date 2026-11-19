/**
 * Chart.js Visualizations for Forecast, Telemetry, and Evaluation
 */

const activeChartInstances = {};

function clearChart(id) {
    if (activeChartInstances[id]) {
        activeChartInstances[id].destroy();
        delete activeChartInstances[id];
    }
}

function renderOverviewForecastChart(timestamps, temperatures, onThreshold = 28.5, offThreshold = 26.5) {
    const canvas = document.getElementById('overviewForecastChart');
    if (!canvas) return;
    clearChart('overviewForecastChart');

    const labels = timestamps.map((ts, idx) => {
        if (idx % 24 === 0) {
            const parts = ts.split(' ');
            return parts.length > 2 ? `${parts[1]} ${parts[2]}` : (parts.length > 1 ? parts[1].substring(0, 5) : ts);
        }
        return '';
    });

    activeChartInstances['overviewForecastChart'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Inlet (°C)',
                    data: temperatures,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.35,
                    fill: true
                },
                {
                    label: `ON Threshold (${onThreshold}°C)`,
                    data: Array(temperatures.length).fill(onThreshold),
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [5, 4],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: `OFF Threshold (${offThreshold}°C)`,
                    data: Array(temperatures.length).fill(offThreshold),
                    borderColor: '#10b981',
                    borderWidth: 1.5,
                    borderDash: [5, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        title: (items) => timestamps[items[0].dataIndex] || '',
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} °C`
                    }
                }
            },
            scales: {
                x: { grid: { display: false } },
                y: { title: { display: true, text: 'Temperature (°C)' }, suggestedMin: 22, suggestedMax: 40, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderOverviewTelemetryChart(telemetry) {
    const canvas = document.getElementById('overviewTelemetryChart');
    if (!canvas || !telemetry || !telemetry.timestamps) return;
    clearChart('overviewTelemetryChart');

    const step = Math.max(1, Math.floor(telemetry.timestamps.length / 10));
    const labels = telemetry.timestamps.map((ts, idx) => {
        if (idx % step === 0) {
            const parts = ts.split(' ');
            return parts.length > 2 ? `${parts[1]} ${parts[2]}` : (parts.length > 1 ? parts[1].substring(0, 5) : ts);
        }
        return '';
    });

    const datasets = [];
    if (telemetry.inlet_temperature && telemetry.inlet_temperature.length) {
        datasets.push({ label: 'Inlet Temp (°C)', data: telemetry.inlet_temperature, borderColor: '#2563eb', borderWidth: 2, pointRadius: 0 });
    }
    if (telemetry.outlet_temperature && telemetry.outlet_temperature.length) {
        datasets.push({ label: 'Outlet Temp (°C)', data: telemetry.outlet_temperature, borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0 });
    }
    if (telemetry.cold_aisle_temperature && telemetry.cold_aisle_temperature.length) {
        datasets.push({ label: 'Cold Aisle (°C)', data: telemetry.cold_aisle_temperature, borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 });
    }

    activeChartInstances['overviewTelemetryChart'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
            scales: {
                x: { grid: { display: false } },
                y: { title: { display: true, text: 'Temperature (°C)' }, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderDetailedForecastChart(timestamps, temperatures, onThreshold = 28.5, offThreshold = 26.5) {
    const canvas = document.getElementById('detailedForecastChart');
    if (!canvas) return;
    clearChart('detailedForecastChart');

    const labels = timestamps.map((ts, idx) => {
        if (idx % 12 === 0) {
            const parts = ts.split(' ');
            return parts.length > 2 ? `${parts[1]} ${parts[2]}` : (parts.length > 1 ? parts[1].substring(0, 5) : ts);
        }
        return '';
    });

    activeChartInstances['detailedForecastChart'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted Server Inlet (°C)',
                    data: temperatures,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 2.5,
                    borderDash: [4, 2],
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: `Chiller ON Threshold (${onThreshold}°C)`,
                    data: Array(temperatures.length).fill(onThreshold),
                    borderColor: '#dc2626',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: `Chiller OFF Threshold (${offThreshold}°C)`,
                    data: Array(temperatures.length).fill(offThreshold),
                    borderColor: '#16a34a',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        title: (items) => timestamps[items[0].dataIndex] || '',
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} °C`
                    }
                }
            },
            scales: {
                x: { grid: { display: false } },
                y: { title: { display: true, text: 'Inlet Temperature (°C)' }, suggestedMin: 22, suggestedMax: 40, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderTrainingLossChart(history) {
    const canvas = document.getElementById('trainingLossChart');
    if (!canvas || !history || !history.loss) return;
    clearChart('trainingLossChart');

    activeChartInstances['trainingLossChart'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: history.loss.map((_, i) => `Epoch ${i + 1}`),
            datasets: [
                { label: 'Train Loss (MSE)', data: history.loss, borderColor: '#2563eb', borderWidth: 2, pointRadius: 3 },
                { label: 'Val Loss (MSE)', data: history.val_loss, borderColor: '#f59e0b', borderWidth: 2, pointRadius: 3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { grid: { display: false } },
                y: { title: { display: true, text: 'MSE Loss' }, beginAtZero: true, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderActualVsPredictedChart(sampleData) {
    const canvas = document.getElementById('actualVsPredictedChart');
    if (!canvas || !sampleData || !sampleData.actual) return;
    clearChart('actualVsPredictedChart');

    const labels = sampleData.actual.map((_, i) => (i % 24 === 0 ? `+${(i * 5 / 60).toFixed(0)}h` : ''));

    activeChartInstances['actualVsPredictedChart'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Actual Inlet (°C)', data: sampleData.actual, borderColor: '#10b981', borderWidth: 2, pointRadius: 0 },
                { label: 'LSTM Prediction (°C)', data: sampleData.lstm_pred, borderColor: '#2563eb', borderDash: [4, 4], borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { grid: { display: false } },
                y: { title: { display: true, text: 'Temperature (°C)' }, grid: { color: '#f1f5f9' } }
            }
        }
    });
}
