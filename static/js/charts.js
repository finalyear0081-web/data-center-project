/**
 * Chart.js Visualization Module for Data Center Monitoring
 * Manages clean, professional charts for forecasts, sensor telemetry, and benchmarks.
 */

const chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

/**
 * Render Overview 24h Temperature Forecast Chart
 */
function renderOverviewForecastChart(timestamps, temperatures, onThreshold = 28.0, offThreshold = 26.0) {
    const ctx = document.getElementById('overviewForecastChart');
    if (!ctx) return;
    destroyChart('overviewForecastChart');

    // Downsample labels for readability (every 12th label = 1 hour)
    const formattedLabels = timestamps.map((ts, idx) => {
        if (idx % 24 === 0) {
            const parts = ts.split(' ');
            return parts.length > 1 ? parts[1].substring(0, 5) : ts;
        }
        return '';
    });

    chartInstances['overviewForecastChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    label: 'Predicted Inlet Temp (°C)',
                    data: temperatures,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.35,
                    fill: true
                },
                {
                    label: `Chiller ON (${onThreshold}°C)`,
                    data: Array(temperatures.length).fill(onThreshold),
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: `Chiller OFF (${offThreshold}°C)`,
                    data: Array(temperatures.length).fill(offThreshold),
                    borderColor: '#10b981',
                    borderWidth: 1.5,
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
                legend: { position: 'top', labels: { boxWidth: 14, font: { size: 12 } } },
                tooltip: {
                    callbacks: {
                        title: (items) => timestamps[items[0].dataIndex] || '',
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} °C`
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 0 } },
                y: {
                    title: { display: true, text: 'Temperature (°C)' },
                    suggestedMin: 20,
                    suggestedMax: 31,
                    grid: { color: '#f1f5f9' }
                }
            }
        }
    });
}

/**
 * Render Detailed 24h Forecast Chart with threshold zones
 */
function renderDetailedForecastChart(timestamps, temperatures, onThreshold = 28.0, offThreshold = 26.0) {
    const ctx = document.getElementById('detailedForecastChart');
    if (!ctx) return;
    destroyChart('detailedForecastChart');

    const formattedLabels = timestamps.map((ts, idx) => {
        if (idx % 12 === 0) {
            const parts = ts.split(' ');
            return parts.length > 1 ? parts[1].substring(0, 5) : ts;
        }
        return '';
    });

    chartInstances['detailedForecastChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    label: 'Predicted Server Inlet (°C)',
                    data: temperatures,
                    borderColor: '#0284c7',
                    backgroundColor: 'rgba(2, 132, 199, 0.1)',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: `Chiller Activation Limit (${onThreshold}°C)`,
                    data: Array(temperatures.length).fill(onThreshold),
                    borderColor: '#dc2626',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: `Chiller Deactivation (${offThreshold}°C)`,
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
                x: { grid: { display: false }, ticks: { autoSkip: false } },
                y: {
                    title: { display: true, text: 'Inlet Temperature (°C)' },
                    suggestedMin: 19,
                    suggestedMax: 32,
                    grid: { color: '#f1f5f9' }
                }
            }
        }
    });
}

/**
 * Render Historical Sensor Telemetry Charts
 */
function renderTelemetrySensorsChart(telemetry) {
    const ctx = document.getElementById('telemetrySensorsChart');
    if (!ctx) return;
    destroyChart('telemetrySensorsChart');

    const step = Math.max(1, Math.floor(telemetry.timestamps.length / 15));
    const formattedLabels = telemetry.timestamps.map((ts, idx) => {
        if (idx % step === 0) {
            const parts = ts.split(' ');
            return parts.length > 1 ? parts[1].substring(0, 5) : ts;
        }
        return '';
    });

    chartInstances['telemetrySensorsChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: formattedLabels,
            datasets: [
                { label: 'Inlet Temp', data: telemetry.inlet_temperature, borderColor: '#2563eb', borderWidth: 2, pointRadius: 0 },
                { label: 'Outlet Temp', data: telemetry.outlet_temperature, borderColor: '#dc2626', borderWidth: 1.5, pointRadius: 0 },
                { label: 'Cold Aisle', data: telemetry.cold_aisle_temperature, borderColor: '#059669', borderWidth: 1.5, pointRadius: 0 },
                { label: 'Hot Aisle', data: telemetry.hot_aisle_temperature, borderColor: '#d97706', borderWidth: 1.5, pointRadius: 0 },
                { label: 'Outdoor Temp', data: telemetry.outdoor_temperature, borderColor: '#6b7280', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { title: { display: true, text: 'Temperature (°C)' }, grid: { color: '#f1f5f9' } },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderTelemetryPowerChart(telemetry) {
    const ctx = document.getElementById('telemetryPowerChart');
    if (!ctx) return;
    destroyChart('telemetryPowerChart');

    chartInstances['telemetryPowerChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: telemetry.timestamps.map((_, i) => (i % 24 === 0 ? '' : '')),
            datasets: [
                { label: 'IT Power (kW)', data: telemetry.it_power_consumption, borderColor: '#4f46e5', borderWidth: 2, pointRadius: 0 },
                { label: 'Cooling Power (kW)', data: telemetry.cooling_power_consumption, borderColor: '#0891b2', borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { title: { display: true, text: 'Power (kW)' }, grid: { color: '#f1f5f9' } },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderTelemetryCpuHumidityChart(telemetry) {
    const ctx = document.getElementById('telemetryCpuHumidityChart');
    if (!ctx) return;
    destroyChart('telemetryCpuHumidityChart');

    chartInstances['telemetryCpuHumidityChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: telemetry.timestamps.map((_, i) => (i % 24 === 0 ? '' : '')),
            datasets: [
                { label: 'CPU Usage (%)', data: telemetry.cpu_usage, borderColor: '#ea580c', borderWidth: 2, pointRadius: 0 },
                { label: 'Humidity (%)', data: telemetry.humidity, borderColor: '#0284c7', borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { title: { display: true, text: 'Percentage (%)' }, suggestedMin: 10, suggestedMax: 100, grid: { color: '#f1f5f9' } },
                x: { grid: { display: false } }
            }
        }
    });
}

/**
 * Render Model Benchmarks & Comparison Charts
 */
function renderModelComparisonChart(modelsData) {
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx || !modelsData) return;
    destroyChart('modelComparisonChart');

    const labels = [];
    const maeList = [];
    const rmseList = [];

    for (const [key, item] of Object.entries(modelsData)) {
        labels.push(item.name || key);
        maeList.push(item.mae);
        rmseList.push(item.rmse);
    }

    chartInstances['modelComparisonChart'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'MAE (°C)', data: maeList, backgroundColor: '#3b82f6' },
                { label: 'RMSE (°C)', data: rmseList, backgroundColor: '#93c5fd' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { title: { display: true, text: 'Error in °C (Lower is Better)' }, beginAtZero: true, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderTrainingLossChart(history) {
    const ctx = document.getElementById('trainingLossChart');
    if (!ctx || !history || !history.loss) return;
    destroyChart('trainingLossChart');

    const epochs = history.loss.map((_, i) => `Epoch ${i + 1}`);

    chartInstances['trainingLossChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Training Loss (MSE)', data: history.loss, borderColor: '#2563eb', borderWidth: 2, pointRadius: 3 },
                { label: 'Validation Loss (MSE)', data: history.val_loss, borderColor: '#f59e0b', borderWidth: 2, pointRadius: 3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { title: { display: true, text: 'Mean Squared Error' }, beginAtZero: true, grid: { color: '#f1f5f9' } }
            }
        }
    });
}

function renderActualVsPredictedChart(sampleData) {
    const ctx = document.getElementById('actualVsPredictedChart');
    if (!ctx || !sampleData || !sampleData.actual) return;
    destroyChart('actualVsPredictedChart');

    const labels = sampleData.actual.map((_, i) => (i % 24 === 0 ? `+${i * 5 / 60}h` : ''));

    chartInstances['actualVsPredictedChart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Actual Inlet (°C)', data: sampleData.actual, borderColor: '#10b981', borderWidth: 2, pointRadius: 0 },
                { label: 'LSTM Forecast (°C)', data: sampleData.lstm_pred, borderColor: '#2563eb', borderDash: [4, 4], borderWidth: 2, pointRadius: 0 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: { title: { display: true, text: 'Temp (°C)' }, grid: { color: '#f1f5f9' } }
            }
        }
    });
}
