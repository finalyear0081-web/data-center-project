/**
 * Model Training & Live Pipeline Monitor Controller
 * Handles model training invocation, live status polling, and loss curve plotting.
 */

let lossChart = null;
let trainingPollTimer = null;

document.addEventListener("DOMContentLoaded", function () {
    initLossChart();
    loadCurrentTrainingStatus();
    loadLossHistory();

    const form = document.getElementById("modelTrainForm");
    if (form) {
        form.addEventListener("submit", function (e) {
            e.preventDefault();
            triggerTraining();
        });
    }
});

function initLossChart() {
    const ctx = document.getElementById("trainingLossChart");
    if (!ctx) return;

    lossChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Training Loss (Thermal Weighted MSE)",
                    data: [],
                    borderColor: "#2563eb",
                    backgroundColor: "rgba(37, 99, 235, 0.08)",
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                },
                {
                    label: "Validation Loss",
                    data: [],
                    borderColor: "#10b981",
                    backgroundColor: "transparent",
                    borderWidth: 2,
                    borderDash: [4, 4],
                    tension: 0.2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top", labels: { boxWidth: 12, font: { size: 11 } } },
                tooltip: { mode: "index", intersect: false }
            },
            scales: {
                x: {
                    title: { display: true, text: "Epoch", font: { size: 11 } },
                    grid: { color: "#f1f5f9" }
                },
                y: {
                    title: { display: true, text: "Loss (Thermal MSE)", font: { size: 11 } },
                    grid: { color: "#f1f5f9" }
                }
            }
        }
    });
}

async function triggerTraining() {
    const archRadio = document.querySelector('input[name="modelArchitecture"]:checked');
    const architecture = archRadio ? archRadio.value : "RESIDUAL_LSTM";
    const epochs = parseInt(document.getElementById("trainEpochs").value, 10);
    const batchSize = parseInt(document.getElementById("trainBatchSize").value, 10);

    const btn = document.getElementById("btnStartTraining");
    btn.disabled = true;

    showTrainFeedback("Initiating model training pipeline...", "info");

    try {
        const resp = await fetch("/api/model/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                architecture: architecture,
                epochs: epochs,
                batch_size: batchSize
            })
        });

        const res = await resp.json();
        if (resp.ok && res.success) {
            showTrainFeedback(res.message || "Model training initiated!", "success");
            startTrainingMonitor();
        } else {
            btn.disabled = false;
            showTrainFeedback(`Training Failed: ${res.error || res.message}`, "danger");
        }
    } catch (e) {
        btn.disabled = false;
        showTrainFeedback(`Network Error: ${e.message}`, "danger");
    }
}

function startTrainingMonitor() {
    if (trainingPollTimer) clearInterval(trainingPollTimer);

    trainingPollTimer = setInterval(async () => {
        try {
            const resp = await fetch("/api/model/status");
            const res = await resp.json();
            if (res.success && res.data) {
                const s = res.data;
                const status = s.status || "RUNNING";
                const progress = Math.round((s.progress || 0) * 100);

                const badge = document.getElementById("liveTrainStatusBadge");
                const bar = document.getElementById("liveProgressBar");
                const pct = document.getElementById("liveProgressPercent");
                const msg = document.getElementById("liveStageMessage");
                const btn = document.getElementById("btnStartTraining");

                if (badge) {
                    badge.textContent = status;
                    badge.className = `badge ${status === 'COMPLETED' ? 'bg-success' : (status === 'RUNNING' ? 'bg-primary' : (status === 'FAILED' ? 'bg-danger' : 'bg-secondary'))}`;
                }
                if (bar) bar.style.width = progress + "%";
                if (pct) pct.textContent = progress + "%";
                if (msg) msg.textContent = s.message || status;

                if (status === "COMPLETED" || status === "READY") {
                    clearInterval(trainingPollTimer);
                    btn.disabled = false;
                    loadLossHistory();
                    showTrainFeedback("Model converged and saved successfully!", "success");
                } else if (status === "FAILED") {
                    clearInterval(trainingPollTimer);
                    btn.disabled = false;
                    showTrainFeedback(`Training error: ${s.message}`, "danger");
                }
            }
        } catch (e) {
            console.error("Monitor poll error:", e);
        }
    }, 2000);
}

async function loadCurrentTrainingStatus() {
    try {
        const resp = await fetch("/api/model/status");
        const res = await resp.json();
        if (res.success && res.data) {
            const s = res.data;
            const badge = document.getElementById("liveTrainStatusBadge");
            const bar = document.getElementById("liveProgressBar");
            const pct = document.getElementById("liveProgressPercent");
            const msg = document.getElementById("liveStageMessage");

            if (badge) badge.textContent = s.status || "IDLE";
            if (bar) bar.style.width = Math.round((s.progress || 0) * 100) + "%";
            if (pct) pct.textContent = Math.round((s.progress || 0) * 100) + "%";
            if (msg) msg.textContent = s.message || "Ready";

            if (s.status === "RUNNING") {
                startTrainingMonitor();
            }
        }
    } catch (e) {
        console.error("Load status error:", e);
    }
}

async function loadLossHistory() {
    try {
        const resp = await fetch("/api/model/history");
        const res = await resp.json();
        if (res.success && res.data) {
            const hist = res.data;
            const loss = hist.loss || [];
            const valLoss = hist.val_loss || [];

            if (lossChart && loss.length > 0) {
                const labels = loss.map((_, i) => `Epoch ${i + 1}`);
                lossChart.data.labels = labels;
                lossChart.data.datasets[0].data = loss;
                lossChart.data.datasets[1].data = valLoss;
                lossChart.update();

                document.getElementById("metricEpochsTrained").textContent = loss.length;
                const minVal = valLoss.length > 0 ? Math.min(...valLoss).toFixed(5) : "--";
                document.getElementById("metricBestValLoss").textContent = minVal;
                document.getElementById("metricActiveArch").textContent = hist.architecture || "Residual LSTM";
            }
        }
    } catch (e) {
        console.error("Loss history error:", e);
    }
}

function showTrainFeedback(msg, type) {
    const el = document.getElementById("trainFeedbackBox");
    if (!el) return;
    el.className = `alert alert-${type} py-2 small mb-3`;
    el.textContent = msg;
    el.classList.remove("d-none");
}
