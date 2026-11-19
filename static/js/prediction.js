/**
 * Dedicated Inference and Retraining Trigger Module
 */

async function triggerInferenceCycle() {
    const btn = event?.currentTarget;
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Running Forecast...';
    }

    try {
        const res = await fetch('/api/model/predict', { method: 'POST' });
        const json = await res.json();
        if (json.success) {
            await loadForecastData();
            await loadDashboardSummary();
            await loadAlertsFeed();
            alert('24-Hour LSTM temperature forecast generated successfully.');
        } else {
            alert('Prediction failed: ' + (json.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Error connecting to prediction API: ' + err.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-circle me-1"></i>Run Live Inference';
        }
    }
}

async function triggerModelRetrain() {
    if (!confirm('Initiate LSTM model retraining on the dataset? Training will execute in background.')) {
        return;
    }

    const token = localStorage.getItem('auth_token') || 'demo-token-admin';
    try {
        const res = await fetch('/api/model/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ epochs: 10, batch_size: 64 })
        });
        const json = await res.json();
        if (json.success) {
            alert('Model retraining initiated! You can monitor status and metrics in this tab.');
            pollTrainingStatus();
        } else {
            alert('Retraining request failed: ' + (json.error || json.message));
        }
    } catch (err) {
        alert('Server error: ' + err.message);
    }
}

function pollTrainingStatus() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch('/api/model/status');
            const json = await res.json();
            if (json.success && json.status) {
                if (json.status.status === 'COMPLETED') {
                    clearInterval(interval);
                    loadModelBenchmarks();
                    alert('Model retraining completed successfully! Benchmark charts have been updated.');
                }
            }
        } catch (e) {
            clearInterval(interval);
        }
    }, 5000);
}
