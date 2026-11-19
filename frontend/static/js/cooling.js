/**
 * Chiller Control Subsystem and Mode Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initial pause of fan SVG until status loads
    const initialSvg = document.getElementById('chillerFanSvg');
    if (initialSvg && initialSvg.pauseAnimations) {
        try { initialSvg.pauseAnimations(); } catch (e) {}
    }
    loadCoolingStatus();
    loadCoolingHistory();
    setupThresholdForm();
    setInterval(loadCoolingStatus, 15000);
});

async function loadCoolingStatus() {
    try {
        const res = await fetch('/api/cooling/status');
        const json = await res.json();
        if (!json.success || !json.data) return;

        updateChillerUI(json.data);
    } catch (err) {
        console.error('Error loading cooling status:', err);
    }
}

let lastChillerRunningState = null;
let chillerTransitionTimer = null;

function updateChillerUI(data) {
    const cabinet = document.getElementById('chillerCabinet');
    const pill = document.getElementById('chillerStatusPill');
    const dot = document.getElementById('chillerStatusDot');
    const text = document.getElementById('chillerStatusText');
    const modeBadge = document.getElementById('displayOperatingMode');
    const reasonText = document.getElementById('chillerReasonText');
    const powerVal = document.getElementById('chillerPowerVal');
    const lastChange = document.getElementById('chillerLastChange');

    // 1. Chiller State (ON/RUNNING vs OFF/STOPPED)
    const isRunning = Boolean(data.is_running || data.chiller_state === 'ON' || data.chiller_state === 'RUNNING' || data.canonical_chiller_state === 'ON');
    const fanSvg = document.getElementById('chillerFanSvg') || (cabinet ? cabinet.querySelector('svg') : null);

    if (isRunning) {
        if (pill) pill.className = 'status-pill-lg running';
        if (dot) dot.className = 'dot-indicator green';
        if (text) text.textContent = 'CHILLER RUNNING (ON)';
        if (cabinet) cabinet.className = 'chiller-cabinet running';

        // Rotate fan blades continuously when RUNNING
        if (fanSvg && fanSvg.unpauseAnimations) {
            try { fanSvg.unpauseAnimations(); } catch (e) {}
        }
    } else {
        if (pill) pill.className = 'status-pill-lg stopped';
        if (dot) dot.className = 'dot-indicator gray';
        if (text) text.textContent = 'CHILLER STOPPED (OFF)';
        if (cabinet) cabinet.className = 'chiller-cabinet';

        // Stop rotation and keep fan stationary when STOPPED
        if (fanSvg && fanSvg.pauseAnimations) {
            try { fanSvg.pauseAnimations(); } catch (e) {}
        }
    }
    lastChillerRunningState = isRunning;

    if (reasonText) reasonText.textContent = data.activation_reason || 'System operating in normal band.';
    if (powerVal) powerVal.textContent = `${data.cooling_power_consumption_kw.toFixed(1)} kW`;
    if (lastChange) lastChange.textContent = data.last_transition_time || '--';

    // 2. Operating Mode
    const mode = data.operating_mode || 'AUTOMATIC';
    if (modeBadge) modeBadge.textContent = mode;

    const btnAuto = document.getElementById('btnModeAuto');
    const btnManual = document.getElementById('btnModeManual');
    const manualSection = document.getElementById('manualControlsSection');
    const autoDesc = document.getElementById('autoModeDescription');

    if (mode === 'AUTOMATIC') {
        if (btnAuto) btnAuto.className = 'btn btn-sm btn-outline-primary active';
        if (btnManual) btnManual.className = 'btn btn-sm btn-outline-primary';
        if (manualSection) manualSection.classList.add('d-none');
        if (autoDesc) autoDesc.classList.remove('d-none');
    } else {
        if (btnAuto) btnAuto.className = 'btn btn-sm btn-outline-primary';
        if (btnManual) btnManual.className = 'btn btn-sm btn-outline-primary active';
        if (manualSection) manualSection.classList.remove('d-none');
        if (autoDesc) autoDesc.classList.add('d-none');
    }

    // 3. Threshold fields & descriptions
    const inOn = document.getElementById('inputOnThreshold');
    const inOff = document.getElementById('inputOffThreshold');
    if (inOn && document.activeElement !== inOn && data.on_threshold != null) inOn.value = data.on_threshold;
    if (inOff && document.activeElement !== inOff && data.off_threshold != null) inOff.value = data.off_threshold;

    const autoOn = document.getElementById('autoOnLabel');
    const autoOff = document.getElementById('autoOffLabel');
    if (autoOn && data.on_threshold != null) autoOn.textContent = `≥ ${data.on_threshold.toFixed(1)}°C`;
    if (autoOff && data.off_threshold != null) autoOff.textContent = `≤ ${data.off_threshold.toFixed(1)}°C`;
}

async function switchMode(targetMode) {
    try {
        const res = await fetch('/api/cooling/mode', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mode: targetMode})
        });
        const json = await res.json();
        if (json.success) {
            updateChillerUI(json.data);
            loadCoolingHistory();
        } else {
            alert('Mode switch failed: ' + json.error);
        }
    } catch (err) {
        alert('Server connection error: ' + err.message);
    }
}

async function sendManualCommand(action) {
    try {
        const res = await fetch('/api/cooling/manual', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: action})
        });
        const json = await res.json();
        if (json.success) {
            updateChillerUI(json.data);
            loadCoolingHistory();
        } else {
            alert('Manual command failed: ' + json.error);
        }
    } catch (err) {
        alert('Server connection error: ' + err.message);
    }
}

function setupThresholdForm() {
    const form = document.getElementById('coolingThresholdForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const alertBox = document.getElementById('thresholdAlertBox');
        alertBox.classList.add('d-none');

        const onTh = parseFloat(document.getElementById('inputOnThreshold').value);
        const offTh = parseFloat(document.getElementById('inputOffThreshold').value);

        try {
            const res = await fetch('/api/cooling/thresholds', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({on_threshold: onTh, off_threshold: offTh})
            });
            const json = await res.json();
            if (json.success) {
                alertBox.className = 'alert alert-success py-2 small mb-3';
                alertBox.textContent = json.message;
                alertBox.classList.remove('d-none');
                loadCoolingStatus();
                loadCoolingHistory();
            } else {
                alertBox.className = 'alert alert-danger py-2 small mb-3';
                alertBox.textContent = json.error || 'Failed to update thresholds.';
                alertBox.classList.remove('d-none');
            }
        } catch (err) {
            alertBox.className = 'alert alert-danger py-2 small mb-3';
            alertBox.textContent = 'Server error: ' + err.message;
            alertBox.classList.remove('d-none');
        }
    });
}

async function loadCoolingHistory() {
    try {
        const res = await fetch('/api/cooling/history?limit=40');
        const json = await res.json();
        if (!json.success || !json.data) return;

        const tbody = document.getElementById('coolingEventsBody');
        const events = json.data;

        if (events.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">No state transition events recorded.</td></tr>';
            return;
        }

        tbody.innerHTML = events.map(ev => {
            const prevBadge = (ev.previous_state === 'RUNNING' || ev.previous_state === 'ON')
                ? '<span class="badge bg-success">RUNNING</span>'
                : '<span class="badge bg-secondary">STOPPED</span>';

            const newBadge = (ev.new_state === 'RUNNING' || ev.new_state === 'ON')
                ? '<span class="badge bg-success">RUNNING</span>'
                : '<span class="badge bg-secondary">STOPPED</span>';

            const modeBadge = ev.mode === 'MANUAL'
                ? '<span class="badge bg-danger text-uppercase">MANUAL</span>'
                : '<span class="badge bg-light text-secondary border text-uppercase">AUTO</span>';

            return `
                <tr>
                    <td class="small fw-semibold">${ev.timestamp}</td>
                    <td>${modeBadge}</td>
                    <td>${prevBadge} &rarr; ${newBadge}</td>
                    <td>${ev.trigger_temperature ? ev.trigger_temperature.toFixed(2) + ' °C' : '--'}</td>
                    <td class="small text-secondary">${ev.reason || '--'}</td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error loading cooling events:', err);
    }
}
