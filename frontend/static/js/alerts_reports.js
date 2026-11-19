/**
 * Alerts & Reports Controller
 * Monitors thermal excursion incidents and exports operational audit logs.
 */

let incidentRecords = [];
let activeOnThreshold = 28.5;
let activeOffThreshold = 26.5;

document.addEventListener("DOMContentLoaded", function () {
    loadThermalAuditData();
});

async function loadThermalAuditData() {
    try {
        // Fetch authoritative thresholds
        try {
            const thResp = await fetch("/api/cooling/thresholds");
            const thRes = await thResp.json();
            if (thRes.success && thRes.data) {
                activeOnThreshold = Number(thRes.data.on_threshold) || 28.5;
                activeOffThreshold = Number(thRes.data.off_threshold) || 26.5;
            }
        } catch (thErr) {
            console.warn("Could not fetch thresholds, using defaults:", thErr);
        }

        // Update threshold UI display elements
        const cfgOn = document.getElementById("cfgOnThreshold");
        const cfgOff = document.getElementById("cfgOffThreshold");
        if (cfgOn) cfgOn.value = activeOnThreshold.toFixed(1);
        if (cfgOff) cfgOff.value = activeOffThreshold.toFixed(1);

        const bNorm = document.getElementById("badgeRangeNormal");
        const bWarn = document.getElementById("badgeRangeWarning");
        const bCool = document.getElementById("badgeRangeCooling");
        const dbLabel = document.getElementById("deadbandRangeLabel");

        if (bNorm) bNorm.textContent = `< ${activeOffThreshold.toFixed(1)}°C`;
        if (bWarn) bWarn.textContent = `${activeOffThreshold.toFixed(1)}°C - ${activeOnThreshold.toFixed(1)}°C`;
        if (bCool) bCool.textContent = `>= ${activeOnThreshold.toFixed(1)}°C`;
        if (dbLabel) dbLabel.textContent = `Hysteresis Deadband (${activeOffThreshold.toFixed(1)}°C - ${activeOnThreshold.toFixed(1)}°C):`;

        const resp = await fetch("/api/prediction/latest");
        const res = await resp.json();
        if (res.success && res.data) {
            const pred = res.data.prediction || {};
            const cooling = res.data.cooling || {};
            if (cooling.on_threshold != null) activeOnThreshold = Number(cooling.on_threshold);
            if (cooling.off_threshold != null) activeOffThreshold = Number(cooling.off_threshold);

            const temps = pred.forecast_temperatures || [];
            const timestamps = pred.forecast_timestamps || [];

            processIncidents(temps, timestamps);
        }
    } catch (e) {
        console.error("Failed to load prediction alerts data:", e);
    }
}

function processIncidents(temps, timestamps) {
    let normalCount = 0;
    let warningCount = 0;
    let criticalCount = 0;
    let emergencyCount = 0;

    incidentRecords = [];

    temps.forEach((t, i) => {
        const ts = timestamps[i] || `Step +${i + 1}`;
        let severity = "NORMAL";
        let badgeClass = "success";
        let chillerAction = "STANDBY (OFF)";

        if (t >= 33.0) {
            emergencyCount++;
            severity = "EMERGENCY";
            badgeClass = "dark";
            chillerAction = "MAX COOLING (FORCED ON)";
        } else if (t >= activeOnThreshold) {
            criticalCount++;
            severity = "CRITICAL";
            badgeClass = "danger";
            chillerAction = "AUTOMATIC ACTIVATION (ON)";
        } else if (t > activeOffThreshold) {
            warningCount++;
            severity = "WARNING";
            badgeClass = "warning";
            chillerAction = "DEADBAND HOLD";
        } else {
            normalCount++;
        }

        if (severity !== "NORMAL") {
            const delta = (t - activeOnThreshold).toFixed(2);
            incidentRecords.push({
                timestamp: ts,
                severity: severity,
                badgeClass: badgeClass,
                inlet_temp: t,
                delta: delta > 0 ? `+${delta}°C` : `${delta}°C`,
                action: chillerAction,
                status: "Forecasted"
            });
        }
    });

    const total = temps.length || 1;
    const normalPct = Math.round((normalCount / total) * 100);

    document.getElementById("statNormalPercent").textContent = `${normalPct}%`;
    document.getElementById("statWarningCount").textContent = warningCount;
    document.getElementById("statCriticalCount").textContent = criticalCount;
    document.getElementById("statEmergencyCount").textContent = emergencyCount;
    document.getElementById("incidentTableCount").textContent = `${incidentRecords.length} Incidents Forecasted`;

    renderIncidentsTable(incidentRecords);
}

function renderIncidentsTable(records) {
    const tbody = document.getElementById("incidentsTableBody");
    if (!tbody) return;

    if (!records || records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-success fw-semibold"><i data-lucide="check-circle" class="me-1"></i>No thermal threshold violations forecasted. Facility within safe operating envelope.</td></tr>';
        lucide.createIcons();
        return;
    }

    tbody.innerHTML = "";
    records.slice(0, 50).forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="font-monospace small">${r.timestamp}</td>
            <td><span class="badge bg-${r.badgeClass}">${r.severity}</span></td>
            <td class="fw-bold">${r.inlet_temp}°C</td>
            <td><span class="${parseFloat(r.delta) >= 0 ? 'text-danger fw-bold' : 'text-muted'}">${r.delta}</span></td>
            <td><span class="badge bg-light text-secondary border">${r.action}</span></td>
            <td><span class="text-muted small">${r.status}</span></td>
        `;
        tbody.appendChild(tr);
    });
    lucide.createIcons();
}

function exportAuditReport(format) {
    if (!incidentRecords || incidentRecords.length === 0) {
        alert("No incident records available to export.");
        return;
    }

    if (format === "json") {
        const jsonStr = JSON.stringify(incidentRecords, null, 2);
        const blob = new Blob([jsonStr], { type: "application/json" });
        downloadBlob(blob, `thermal_incident_audit_${Date.now()}.json`);
    } else {
        // CSV export
        const headers = ["Timestamp", "Severity", "Inlet_Temperature_C", "Threshold_Delta", "Chiller_Action", "Status"];
        const rows = incidentRecords.map(r => [
            `"${r.timestamp}"`,
            r.severity,
            r.inlet_temp,
            `"${r.delta}"`,
            `"${r.action}"`,
            r.status
        ]);
        const csvContent = [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        downloadBlob(blob, `thermal_incident_audit_${Date.now()}.csv`);
    }
}

function downloadBlob(blob, filename) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
