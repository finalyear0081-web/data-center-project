/**
 * Dataset Suitability Analysis Controller
 * Renders 5-Point Diagnostic Engine results, canonical feature mappings, and handles administrator remapping.
 */

let activeAnalysisData = null;
let rawColumnsList = [];

document.addEventListener("DOMContentLoaded", function () {
    loadSuitabilityAnalysis();
    loadRawColumnsForRemap();
});

async function loadSuitabilityAnalysis() {
    try {
        const resp = await fetch("/api/dataset/analysis");
        const res = await resp.json();
        if (res.success && res.data) {
            activeAnalysisData = res.data;
            renderSuitabilityAnalysis(res.data);
        }
    } catch (e) {
        console.error("Failed to load suitability analysis:", e);
    }
}

function renderSuitabilityAnalysis(data) {
    // 1. Hero Verdict Banner
    const verdict = data.verdict || "SUITABLE";
    const badgeColor = data.badge_color || "success";
    const badge = document.getElementById("verdictBadge");
    const title = document.getElementById("verdictTitle");
    const summary = document.getElementById("verdictSummary");
    const horizonBadge = document.getElementById("recommendedHorizonBadge");
    const stepsBadge = document.getElementById("recommendedStepsBadge");

    if (badge) {
        badge.className = `badge bg-${badgeColor} px-2 py-1`;
        badge.textContent = verdict;
    }
    if (title) title.textContent = data.verdict_label || verdict;
    if (summary) summary.textContent = data.summary_message || "";
    if (horizonBadge) horizonBadge.textContent = `${data.recommended_horizon_hours || 24.0} Hours`;
    if (stepsBadge) stepsBadge.textContent = `(${data.recommended_horizon_steps || 144} steps)`;

    // 2. Schema Card
    const schema = data.schema_checks || {};
    document.getElementById("schemaTypePill").textContent = schema.detected_schema_name || "Schema";
    document.getElementById("diagTargetStatus").textContent = schema.target_detected ? "Detected (Target)" : "Missing";
    document.getElementById("diagTargetStatus").className = schema.target_detected ? "fw-bold text-success" : "fw-bold text-danger";
    document.getElementById("diagTargetCol").textContent = schema.target_source_column || "--";
    document.getElementById("diagTargetConf").textContent = Math.round((schema.target_confidence || 0) * 100) + "%";
    document.getElementById("diagAvailableSensors").textContent = `${schema.available_features_count || 0} features`;

    // 3. Data Quality Card
    const quality = data.data_quality || {};
    document.getElementById("qualityRowsPill").textContent = `${(quality.total_records || 0).toLocaleString()} Rows`;
    document.getElementById("diagDuplicates").textContent = quality.duplicate_records_count || 0;
    document.getElementById("diagInfinite").textContent = quality.infinite_values_count || 0;
    document.getElementById("diagConstantCols").textContent = (quality.constant_columns && quality.constant_columns.length > 0) ? quality.constant_columns.join(", ") : "None";

    // 4. Time-Series Diagnostics Card
    const timeInfo = data.time_series_analysis || {};
    document.getElementById("diagInterval").textContent = `${timeInfo.inferred_interval_minutes || 5} min`;
    document.getElementById("diagRegularity").textContent = timeInfo.is_regular_sampling ? "Regular (>= 85%)" : "Irregular Jitter";
    document.getElementById("diagRegularity").className = timeInfo.is_regular_sampling ? "fw-bold text-success" : "fw-bold text-warning";
    document.getElementById("diagDuration").textContent = `${timeInfo.total_duration_hours || 0} hrs`;
    document.getElementById("diagExperiments").textContent = `${timeInfo.experiments_count || 1} partition(s)`;

    // 5. Thermal Volatility Card
    const temp = data.temperature_analysis || {};
    document.getElementById("diagThermalVolatility").textContent = temp.thermal_volatility || "Normal";
    document.getElementById("diagTempRange").textContent = (temp.min_celsius !== null) ? `${temp.min_celsius}°C - ${temp.max_celsius}°C` : "--";
    document.getElementById("diagTempMeanStd").textContent = (temp.mean_celsius !== null) ? `${temp.mean_celsius}°C ± ${temp.std_celsius}°C` : "--";
    document.getElementById("diagCount28").textContent = temp.counts_above_28 || 0;
    document.getElementById("diagCount30_33").textContent = `${temp.counts_above_30 || 0} / ${temp.counts_above_33 || 0}`;
    document.getElementById("diagOutOfBounds").textContent = temp.out_of_bounds_count || 0;

    fetch("/api/cooling/thresholds").then(r => r.json()).then(res => {
        if (res.success && res.data) {
            const lbl = document.getElementById("diagLabelOn");
            if (lbl) lbl.textContent = `Readings >= ${(res.data.on_threshold || 28.5).toFixed(1)}°C (ON):`;
        }
    }).catch(() => {});

    // 6. Diagnostics Notes (Reasons, Warnings, Recommendations)
    const reasonsBox = document.getElementById("diagnosticsReasonsContainer");
    const warningsBox = document.getElementById("diagnosticsWarningsContainer");
    const recsBox = document.getElementById("diagnosticsRecommendationsContainer");

    if (reasonsBox) {
        if (data.reasons && data.reasons.length > 0) {
            reasonsBox.innerHTML = data.reasons.map(r => `<div class="text-danger mb-1"><i data-lucide="x-circle" class="me-1" style="width: 14px; height: 14px;"></i>${r}</div>`).join("");
        } else {
            reasonsBox.innerHTML = "";
        }
    }
    if (warningsBox) {
        if (data.warnings && data.warnings.length > 0) {
            warningsBox.innerHTML = data.warnings.map(w => `<div class="text-warning mb-1"><i data-lucide="alert-triangle" class="me-1" style="width: 14px; height: 14px;"></i>${w}</div>`).join("");
        } else {
            warningsBox.innerHTML = "";
        }
    }
    if (recsBox) {
        if (data.recommendations && data.recommendations.length > 0) {
            recsBox.innerHTML = data.recommendations.map(rc => `<div class="text-primary mb-1"><i data-lucide="check" class="me-1" style="width: 14px; height: 14px;"></i>${rc}</div>`).join("");
        } else {
            recsBox.innerHTML = "";
        }
    }

    // 7. Canonical Mapping Table
    renderMappingTable(data.mapping_details || {});
    lucide.createIcons();
}

function renderMappingTable(mappings) {
    const tbody = document.getElementById("schemaMappingBody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const entries = Object.entries(mappings);
    if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-3 text-muted">No mappings available.</td></tr>';
        return;
    }

    entries.forEach(([canonKey, meta]) => {
        const tr = document.createElement("tr");
        const isTarget = (canonKey === "inlet_temperature");
        const roleLabel = isTarget ? '<span class="badge bg-danger-subtle text-danger">Target (Y)</span>' : 
                         (canonKey === "timestamp" ? '<span class="badge bg-primary-subtle text-primary">Time Index</span>' : 
                         '<span class="badge bg-light text-secondary border">Feature (X)</span>');

        const confPct = Math.round((meta.confidence || 0) * 100);
        const confBadge = meta.available ? 
            `<span class="badge ${confPct >= 90 ? 'bg-success' : 'bg-warning'}">${confPct}%</span>` : 
            '<span class="badge bg-secondary">0%</span>';

        const statusBadge = meta.available ?
            '<span class="text-success fw-semibold"><i data-lucide="check" style="width: 14px; height: 14px;"></i> Available</span>' :
            '<span class="text-muted"><i data-lucide="minus" style="width: 14px; height: 14px;"></i> Missing</span>';

        tr.innerHTML = `
            <td class="fw-bold font-monospace">${canonKey}</td>
            <td>${roleLabel}</td>
            <td class="fw-semibold text-truncate" style="max-width: 160px;">${meta.source_column || '<span class="text-muted font-italic">Unmapped</span>'}</td>
            <td>${confBadge}</td>
            <td>${meta.unit || "--"}</td>
            <td>${meta.missing_percentage !== undefined ? meta.missing_percentage + "%" : "--"}</td>
            <td>${statusBadge}</td>
            <td class="text-end">
                <button class="btn btn-sm btn-outline-secondary py-0 px-2" onclick="openRemapFor('${canonKey}')">
                    Remap
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

async function loadRawColumnsForRemap() {
    try {
        const resp = await fetch("/api/dataset/preview?limit=1");
        const res = await resp.json();
        if (res.success && res.data && res.data.records && res.data.records.length > 0) {
            rawColumnsList = Object.keys(res.data.records[0]);
            populateRemapSelects(rawColumnsList);
        }
    } catch (e) {
        console.error("Failed to load raw columns:", e);
    }
}

function populateRemapSelects(columns) {
    document.querySelectorAll(".raw-col-select").forEach(select => {
        select.innerHTML = '<option value="">-- None / Automatic --</option>';
        columns.forEach(col => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            select.appendChild(opt);
        });
    });
}

function openRemapFor(canonKey) {
    const modalEl = document.getElementById("remapModal");
    if (!modalEl) return;
    const targetSelect = document.getElementById(`remap_${canonKey}`);
    if (targetSelect) {
        targetSelect.focus();
    }
    const bsModal = new bootstrap.Modal(modalEl);
    bsModal.show();
}

async function submitRemapping() {
    const mappings = {};
    const keys = [
        "inlet_temperature", "outlet_temperature", "timestamp",
        "it_power_consumption", "facility_power_consumption", "experiment_id"
    ];

    keys.forEach(k => {
        const val = document.getElementById(`remap_${k}`).value;
        if (val) mappings[k] = val;
    });

    const feedback = document.getElementById("remapFeedback");
    feedback.className = "alert alert-info py-2 small";
    feedback.textContent = "Updating mappings and re-evaluating dataset...";
    feedback.classList.remove("d-none");

    try {
        const resp = await fetch("/api/dataset/remap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mappings })
        });
        const res = await resp.json();

        if (res.success && res.data) {
            feedback.className = "alert alert-success py-2 small";
            feedback.textContent = "Mappings saved! Re-evaluating complete.";
            setTimeout(() => {
                const modalEl = document.getElementById("remapModal");
                const modal = bootstrap.Modal.getInstance(modalEl);
                if (modal) modal.hide();
                loadSuitabilityAnalysis();
            }, 800);
        } else {
            feedback.className = "alert alert-danger py-2 small";
            feedback.textContent = res.error || "Failed to update mappings.";
        }
    } catch (e) {
        feedback.className = "alert alert-danger py-2 small";
        feedback.textContent = `Error: ${e.message}`;
    }
}

async function triggerReanalysis() {
    try {
        const resp = await fetch("/api/dataset/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });
        const res = await resp.json();
        if (res.success && res.data) {
            renderSuitabilityAnalysis(res.data);
        }
    } catch (e) {
        console.error("Re-analysis failed:", e);
    }
}
