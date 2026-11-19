/**
 * Data Source Management Controller
 * Handles File Uploads (CSV, XLSX, XLS, JSON) and External Telemetry REST API Connections.
 */

let selectedFile = null;
let pollIntervalTimer = null;

document.addEventListener("DOMContentLoaded", function () {
    initDropZone();
    loadActiveSourceProfile();
    loadIngestionHistory();
    loadApiStatus();
});

function initDropZone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("csvFileInput");
    const uploadForm = document.getElementById("datasetUploadForm");

    if (!dropZone || !fileInput) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleFileSelection(dt.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    if (uploadForm) {
        uploadForm.addEventListener("submit", function (e) {
            e.preventDefault();
            uploadSelectedFile();
        });
    }
}

function handleFileSelection(file) {
    selectedFile = file;
    const fileInfo = document.getElementById("fileSelectionInfo");
    const nameEl = document.getElementById("selectedFileName");
    const sizeEl = document.getElementById("selectedFileSize");

    if (fileInfo && nameEl && sizeEl) {
        fileInfo.classList.remove("d-none");
        nameEl.textContent = file.name;
        const sizeKb = (file.size / 1024).toFixed(1);
        sizeEl.textContent = sizeKb > 1024 ? (sizeKb / 1024).toFixed(2) + " MB" : sizeKb + " KB";
    }
}

async function uploadSelectedFile() {
    if (!selectedFile) {
        showFeedback("uploadFeedbackBox", "Please select a file to upload.", "danger");
        return;
    }

    const btnSubmit = document.getElementById("btnUploadSubmit");
    const progressContainer = document.getElementById("pipelineProgressContainer");
    const progressBar = document.getElementById("pipelineProgressBar");
    const progressPct = document.getElementById("pipelineProgressPct");
    const stageText = document.getElementById("pipelineStageText");

    btnSubmit.disabled = true;
    progressContainer.classList.remove("d-none");
    showFeedback("uploadFeedbackBox", "Uploading dataset to storage...", "info");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const resp = await fetch("/api/dataset/upload", {
            method: "POST",
            body: formData
        });

        const result = await resp.json();

        if (resp.ok && result.success) {
            showFeedback("uploadFeedbackBox", "Dataset uploaded! Pipeline triggered. Evaluating suitability...", "success");
            startPipelinePolling();
        } else {
            btnSubmit.disabled = false;
            const errs = result.errors ? result.errors.join("; ") : (result.error || "Upload failed");
            showFeedback("uploadFeedbackBox", `Validation Error: ${errs}`, "danger");
        }
    } catch (err) {
        btnSubmit.disabled = false;
        showFeedback("uploadFeedbackBox", `Network Error: ${err.message}`, "danger");
    }
}

function startPipelinePolling() {
    if (pollIntervalTimer) clearInterval(pollIntervalTimer);
    
    pollIntervalTimer = setInterval(async () => {
        try {
            const resp = await fetch("/api/dataset/status");
            const res = await resp.json();
            if (res.success && res.data) {
                const job = res.data;
                const pct = Math.round((job.progress || 0) * 100);
                
                const bar = document.getElementById("pipelineProgressBar");
                const pctEl = document.getElementById("pipelineProgressPct");
                const textEl = document.getElementById("pipelineStageText");
                const btnSubmit = document.getElementById("btnUploadSubmit");

                if (bar) bar.style.width = pct + "%";
                if (pctEl) pctEl.textContent = pct + "%";
                if (textEl) textEl.textContent = job.stage_message || job.state;

                if (job.state === "COMPLETED") {
                    clearInterval(pollIntervalTimer);
                    btnSubmit.disabled = false;
                    showFeedback("uploadFeedbackBox", "Pipeline completed successfully! Dataset validated and active.", "success");
                    loadActiveSourceProfile();
                    loadIngestionHistory();
                } else if (job.state === "FAILED") {
                    clearInterval(pollIntervalTimer);
                    btnSubmit.disabled = false;
                    showFeedback("uploadFeedbackBox", `Pipeline Failed: ${job.error || "Unknown error"}`, "danger");
                }
            }
        } catch (e) {
            console.error("Pipeline poll error:", e);
        }
    }, 2000);
}

function toggleAuthFields() {
    const authType = document.getElementById("apiAuthType").value;
    const headerInput = document.getElementById("apiHeaderName");
    const keyContainer = document.getElementById("apiKeyContainer");

    if (authType === "NONE") {
        keyContainer.classList.add("d-none");
    } else {
        keyContainer.classList.remove("d-none");
    }

    if (authType === "BEARER_TOKEN") {
        headerInput.value = "Authorization";
    } else if (authType === "API_KEY_HEADER") {
        headerInput.value = "X-API-Key";
    }
}

function toggleSecretVisibility() {
    const input = document.getElementById("apiKeySecret");
    const icon = document.getElementById("eyeIcon");
    if (input.type === "password") {
        input.type = "text";
        icon.setAttribute("data-lucide", "eye-off");
    } else {
        input.type = "password";
        icon.setAttribute("data-lucide", "eye");
    }
    lucide.createIcons();
}

async function testApiConnection() {
    const url = document.getElementById("apiEndpointUrl").value.trim();
    if (!url) {
        showFeedback("apiTestFeedback", "Please enter an endpoint URL.", "warning");
        return;
    }

    const payload = {
        endpoint_url: url,
        auth_type: document.getElementById("apiAuthType").value,
        api_key: document.getElementById("apiKeySecret").value,
        header_name: document.getElementById("apiHeaderName").value
    };

    const btn = document.getElementById("btnTestApi");
    btn.disabled = true;
    showFeedback("apiTestFeedback", "Pinging endpoint...", "info");

    try {
        const resp = await fetch("/api/datasource/test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const res = await resp.json();
        btn.disabled = false;

        if (res.success) {
            const data = res.data;
            showFeedback(
                "apiTestFeedback", 
                `Connected successfully! (${data.records_detected} telemetry records detected, latency: ${data.latency_ms}ms).`,
                "success"
            );
        } else {
            showFeedback("apiTestFeedback", `Connection Test Failed: ${res.error || res.message}`, "danger");
        }
    } catch (e) {
        btn.disabled = false;
        showFeedback("apiTestFeedback", `Network error: ${e.message}`, "danger");
    }
}

const apiConnectForm = document.getElementById("apiConnectForm");
if (apiConnectForm) {
    apiConnectForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        const url = document.getElementById("apiEndpointUrl").value.trim();
        const authType = document.getElementById("apiAuthType").value;
        const apiKey = document.getElementById("apiKeySecret").value;
        const headerName = document.getElementById("apiHeaderName").value;
        const jsonPath = document.getElementById("apiJsonPath").value;
        const polling = document.getElementById("apiPollingInterval").value;

        const btn = document.getElementById("btnConnectApi");
        btn.disabled = true;
        showFeedback("apiTestFeedback", "Connecting API and syncing telemetry...", "info");

        try {
            // 1. Connect
            const connResp = await fetch("/api/datasource/connect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    endpoint_url: url,
                    auth_type: authType,
                    api_key: apiKey,
                    header_name: headerName,
                    json_path: jsonPath,
                    polling_interval_minutes: polling
                })
            });
            const connRes = await connResp.json();
            if (!connRes.success) {
                btn.disabled = false;
                showFeedback("apiTestFeedback", `Connection Failed: ${connRes.error || connRes.message}`, "danger");
                return;
            }

            // 2. Sync
            const syncResp = await fetch("/api/datasource/sync", { method: "POST" });
            const syncRes = await syncResp.json();
            btn.disabled = false;

            if (syncRes.success) {
                showFeedback("apiTestFeedback", `Synced ${syncRes.data.total_records} records! Pipeline started.`, "success");
                startPipelinePolling();
                loadActiveSourceProfile();
                loadIngestionHistory();
            } else {
                showFeedback("apiTestFeedback", `Sync Failed: ${syncRes.error || syncRes.message}`, "danger");
            }
        } catch (err) {
            btn.disabled = false;
            showFeedback("apiTestFeedback", `Error: ${err.message}`, "danger");
        }
    });
}

async function loadActiveSourceProfile() {
    try {
        const resp = await fetch("/api/dataset/analysis");
        const res = await resp.json();
        if (res.success && res.data) {
            const data = res.data;
            const schema = data.schema_checks || {};
            const timeInfo = data.time_series_analysis || {};
            const quality = data.data_quality || {};

            document.getElementById("activeSourceName").textContent = schema.detected_schema_name || "Active Telemetry";
            document.getElementById("activeSourceSchema").textContent = schema.detected_schema_name || "--";
            document.getElementById("activeSourceRecords").textContent = (quality.total_records || 0).toLocaleString();
            document.getElementById("activeSourceInterval").textContent = (timeInfo.inferred_interval_minutes || "--") + " mins";
            document.getElementById("activeSourceHorizon").textContent = (data.recommended_horizon_hours || 24.0) + "h";
            document.getElementById("activeSuitabilityVerdict").textContent = data.verdict_label || data.verdict || "Evaluated";

            const badge = document.getElementById("activeSourceStatusBadge");
            if (badge) {
                badge.className = `badge bg-${data.badge_color || "success"}`;
                badge.textContent = data.verdict || "Active";
            }
        }
    } catch (e) {
        console.error("Profile load error:", e);
    }
}

function formatFileSize(bytes) {
    if (!bytes || isNaN(bytes)) return "--";
    const num = Number(bytes);
    if (num < 1024) return `${num} B`;
    if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
    return `${(num / (1024 * 1024)).toFixed(2)} MB`;
}

function getFormatBadge(filename, ext) {
    let extension = (ext || (filename ? filename.split('.').pop() : '') || '').toLowerCase().replace('.', '');
    if (extension === 'csv') {
        return '<span class="badge bg-primary-subtle text-primary border border-primary-subtle">CSV</span>';
    } else if (extension === 'xlsx' || extension === 'xls') {
        return '<span class="badge bg-success-subtle text-success border border-success-subtle">EXCEL</span>';
    } else if (extension === 'json') {
        return '<span class="badge bg-warning-subtle text-warning border border-warning-subtle">JSON</span>';
    }
    return `<span class="badge bg-secondary-subtle text-secondary border">${extension.toUpperCase()}</span>`;
}

async function loadIngestionHistory() {
    const tbody = document.getElementById("ingestionHistoryBody");
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted"><div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div>Loading Cloudinary datasets...</td></tr>';

    try {
        const resp = await fetch("/api/dataset/history?cloudinary_only=true");
        const res = await resp.json();

        if (!resp.ok || !res.success) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center py-4 text-danger"><i data-lucide="alert-triangle" class="me-1" style="width: 14px; height: 14px;"></i>${res.error || res.message || "Cloudinary is unconfigured or unavailable."}</td></tr>`;
            if (window.lucide) window.lucide.createIcons();
            return;
        }

        if (Array.isArray(res.data) && res.data.length > 0) {
            tbody.innerHTML = "";
            const history = [...res.data].reverse();
            history.forEach(item => {
                const tr = document.createElement("tr");
                if (item.is_active) {
                    tr.classList.add("table-active");
                }

                const fname = item.original_filename || item.filename || "dataset.csv";
                const fmtBadge = getFormatBadge(fname, item.file_extension || item.format);
                const sizeStr = formatFileSize(item.file_size_bytes || item.file_size || item.bytes);
                const recs = (item.row_count || item.total_records || 0).toLocaleString();
                const uploadedAt = item.upload_timestamp || item.uploaded_at || item.created_at || item.timestamp || "--";
                
                const cloudStatusBadge = '<span class="badge bg-success-subtle text-success border border-success-subtle"><i data-lucide="cloud" class="me-1" style="width: 12px; height: 12px;"></i>Stored (Raw)</span>';

                const targetId = item.dataset_id || item.filename || item.cloudinary_public_id;
                const actionBtn = item.is_active ?
                    '<span class="badge bg-success text-white py-1 px-2"><i data-lucide="check" style="width: 12px; height: 12px;"></i> Active</span>' :
                    `<button class="btn btn-outline-primary btn-sm py-0 px-2" style="font-size: 0.75rem;" onclick="activateDataset('${targetId}')">Activate</button>`;

                tr.innerHTML = `
                    <td class="fw-semibold text-truncate" style="max-width: 220px;" title="${fname}">
                        <i data-lucide="file-spreadsheet" class="me-1 text-secondary" style="width: 13px; height: 13px;"></i>${fname}
                    </td>
                    <td>${fmtBadge}</td>
                    <td>${sizeStr}</td>
                    <td class="text-muted small">${uploadedAt}</td>
                    <td>${recs}</td>
                    <td>${cloudStatusBadge}</td>
                    <td style="text-align: center;">${actionBtn}</td>
                `;
                tbody.appendChild(tr);
            });
            if (window.lucide) window.lucide.createIcons();
        } else {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-muted"><i data-lucide="cloud-off" class="me-1" style="width: 16px; height: 16px;"></i>No Cloudinary-stored datasets found. Upload a dataset to store it in Cloudinary.</td></tr>';
            if (window.lucide) window.lucide.createIcons();
        }
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center py-4 text-danger"><i data-lucide="alert-triangle" class="me-1" style="width: 14px; height: 14px;"></i>Connection error: ${e.message}</td></tr>`;
        if (window.lucide) window.lucide.createIcons();
    }
}

async function activateDataset(idOrName) {
    try {
        const resp = await fetch("/api/dataset/activate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_id: idOrName, filename: idOrName })
        });
        const res = await resp.json();
        if (res.success) {
            showFeedback("uploadFeedbackBox", `Dataset '${idOrName}' is now active!`, "success");
            loadActiveSourceProfile();
            loadIngestionHistory();
        } else {
            showFeedback("uploadFeedbackBox", `Activation failed: ${res.error || res.message}`, "danger");
        }
    } catch (e) {
        showFeedback("uploadFeedbackBox", `Error: ${e.message}`, "danger");
    }
}

async function loadApiStatus() {
    try {
        const resp = await fetch("/api/datasource/status");
        const res = await resp.json();
        if (res.success && res.data) {
            const cfg = res.data;
            if (cfg.endpoint_url) {
                document.getElementById("apiEndpointUrl").value = cfg.endpoint_url;
                document.getElementById("apiAuthType").value = cfg.auth_type || "NONE";
                document.getElementById("apiHeaderName").value = cfg.header_name || "Authorization";
                document.getElementById("apiJsonPath").value = cfg.json_path || "";
                document.getElementById("apiPollingInterval").value = cfg.polling_interval_minutes || 15;
                if (cfg.api_key_masked && cfg.api_key_masked !== "None") {
                    document.getElementById("apiKeySecret").placeholder = `Stored: ${cfg.api_key_masked}`;
                }
                toggleAuthFields();
            }
        }
    } catch (e) {
        console.error("API status load error:", e);
    }
}

function showFeedback(elementId, message, type) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.className = `alert alert-${type} py-2 small mb-3`;
    el.textContent = message;
    el.classList.remove("d-none");
}
