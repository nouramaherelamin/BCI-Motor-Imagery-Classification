"use strict";

// Same-origin by default: nginx proxies /api to the backend container.
const API_BASE = (window.BCI_API_BASE || "/api/v1").replace(/\/$/, "");
const CHANNELS = ["FZ", "C3", "CZ", "C4"];
const N_SAMPLES = 2500;
const MAX_FILE_BYTES = 2 * 1024 * 1024;
const ALLOWED_EXT = [".csv", ".xlsx"];
const REQUEST_TIMEOUT_MS = 30000;

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"),
  notice: $("model-notice"),
  filePanel: $("file-panel"),
  examplePanel: $("example-panel"),
  fileInput: $("file-input"),
  fileCheck: $("file-check"),
  exampleSelect: $("example-select"),
  exampleDownload: $("example-download"),
  predictBtn: $("predict-btn"),
  error: $("error"),
  result: $("result"),
};

const state = {
  source: "file",
  fileValid: false,
  modelReady: false,
  examples: [],
  busy: false,
};

// ---------- helpers ----------

async function apiFetch(path, options = {}) {
  let response;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    response = await fetch(API_BASE + path, { ...options, signal: controller.signal });
  } catch (err) {
    throw new Error(
      err.name === "AbortError"
        ? "The backend did not respond in time. Please try again."
        : "Cannot reach the backend. Check that the API service is running."
    );
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  const text = await response.text();
  try { body = text ? JSON.parse(text) : null; } catch (_) { body = null; }
  if (!response.ok) {
    throw new Error(describeApiError(response.status, body));
  }
  return body;
}

function describeApiError(status, body) {
  const err = body && body.error;
  if (status === 502 || status === 504) return "The backend service is unavailable. Check that the API container is running.";
  if (status === 413) return "The file is too large for the server (limit 2 MB).";
  if (status === 503) return "The model is not available right now. Try again once the backend reports it as ready.";
  if (err && err.code === "validation_error" && Array.isArray(err.details) && err.details.length) {
    const d = err.details[0];
    return `Invalid input: ${d.msg} (at ${d.loc.join(" → ")}).`;
  }
  if (err && err.code === "prediction_failed") {
    return "The model could not process this trial (for example, the signal may be flat or corrupted).";
  }
  if (err && err.message) return err.message;
  return `Request failed with HTTP ${status}.`;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
  els.error.textContent = "";
}

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.className = `status status-${kind}`;
}

function updatePredictButton() {
  const inputReady = state.source === "file" ? state.fileValid : Boolean(els.exampleSelect.value);
  els.predictBtn.disabled = state.busy || !state.modelReady || !inputReady;
}

function setBusy(busy) {
  state.busy = busy;
  els.predictBtn.innerHTML = busy ? '<span class="spinner" aria-hidden="true"></span> Predicting…' : "Predict";
  els.predictBtn.setAttribute("aria-busy", String(busy));
  updatePredictButton();
}

// ---------- backend status & model info ----------

async function loadStatus() {
  try {
    const health = await apiFetch("/health");
    state.modelReady = health.model_loaded;
    if (health.model_loaded) {
      setStatus("Backend online · model ready", "ok");
    } else {
      setStatus(`Backend online · model ${health.model_status.replace("_", " ")}`, "warn");
      showError("The backend is running but the model is not loaded, so predictions are disabled.");
    }
  } catch (err) {
    state.modelReady = false;
    setStatus("Backend unreachable", "err");
    showError(err.message);
    return;
  } finally {
    updatePredictButton();
  }

  try {
    const info = await apiFetch("/model/info");
    renderModelNotice(info);
  } catch (_) { /* informational only */ }
}

function renderModelNotice(info) {
  const meta = info && info.metadata;
  const sel = meta && meta.evaluation_summary && meta.evaluation_summary.selected_model;
  if (!sel) return;
  const pct = (v) => (v * 100).toFixed(1) + "%";
  els.notice.innerHTML = "";
  const p = document.createElement("p");
  p.innerHTML =
    `<strong>About this model:</strong> ${escapeHtml(info.model_name || "")} (${escapeHtml(info.model_version || "")}). ` +
    `Cross-validated balanced accuracy is <strong>${pct(sel.balanced_accuracy)}</strong> ` +
    `(ROC AUC ${sel.roc_auc.toFixed(3)}, ${escapeHtml(sel.cv_strategy)}, ${sel.n_trials_evaluated} trials) — ` +
    `close to chance level, so treat predictions as a demonstration of the pipeline rather than a reliable decision.`;
  els.notice.appendChild(p);
  els.notice.hidden = false;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- examples ----------

async function loadExamples() {
  try {
    state.examples = await apiFetch("/examples");
  } catch (_) {
    state.examples = [];
  }
  els.exampleSelect.innerHTML = "";
  if (!state.examples.length) {
    els.exampleSelect.innerHTML = '<option value="">No example trials available</option>';
  } else {
    for (const ex of state.examples) {
      const opt = document.createElement("option");
      opt.value = ex.id;
      opt.textContent = `${ex.source_file} — true label: ${ex.ground_truth_label}`;
      els.exampleSelect.appendChild(opt);
    }
  }
  updateExampleDownload();
  updatePredictButton();
}

function updateExampleDownload() {
  const id = els.exampleSelect.value;
  els.exampleDownload.hidden = !id;
  els.exampleDownload.href = id ? `${API_BASE}/examples/${encodeURIComponent(id)}/csv` : "#";
}

// ---------- file validation (client side; the server re-validates everything) ----------

async function validateFile(file) {
  if (!file) return { ok: false, message: "" };
  const name = file.name.toLowerCase();
  const ext = name.slice(name.lastIndexOf("."));
  if (!ALLOWED_EXT.includes(ext)) return { ok: false, message: `Unsupported file type "${ext}". Choose a .csv or .xlsx file.` };
  if (file.size === 0) return { ok: false, message: "The selected file is empty." };
  if (file.size > MAX_FILE_BYTES) return { ok: false, message: "The file is larger than 2 MB." };
  if (ext === ".xlsx") return { ok: true, message: `${file.name} selected (Excel content is checked by the server).` };

  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (!lines.length) return { ok: false, message: "The selected file is empty." };
  const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, "").toUpperCase());
  const missing = CHANNELS.filter((c) => !header.includes(c));
  if (missing.length) return { ok: false, message: `Header is missing column(s): ${missing.join(", ")}.` };
  const rows = lines.length - 1;
  if (rows !== N_SAMPLES) return { ok: false, message: `Found ${rows} data rows; exactly ${N_SAMPLES} are required.` };
  return { ok: true, message: `${file.name}: header and ${rows} rows look valid.` };
}

async function onFileChange() {
  clearError();
  els.result.hidden = true;
  const file = els.fileInput.files[0];
  const check = await validateFile(file);
  state.fileValid = check.ok;
  els.fileCheck.textContent = check.message;
  els.fileCheck.className = `file-check ${check.ok ? "ok" : "bad"}`;
  els.fileCheck.hidden = !check.message;
  updatePredictButton();
}

// ---------- prediction ----------

async function predict() {
  clearError();
  els.result.hidden = true;
  setBusy(true);
  try {
    let result;
    let sourceText;
    let truth = null;
    if (state.source === "file") {
      const file = els.fileInput.files[0];
      const form = new FormData();
      form.append("file", file, file.name);
      result = await apiFetch("/predict/file", { method: "POST", body: form });
      sourceText = `Uploaded file: ${file.name}`;
    } else {
      const id = els.exampleSelect.value;
      const example = await apiFetch(`/examples/${encodeURIComponent(id)}`);
      result = await apiFetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(example.payload),
      });
      sourceText = `Example trial: ${example.source_file}`;
      truth = example.ground_truth_label;
    }
    renderResult(result, sourceText, truth);
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

function renderResult(result, sourceText, truth) {
  const label = result.predicted_label;
  const labelEl = $("result-label");
  labelEl.textContent = label.toUpperCase();
  labelEl.className = `result-label ${label.toLowerCase()}`;

  const probs = result.probabilities || {};
  for (const cls of ["Left", "Right"]) {
    const key = cls.toLowerCase();
    const p = typeof probs[cls] === "number" ? probs[cls] : null;
    $(`bar-${key}`).style.width = p === null ? "0" : `${(p * 100).toFixed(1)}%`;
    $(`pct-${key}`).textContent = p === null ? "n/a" : `${(p * 100).toFixed(1)}%`;
  }

  const gt = $("ground-truth");
  if (truth) {
    const match = truth === label;
    gt.textContent = `True label: ${truth} — prediction ${match ? "matches" : "does not match"}`;
    gt.className = `ground-truth ${match ? "match" : "mismatch"}`;
    gt.hidden = false;
  } else {
    gt.hidden = true;
  }
  $("result-source").textContent = sourceText;
  $("result-version").textContent = result.model_version;
  els.result.hidden = false;
  els.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------- wiring ----------

document.querySelectorAll('input[name="source"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    state.source = e.target.value;
    els.filePanel.hidden = state.source !== "file";
    els.examplePanel.hidden = state.source !== "example";
    clearError();
    els.result.hidden = true;
    updatePredictButton();
  });
});
els.fileInput.addEventListener("change", onFileChange);
els.exampleSelect.addEventListener("change", () => { updateExampleDownload(); updatePredictButton(); });
els.predictBtn.addEventListener("click", predict);

loadStatus();
loadExamples();
