let config = null;
let latestRun = null;
let latestTab = "accuracy";
let drawerRun = null;
let drawerTab = "parsed";
let runStatusTimer = null;

const $ = (id) => document.getElementById(id);

function jsonText(value) {
  return JSON.stringify(value, null, 2);
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (error) {
    throw new Error("Cannot reach the local server. Run `python server.py` in nucleoside-gel-rag-webapp, then open http://127.0.0.1:8777.");
  }
  let data;
  try {
    data = await response.json();
  } catch (error) {
    throw new Error(`The local server returned an invalid response (HTTP ${response.status}). Check the server terminal for details.`);
  }
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function setStatus(message, state = "ok") {
  const target = $("status");
  target.className = `status-line status-${state}`;
  target.textContent = message;
}

function addStatusEvent(message, state = "pending") {
  const log = $("statusLog");
  const entry = document.createElement("div");
  entry.className = `status-log-entry status-${state}`;
  entry.textContent = `${new Date().toLocaleTimeString()} · ${message}`;
  log.prepend(entry);
  while (log.children.length > 6) log.lastElementChild.remove();
}

function setConfigStatus(message, state = "ok") {
  ["configStatus", "resourceConfigStatus"].forEach((id) => {
    const target = $(id);
    target.className = `status-line status-${state}`;
    target.textContent = message;
  });
}

function showView(view) {
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.id === view));
  document.querySelectorAll(".nav").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  if (view === "history") loadRuns();
}

function fillConfigForm(data) {
  config = data;
  $("baseUrl").value = data.base_url || "";
  $("apiKey").value = data.api_key || "";
  $("defaultModel").value = data.model || "";
  $("runModel").value = data.model || "";
  $("temperature").value = data.temperature ?? 0.2;
  $("maxTokens").value = data.max_tokens ?? 10000;
  $("timeout").value = data.timeout ?? 300;
  $("alvadescExe").value = data.alvadesc_exe || "";
  $("alvadescWrapperPath").value = data.alvadesc_wrapper_path || "";
}

async function loadConfig() {
  const loaded = await api("/api/config");
  fillConfigForm(loaded);
  setStatus(`Ready · ${loaded.model || "model not configured"} · ${loaded.reference_count || 0} reference molecules`);
  addStatusEvent("Local server connected and configuration loaded.", "ok");
  setConfigStatus("Configuration loaded from data/config.json.");
}

async function saveConfig() {
  const buttons = [...document.querySelectorAll(".save-config")];
  const payload = {
    base_url: $("baseUrl").value,
    api_key: $("apiKey").value,
    model: $("defaultModel").value,
    temperature: Number($("temperature").value || 0.2),
    max_tokens: Number($("maxTokens").value || 10000),
    timeout: Number($("timeout").value || 300),
    alvadesc_exe: $("alvadescExe").value,
    alvadesc_wrapper_path: $("alvadescWrapperPath").value,
  };
  buttons.forEach((button) => {
    button.disabled = true;
    button.textContent = "Saving...";
  });
  setConfigStatus("Saving LLM configuration...", "pending");
  try {
    const saved = await api("/api/config", { method: "POST", body: JSON.stringify(payload) });
    fillConfigForm(saved);
    setConfigStatus(`Saved · ${saved.model} · ${saved.base_url}`);
    setStatus(`Ready · ${saved.model} · ${saved.reference_count || 0} reference molecules`);
    addStatusEvent("Configuration saved successfully.", "ok");
  } catch (error) {
    setConfigStatus(error.message, "bad");
    addStatusEvent(`Configuration save failed: ${error.message}`, "bad");
    throw error;
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
      button.textContent = "Save Config";
    });
  }
}

function runPayload() {
  return {
    strategy: $("strategy").value,
    model: $("runModel").value || config?.model,
    max_k: Number($("maxK").value || 6),
    chemical_description: $("chemicalDescription").value,
    canonical_smiles: $("canonicalSmiles").value,
    target_experiments: $("targetExperiments").value,
  };
}

function validateRunPayload(payload) {
  if (!config) throw new Error("Configuration has not loaded. Check the local server connection and reload the page.");
  if (!config.api_key) throw new Error("API key is missing. Open Config, enter a valid API key, and click Save Config.");
  if (!payload.model?.trim()) throw new Error("Model name is missing. Enter a model in Run or Config.");
  if (!payload.chemical_description?.trim()) throw new Error("Chemical description is required.");
  if (!payload.canonical_smiles?.trim()) throw new Error("Canonical SMILES is required.");
  if (!payload.target_experiments?.trim()) throw new Error("Target condition / experiment is required.");
  if (!Number.isInteger(payload.max_k) || payload.max_k < 0) throw new Error("Max-k must be a non-negative integer.");
}

function runningMessage(payload, seconds) {
  if (seconds < 2) return `Submitting ${payload.strategy} request to the local server · ${seconds} s elapsed`;
  if (seconds < 5) return `Validating SMILES, descriptors, conditions, and reference retrieval · ${seconds} s elapsed`;
  return `Waiting for retrieval and the ${payload.model} LLM response · ${seconds} s elapsed`;
}

async function runStrategy() {
  const button = $("runButton");
  const payload = runPayload();
  try {
    validateRunPayload(payload);
  } catch (error) {
    setStatus(`Cannot start · ${error.message}`, "bad");
    addStatusEvent(error.message, "bad");
    return;
  }
  const startedAt = Date.now();
  button.disabled = true;
  button.textContent = "Running...";
  setStatus(runningMessage(payload, 0), "pending");
  addStatusEvent(`Run started: ${payload.strategy}, ${payload.model}, Max-${payload.max_k}.`, "pending");
  clearInterval(runStatusTimer);
  runStatusTimer = setInterval(() => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    setStatus(runningMessage(payload, seconds), "pending");
  }, 1000);
  try {
    latestRun = await api("/api/run", { method: "POST", body: JSON.stringify(payload) });
    latestTab = "accuracy";
    renderLatest();
    const seconds = ((Date.now() - startedAt) / 1000).toFixed(1);
    if (latestRun.parsed_json == null) {
      setStatus(`Completed with warning in ${seconds} s · The LLM output did not contain valid JSON · Run saved: ${latestRun.id}`, "bad");
      addStatusEvent("Run saved, but the LLM output could not be parsed as JSON.", "bad");
    } else {
      setStatus(`Completed in ${seconds} s · Run saved: ${latestRun.id}`);
      addStatusEvent(`Run completed and saved as ${latestRun.id}.`, "ok");
    }
  } catch (error) {
    setStatus(`Failed · ${error.message}`, "bad");
    addStatusEvent(`Run failed: ${error.message}`, "bad");
  } finally {
    clearInterval(runStatusTimer);
    runStatusTimer = null;
    button.disabled = false;
    button.textContent = "Run Strategy";
  }
}

function outputContent(run, tab) {
  if (!run) return "No run selected.";
  return {
    accuracy: run.accuracy ?? "No accuracy judgment.",
    parsed: run.parsed_json ?? "No parsed JSON.",
    raw: run.raw_output,
    prompt: run.prompt,
    retrieval: run.retrieval_context,
  }[tab];
}

function renderLatest() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === latestTab);
  });
  const content = outputContent(latestRun, latestTab);
  $("latestOutput").textContent = typeof content === "string" ? content : jsonText(content);
}

function accuracyLabel(accuracy) {
  if (!accuracy || !accuracy.available) return "Accuracy unavailable";
  const pct = accuracy.accuracy == null ? "-" : `${(accuracy.accuracy * 100).toFixed(2)}%`;
  return `${accuracy.correct}/${accuracy.total} correct · ${pct}`;
}

async function loadRuns() {
  const runs = await api("/api/runs");
  $("runs").innerHTML = runs
    .map(
      (run) => `
        <div class="run-row">
          <div>${run.target}<br><span class="muted">${run.id}</span></div>
          <div>${run.strategy} · ${run.model} · Max-${run.max_k}<br><span class="muted">${accuracyLabel(run.accuracy)}</span></div>
          <div class="row-actions">
            <button data-id="${run.id}" class="open-run">Open</button>
            <button data-id="${run.id}" class="delete-run">Delete</button>
          </div>
        </div>
      `,
    )
    .join("");
  document.querySelectorAll(".open-run").forEach((button) => {
    button.addEventListener("click", async () => {
      drawerRun = await api(`/api/runs/${button.dataset.id}`);
      drawerTab = "parsed";
      openDrawer();
    });
  });
  document.querySelectorAll(".delete-run").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!confirm(`Delete run ${button.dataset.id}?`)) return;
      await api(`/api/runs/${encodeURIComponent(button.dataset.id)}`, { method: "DELETE" });
      if (drawerRun?.id === button.dataset.id) closeDrawer();
      await loadRuns();
    });
  });
}

function openDrawer() {
  $("drawerKicker").textContent = `${drawerRun.strategy} · ${drawerRun.model} · Max-${drawerRun.max_k}`;
  $("drawerTitle").textContent = drawerRun.target?.chemical_description || drawerRun.id;
  $("runDrawer").classList.add("open");
  $("runDrawer").setAttribute("aria-hidden", "false");
  renderDrawer();
}

function closeDrawer() {
  $("runDrawer").classList.remove("open");
  $("runDrawer").setAttribute("aria-hidden", "true");
}

function renderDrawer() {
  document.querySelectorAll(".drawer-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === drawerTab);
  });
  const content = outputContent(drawerRun, drawerTab);
  $("drawerOutput").textContent = typeof content === "string" ? content : jsonText(content);
}

document.querySelectorAll(".nav").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
  latestTab = button.dataset.tab;
  renderLatest();
}));
document.querySelectorAll(".drawer-tab").forEach((button) => button.addEventListener("click", () => {
  drawerTab = button.dataset.tab;
  renderDrawer();
}));
$("closeDrawer").addEventListener("click", closeDrawer);
document.addEventListener(
  "pointerdown",
  (event) => {
    const drawer = $("runDrawer");
    if (!drawer.classList.contains("open")) return;
    if (event.target.closest("#runDrawer")) return;
    closeDrawer();
  },
  true,
);
document.querySelectorAll(".save-config").forEach((button) => {
  button.addEventListener("click", () => saveConfig().catch(() => {}));
});
$("runButton").addEventListener("click", runStrategy);

loadConfig().catch((error) => {
  setStatus(error.message, "bad");
  addStatusEvent(error.message, "bad");
  setConfigStatus(error.message, "bad");
});
