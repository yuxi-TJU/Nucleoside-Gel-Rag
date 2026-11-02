const state = {
  data: null,
  view: "overview",
  status: "all",
  selectedRow: null,
  selectedTab: "parsed",
  selectedCondition: 0,
  compareMoleculeSearch: "",
  compareSelections: {
    left: { maxK: "", model: "", strategy: "", round: "", experiment: "" },
    right: { maxK: "", model: "", strategy: "", round: "", experiment: "" },
  },
  compareRenderId: 0,
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function pct(value) {
  return value == null || Number.isNaN(value) ? "-" : `${(value * 100).toFixed(2)}%`;
}

function numberValue(value, fallback = 0) {
  return value == null || Number.isNaN(value) ? fallback : value;
}

function unique(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) =>
    String(a).localeCompare(String(b), undefined, { numeric: true }),
  );
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function optionList(select, values, allLabel, preferred) {
  const current = select.value || preferred || "all";
  select.innerHTML = [`<option value="all">${allLabel}</option>`]
    .concat(values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`))
    .join("");
  select.value = values.includes(current) || current === "all" ? current : preferred || "all";
}

function rowStatus(row) {
  if (row.accuracy >= 1) return "correct";
  if (row.accuracy <= 0) return "wrong";
  return "partial";
}

function accuracyBadge(value) {
  const klass = value >= 0.8 ? "good" : value >= 0.5 ? "warn" : "bad";
  return `<span class="badge ${klass}">${pct(value)}</span>`;
}

function currentFilters() {
  return {
    maxK: els.maxFilter.value,
    model: els.modelFilter.value,
    strategy: els.strategyFilter.value,
    round: els.roundFilter.value,
    query: els.searchInput.value.trim().toLowerCase(),
  };
}

function getFilteredExperiments({ ignoreModel = false, ignoreQuery = false } = {}) {
  const filters = currentFilters();
  return state.data.experiments.filter((exp) => {
    if (filters.maxK !== "all" && String(exp.maxK) !== filters.maxK) return false;
    if (!ignoreModel && filters.model !== "all" && exp.model !== filters.model) return false;
    if (filters.strategy !== "all" && exp.strategyLabel !== filters.strategy) return false;
    if (ignoreQuery || !filters.query) return true;
    const haystack = `${exp.model} strategy ${exp.strategyLabel} ${exp.directory}`.toLowerCase();
    return haystack.includes(filters.query);
  });
}

function getFilteredRows() {
  const filters = currentFilters();
  const rows = [];
  getFilteredExperiments({ ignoreQuery: true }).forEach((exp) => {
    exp.molecules.forEach((row) => {
      if (filters.round !== "all" && row.round !== filters.round) return;
      if (state.status !== "all" && rowStatus(row) !== state.status) return;
      if (filters.query && !rowMatchesSearch(row, filters.query)) return;
      rows.push({ exp, row });
    });
  });
  return rows;
}

function rowMatchesSearch(row, query) {
  const normalizedQuery = String(query ?? "").trim().toLowerCase();
  const normalizedId = String(row.moleculeIndex ?? "").trim().toLowerCase();
  if (/^\d+$/.test(normalizedQuery)) return normalizedId === normalizedQuery;

  const textHaystack = `${normalizedId} ${row.chemicalDescription ?? ""}`.toLowerCase();
  if (textHaystack.includes(normalizedQuery)) return true;

  const canonicalQuery = normalizedQuery.replace(/\s+/g, "");
  const canonicalSmiles = String(row.canonicalSmiles ?? "").replace(/\s+/g, "").toLowerCase();
  return canonicalQuery.length > 0 && canonicalSmiles.includes(canonicalQuery);
}

function initDom() {
  [
    "pageTitle",
    "sidebarResizer",
    "maxFilter",
    "modelFilter",
    "strategyFilter",
    "roundFilter",
    "searchLabel",
    "searchInput",
    "accuracyBars",
    "barCaption",
    "experimentTable",
    "resultTable",
    "detailDrawer",
    "closeDrawer",
    "detailKicker",
    "detailTitle",
    "detailStats",
    "detailTabs",
    "detailBody",
    "compareGrid",
    "compareCaption",
    "compareMoleculeSearch",
    "compareLeftMax",
    "compareLeftModel",
    "compareLeftStrategy",
    "compareLeftRound",
    "compareLeftExperiment",
    "compareRightMax",
    "compareRightModel",
    "compareRightStrategy",
    "compareRightRound",
    "compareRightExperiment",
    "compareLeftContent",
    "compareRightContent",
  ].forEach((id) => {
    els[id] = $(id);
  });

  els.filters = document.querySelector(".filters");
  els.appShell = document.querySelector(".app-shell");

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.status = button.dataset.status;
      renderResults();
    });
  });

  [els.maxFilter, els.modelFilter, els.strategyFilter, els.roundFilter].forEach((select) => {
    select.addEventListener("change", renderAll);
  });
  els.searchInput.addEventListener("input", renderAll);
  document.addEventListener(
    "pointerdown",
    (event) => {
      if (!els.detailDrawer.classList.contains("open")) return;
      if (event.target.closest(".drawer")) return;
      closeDrawer();
    },
    true,
  );
  els.compareMoleculeSearch.addEventListener("input", () => {
    state.compareMoleculeSearch = els.compareMoleculeSearch.value.trim();
    renderCompare();
  });
  ["left", "right"].forEach((side) => {
    ["Max", "Model", "Strategy", "Round", "Experiment"].forEach((field) => {
      const key = field === "Max" ? "maxK" : field.charAt(0).toLowerCase() + field.slice(1);
      const select = els[`compare${capitalize(side)}${field}`];
      select.addEventListener("change", () => {
        state.compareSelections[side][key] = select.value;
        renderCompare();
      });
    });
  });
  els.closeDrawer.addEventListener("click", closeDrawer);
  initSidebarResize();
  setView(state.view);
}

function initFilters() {
  const maxValues = unique(state.data.experiments.map((exp) => String(exp.maxK))).sort(
    (a, b) => Number(a) - Number(b),
  );
  const preferredMax = maxValues.includes("3") ? "3" : maxValues[maxValues.length - 1] || "all";
  optionList(els.maxFilter, maxValues, "All", preferredMax);
  optionList(els.modelFilter, unique(state.data.experiments.map((exp) => exp.model)), "All", "all");
  optionList(
    els.strategyFilter,
    unique(state.data.experiments.map((exp) => exp.strategyLabel)),
    "All",
    "all",
  );
  optionList(
    els.roundFilter,
    unique(state.data.experiments.flatMap((exp) => exp.rounds.map((round) => round.round))),
    "All",
    "all",
  );
  initCompareControls();
}

function initCompareControls() {
  if (!state.compareMoleculeSearch) {
    const firstRow = state.data.experiments.flatMap((exp) => exp.molecules)[0];
    state.compareMoleculeSearch = firstRow?.moleculeIndex || "";
  }
  els.compareMoleculeSearch.value = state.compareMoleculeSearch;

  const defaults = buildCompareDefaults();
  ["left", "right"].forEach((side) => {
    state.compareSelections[side] = {
      maxK: state.compareSelections[side].maxK || defaults[side].maxK,
      model: state.compareSelections[side].model || defaults[side].model,
      strategy: state.compareSelections[side].strategy || defaults[side].strategy,
      round: state.compareSelections[side].round || defaults[side].round,
      experiment: state.compareSelections[side].experiment || defaults[side].experiment,
    };
    renderCompareSelectors(side);
  });
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function buildCompareDefaults() {
  const experiments = [...state.data.experiments].sort((a, b) => {
    const maxDiff = Number(b.maxK) - Number(a.maxK);
    if (maxDiff !== 0) return maxDiff;
    const modelDiff = a.model.localeCompare(b.model);
    if (modelDiff !== 0) return modelDiff;
    return Number(a.strategy) - Number(b.strategy);
  });
  const left = experiments[0];
  const right =
    experiments.find((exp) => left && exp.maxK === left.maxK && exp.model !== left.model) ||
    experiments[1] ||
    left;
  return {
    left: experimentToSelection(left),
    right: experimentToSelection(right),
  };
}

function experimentToSelection(exp) {
  const firstRow = exp?.molecules?.[0];
  return {
    maxK: exp ? String(exp.maxK) : "",
    model: exp?.model || "",
    strategy: exp?.strategyLabel || "",
    round: exp?.rounds?.[0]?.round || "Round 1",
    experiment: firstRow?.conditions?.[0]?.label || "",
  };
}

function renderCompareSelectors(side) {
  const selection = state.compareSelections[side];
  const allExperiments = state.data.experiments;

  const maxValues = unique(allExperiments.map((exp) => String(exp.maxK))).sort((a, b) => Number(a) - Number(b));
  updateSelect(els[`compare${capitalize(side)}Max`], maxValues, selection.maxK);
  selection.maxK = els[`compare${capitalize(side)}Max`].value;

  const modelExperiments = allExperiments.filter((exp) => String(exp.maxK) === selection.maxK);
  const modelValues = unique(modelExperiments.map((exp) => exp.model));
  updateSelect(els[`compare${capitalize(side)}Model`], modelValues, selection.model);
  selection.model = els[`compare${capitalize(side)}Model`].value;

  const strategyExperiments = modelExperiments.filter((exp) => exp.model === selection.model);
  const strategyValues = unique(strategyExperiments.map((exp) => exp.strategyLabel));
  updateSelect(els[`compare${capitalize(side)}Strategy`], strategyValues, selection.strategy);
  selection.strategy = els[`compare${capitalize(side)}Strategy`].value;

  const roundExperiments = strategyExperiments.filter((exp) => exp.strategyLabel === selection.strategy);
  const roundValues = unique(roundExperiments.flatMap((exp) => exp.rounds.map((round) => round.round)));
  updateSelect(els[`compare${capitalize(side)}Round`], roundValues, selection.round || "Round 1");
  selection.round = els[`compare${capitalize(side)}Round`].value;

  const moleculeQuery = (state.compareMoleculeSearch || "").trim().toLowerCase();
  const selectedExperiment = roundExperiments.find((candidate) => candidate.rounds.some((item) => item.round === selection.round));
  const conditionRow = selectedExperiment?.molecules.find(
    (row) => row.round === selection.round && (!moleculeQuery || rowMatchesSearch(row, moleculeQuery)),
  ) || selectedExperiment?.molecules.find((row) => row.round === selection.round);
  const experimentValues = unique((conditionRow?.conditions || []).map((condition) => condition.label));
  updateSelect(els[`compare${capitalize(side)}Experiment`], experimentValues, selection.experiment);
  selection.experiment = els[`compare${capitalize(side)}Experiment`].value;
}

function updateSelect(select, values, preferred) {
  select.innerHTML = values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
  if (values.includes(preferred)) select.value = preferred;
  else select.value = values[0] || "";
}

function initSidebarResize() {
  if (!els.sidebarResizer) return;
  const root = document.documentElement;
  const savedWidth = localStorage.getItem("ragDashboardSidebarWidth");
  if (savedWidth) root.style.setProperty("--sidebar-width", `${savedWidth}px`);

  let resizing = false;
  els.sidebarResizer.addEventListener("pointerdown", (event) => {
    resizing = true;
    document.body.classList.add("sidebar-resizing");
    els.sidebarResizer.setPointerCapture(event.pointerId);
  });
  els.sidebarResizer.addEventListener("pointermove", (event) => {
    if (!resizing) return;
    const width = Math.max(190, Math.min(420, event.clientX));
    root.style.setProperty("--sidebar-width", `${width}px`);
    localStorage.setItem("ragDashboardSidebarWidth", String(width));
  });
  els.sidebarResizer.addEventListener("pointerup", (event) => {
    resizing = false;
    document.body.classList.remove("sidebar-resizing");
    els.sidebarResizer.releasePointerCapture(event.pointerId);
  });
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.id === `${view}View`);
  });
  els.searchLabel?.classList.toggle("hidden", view === "overview");
  els.filters?.classList.toggle("hidden", view === "compare");
  els.filters?.classList.toggle("compare-mode", view === "overview");
  els.pageTitle.textContent = { overview: "Overview", results: "Result Browser", compare: "Model Compare" }[view];
  if (view === "results") renderResults();
  if (view === "compare") renderCompare();
}

function renderAll() {
  renderOverview();
  renderResults();
  if (state.view === "compare") renderCompare();
}

function renderOverview() {
  const experiments = getFilteredExperiments();
  const sorted = [...experiments].sort(
    (a, b) => numberValue(b.averageAccuracy, -1) - numberValue(a.averageAccuracy, -1),
  );
  renderBars(sorted.slice(0, 14));
  renderExperimentTable(experiments);
  return;
  const best = sorted[0];
  els.bestAccuracy.textContent = best ? pct(best.averageAccuracy) : "-";
  els.bestLabel.textContent = best ? `${best.model} · Strategy ${best.strategyLabel} · Max-${best.maxK}` : "-";
  els.experimentCount.textContent = experiments.length;

  renderBars(sorted.slice(0, 14));
  renderExperimentTable(experiments);
}

function renderBars(experiments) {
  els.barCaption.textContent = `${experiments.length} shown`;
  els.accuracyBars.innerHTML = experiments
    .map((exp) => {
      const width = Math.max(2, numberValue(exp.averageAccuracy) * 100);
      return `
        <div class="bar-row" title="${escapeHtml(exp.directory)}">
          <div class="bar-label">${escapeHtml(exp.model)} · Strategy ${escapeHtml(exp.strategyLabel)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${pct(exp.averageAccuracy)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderExperimentTable(experiments) {
  const sorted = [...experiments].sort(
    (a, b) => numberValue(b.averageAccuracy, -1) - numberValue(a.averageAccuracy, -1),
  );
  els.experimentTable.innerHTML = sorted
    .map((exp) => {
      const rounds = ["Round 1", "Round 2", "Round 3"].map((roundName) => {
        const round = exp.rounds.find((item) => item.round === roundName);
        return `<td>${round ? accuracyBadge(round.accuracy) : "-"}</td>`;
      });
      return `
        <tr data-exp="${escapeHtml(exp.id)}">
          <td>Max-${exp.maxK}</td>
          <td>${escapeHtml(exp.model)}</td>
          <td>Strategy ${escapeHtml(exp.strategyLabel)}</td>
          <td>${accuracyBadge(exp.averageAccuracy)}</td>
          ${rounds.join("")}
        </tr>
      `;
    })
    .join("");
  els.experimentTable.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      const exp = state.data.experiments.find((item) => item.id === tr.dataset.exp);
      if (!exp) return;
      els.maxFilter.value = String(exp.maxK);
      els.modelFilter.value = exp.model;
      els.strategyFilter.value = exp.strategyLabel;
      setView("results");
      renderAll();
    });
  });
}

function renderResults() {
  const rows = getFilteredRows().sort((a, b) => {
    const idDiff = Number(a.row.moleculeIndex) - Number(b.row.moleculeIndex);
    if (idDiff !== 0) return idDiff;
    const roundDiff = String(a.row.round).localeCompare(String(b.row.round), undefined, { numeric: true });
    if (roundDiff !== 0) return roundDiff;
    const modelDiff = a.exp.model.localeCompare(b.exp.model);
    if (modelDiff !== 0) return modelDiff;
    return Number(a.exp.strategy) - Number(b.exp.strategy);
  });
  els.resultTable.innerHTML = rows
    .slice(0, 1500)
    .map(({ exp, row }, index) => `
        <tr data-index="${index}">
          <td>${escapeHtml(exp.model)}</td>
          <td>Strategy ${escapeHtml(exp.strategyLabel)}</td>
          <td>${escapeHtml(row.round)}</td>
          <td class="mono">${escapeHtml(row.moleculeIndex)}</td>
          <td><div class="truncate">${escapeHtml(row.chemicalDescription)}</div></td>
          <td>${accuracyBadge(row.accuracy)}</td>
          <td>${row.correct ?? "-"} / ${row.total ?? "-"}</td>
        </tr>
      `,
    )
    .join("");
  els.resultTable.querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => {
      const item = rows[Number(tr.dataset.index)];
      state.compareMoleculeSearch = item.row.moleculeIndex;
      if (els.compareMoleculeSearch) els.compareMoleculeSearch.value = state.compareMoleculeSearch;
      openDetail(item.exp, item.row);
    });
  });
}

function openDetail(exp, row, conditionIndex = 0) {
  state.selectedRow = { exp, row };
  const maxCondition = Math.max(0, (row.conditions || []).length - 1);
  state.selectedCondition = Math.min(Math.max(Number(conditionIndex) || 0, 0), maxCondition);
  els.detailDrawer.classList.add("open");
  els.detailDrawer.setAttribute("aria-hidden", "false");
  els.detailKicker.textContent = `${exp.model} · Strategy ${exp.strategyLabel} · Max-${exp.maxK} · ${row.round}`;
  els.detailTitle.textContent = `${row.moleculeIndex}. ${row.chemicalDescription}`;
  els.detailStats.innerHTML = `
    ${accuracyBadge(row.accuracy)}
    <span class="badge">${row.correct ?? "-"} / ${row.total ?? "-"}</span>
  `;
  renderDetailTabs(row);
  renderDetailBody();
  if (state.view === "compare") renderCompare();
}

function closeDrawer() {
  els.detailDrawer.classList.remove("open");
  els.detailDrawer.setAttribute("aria-hidden", "true");
}

function embeddedRowSummary(row) {
  return {
    round: row.round,
    moleculeIndex: row.moleculeIndex,
    chemicalDescription: row.chemicalDescription,
    canonicalSmiles: row.canonicalSmiles,
    accuracy: row.accuracy,
    correct: row.correct,
    total: row.total,
    retrieval: row.retrieval || {},
  };
}

async function renderDetailBody() {
  const item = state.selectedRow;
  if (!item) {
    els.detailBody.innerHTML = `<div class="empty-state">No row selected.</div>`;
    return;
  }
  const { row } = item;
  const conditions = row.conditions || [];
  if (!row.detailPath || !conditions.length) {
    els.detailBody.innerHTML = `<div class="empty-state">No experiment-condition details are available.</div>`;
    return;
  }
  const index = Math.min(state.selectedCondition, conditions.length - 1);
  els.detailBody.innerHTML = `<iframe class="detail-frame" src="${escapeHtml(`${row.detailPath}#condition-${index}`)}" title="Experiment output and input"></iframe>`;
}

function renderDetailTabs(row) {
  const conditions = row.conditions || [];
  els.detailTabs.innerHTML = conditions
    .map((condition, index) => `<button class="tab${index === state.selectedCondition ? " active" : ""}" data-condition="${index}">${escapeHtml(condition.label)}</button>`)
    .join("");
  els.detailTabs.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedCondition = Number(button.dataset.condition);
      renderDetailTabs(row);
      renderDetailBody();
    });
  });
}

async function renderCompareLegacyUnused() {
  if (!state.data || !els.compareGrid) return;
  const renderId = ++state.compareRenderId;
  const filters = currentFilters();
  const moleculeQuery = (state.compareMoleculeSearch || els.compareMoleculeSearch.value || "").trim().toLowerCase();
  if (!moleculeQuery) {
    els.compareCaption.textContent = "Enter an ID, molecule name, or SMILES";
    els.compareGrid.innerHTML = `<div class="empty-state">Enter a molecule search term.</div>`;
    return;
  }

  const selectedModels = state.compareModels;
  const matches = getFilteredExperiments({ ignoreModel: true, ignoreQuery: true })
    .filter((exp) => selectedModels.has(exp.model))
    .flatMap((exp) =>
      exp.molecules
        .filter(
          (candidate) =>
            rowMatchesSearch(candidate, moleculeQuery) &&
            (filters.round === "all" || candidate.round === filters.round),
        )
        .map((candidate) => ({ exp, row: candidate })),
    )
    .sort((a, b) => {
      const idDiff = a.exp.model.localeCompare(b.exp.model);
      if (idDiff !== 0) return idDiff;
      const strategyDiff = Number(a.exp.strategy) - Number(b.exp.strategy);
      if (strategyDiff !== 0) return strategyDiff;
      return String(a.row.round).localeCompare(String(b.row.round), undefined, { numeric: true });
    });

  els.compareCaption.textContent = `search "${state.compareMoleculeSearch || els.compareMoleculeSearch.value}" · ${matches.length} results`;
  if (matches.length === 0) {
    els.compareGrid.innerHTML = `<div class="empty-state">No matching rows for the current filters.</div>`;
    return;
  }

  els.compareGrid.innerHTML = `<div class="empty-state">Loading parsed JSON previews...</div>`;
  const cards = await Promise.all(
    matches.map(async ({ exp, row }) => {
      const parsedText = await loadParsedPreview(row);
      return `
        <article class="compare-card">
          <h4>${escapeHtml(exp.model)}</h4>
          <span>Strategy ${escapeHtml(exp.strategyLabel)} · Max-${exp.maxK} · ${escapeHtml(row.round)}</span>
          <div>${accuracyBadge(row.accuracy)} <span class="badge">${row.correct ?? "-"} / ${row.total ?? "-"}</span></div>
          <pre class="preview-block">${escapeHtml(parsedText)}</pre>
          <button class="ghost-button compare-open" data-exp="${escapeHtml(exp.id)}" data-round="${escapeHtml(row.round)}" data-id="${escapeHtml(row.moleculeIndex)}">Open full outputs</button>
        </article>
      `;
    }),
  );
  if (renderId !== state.compareRenderId) return;
  els.compareGrid.innerHTML = cards.join("");
  els.compareGrid.querySelectorAll(".compare-open").forEach((button) => {
    button.addEventListener("click", () => {
      const exp = state.data.experiments.find((item) => item.id === button.dataset.exp);
      const nextRow = exp?.molecules.find(
        (candidate) => candidate.round === button.dataset.round && candidate.moleculeIndex === button.dataset.id,
      );
      if (exp && nextRow) openDetail(exp, nextRow, Number(button.dataset.condition));
    });
  });
}

async function loadParsedPreview(row, conditionLabel = "") {
  const condition = (row.conditions || []).find((item) => item.label === conditionLabel) || row.conditions?.[0];
  return condition?.json || JSON.stringify(embeddedRowSummary(row), null, 2);
}

async function renderCompare() {
  if (!state.data || !els.compareGrid) return;
  const renderId = ++state.compareRenderId;
  const moleculeQuery = (state.compareMoleculeSearch || els.compareMoleculeSearch.value || "").trim().toLowerCase();
  ["left", "right"].forEach((side) => renderCompareSelectors(side));

  if (!moleculeQuery) {
    els.compareCaption.textContent = "Enter an ID, molecule name, or SMILES";
    els.compareLeftContent.innerHTML = `<div class="empty-state">Enter a molecule search term.</div>`;
    els.compareRightContent.innerHTML = `<div class="empty-state">Enter a molecule search term.</div>`;
    return;
  }

  els.compareCaption.textContent = `search "${state.compareMoleculeSearch || els.compareMoleculeSearch.value}"`;
  els.compareLeftContent.innerHTML = `<div class="empty-state">Loading parsed JSON preview...</div>`;
  els.compareRightContent.innerHTML = `<div class="empty-state">Loading parsed JSON preview...</div>`;

  const [leftHtml, rightHtml] = await Promise.all([
    renderCompareSide("left", moleculeQuery),
    renderCompareSide("right", moleculeQuery),
  ]);
  if (renderId !== state.compareRenderId) return;
  els.compareLeftContent.innerHTML = leftHtml;
  els.compareRightContent.innerHTML = rightHtml;
  els.compareGrid.querySelectorAll(".compare-open").forEach((button) => {
    button.addEventListener("click", () => {
      const exp = state.data.experiments.find((item) => item.id === button.dataset.exp);
      const nextRow = exp?.molecules.find(
        (candidate) => candidate.round === button.dataset.round && candidate.moleculeIndex === button.dataset.id,
      );
      if (exp && nextRow) openDetail(exp, nextRow);
    });
  });
}

async function renderCompareSide(side, moleculeQuery) {
  const item = findCompareItem(side, moleculeQuery);
  if (!item) {
    return `<div class="empty-state">No matching row for this Max-k / model / strategy / round.</div>`;
  }
  const { exp, row, matchCount } = item;
  const selectedExperiment = state.compareSelections[side].experiment;
  const conditionIndex = Math.max(0, row.conditions.findIndex((condition) => condition.label === selectedExperiment));
  const parsedText = await loadParsedPreview(row, selectedExperiment);
  const matchNote =
    matchCount > 1
      ? `<span class="badge warn">${matchCount} molecule matches, showing ID ${escapeHtml(row.moleculeIndex)}</span>`
      : `<span class="badge">ID ${escapeHtml(row.moleculeIndex)}</span>`;
  return `
    <article class="compare-card">
      <h4>${escapeHtml(row.chemicalDescription)}</h4>
      <span class="compare-card-meta">${escapeHtml(exp.model)} · Strategy ${escapeHtml(exp.strategyLabel)} · Max-${exp.maxK} · ${escapeHtml(row.round)}</span>
      <div>${accuracyBadge(row.accuracy)} <span class="badge">${row.correct ?? "-"} / ${row.total ?? "-"}</span> ${matchNote}</div>
      <pre class="preview-block">${escapeHtml(parsedText)}</pre>
      <button class="ghost-button compare-open" data-exp="${escapeHtml(exp.id)}" data-round="${escapeHtml(row.round)}" data-id="${escapeHtml(row.moleculeIndex)}" data-condition="${conditionIndex}">Open full outputs</button>
    </article>
  `;
}

function findCompareItem(side, moleculeQuery) {
  const selection = state.compareSelections[side];
  const exp = state.data.experiments.find(
    (candidate) =>
      String(candidate.maxK) === selection.maxK &&
      candidate.model === selection.model &&
      candidate.strategyLabel === selection.strategy,
  );
  if (!exp) return null;
  const matches = exp.molecules
    .filter((row) => row.round === selection.round && rowMatchesSearch(row, moleculeQuery))
    .sort((a, b) => Number(a.moleculeIndex) - Number(b.moleculeIndex));
  if (!matches.length) return null;
  return { exp, row: matches[0], matchCount: matches.length };
}

async function loadData() {
  const embedded = document.getElementById("dashboard-data");
  if (!embedded) throw new Error("Embedded dashboard data is missing.");
  state.data = JSON.parse(embedded.textContent);
  initFilters();
  renderAll();
}

initDom();
loadData().catch((error) => {
  document.body.innerHTML = `<main class="app-shell"><section class="panel"><h2>Could not load data</h2><p>${escapeHtml(
    error.message,
  )}</p><p>Open the packaged index.html file directly in a modern browser.</p></section></main>`;
});
