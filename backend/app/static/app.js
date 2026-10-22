/**
 * ModelMD - Frontend Application Controller
 * Dual Mode: Native File Upload & Direct Local Path / Project File Selector
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const navTabs = document.querySelectorAll(".nav-tab");
  const moduleViews = document.querySelectorAll(".module-view");

  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      navTabs.forEach((t) => t.classList.remove("active"));
      moduleViews.forEach((v) => v.classList.add("hidden"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-target");
      const targetView = document.getElementById(targetId);
      if (targetView) targetView.classList.remove("hidden");
    });
  });

  // Global Error Handling
  const errorBanner = document.getElementById("error-banner");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");
  const errorDismiss = document.getElementById("error-dismiss");

  function showError(title, message) {
    errorTitle.textContent = title || "Diagnostic Error";
    errorMessage.textContent = message || "An unexpected error occurred.";
    errorBanner.classList.remove("hidden");
    errorBanner.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function hideError() {
    errorBanner.classList.add("hidden");
  }

  errorDismiss.addEventListener("click", hideError);

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Setup Dropzone Drag & Drop
  function setupDropzone(dropzone, input, onFileSelected) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        onFileSelected(e.dataTransfer.files[0]);
      }
    });

    input.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        onFileSelected(e.target.files[0]);
      }
    });
  }

  // Dual Mode Tabs (Upload vs Path)
  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", (e) => {
      const cardType = tab.getAttribute("data-card");
      const mode = tab.getAttribute("data-target-mode");

      // Update tabs in this card
      document.querySelectorAll(`.mode-tab[data-card="${cardType}"]`).forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const dropzone = document.getElementById(`${cardType}-dropzone`);
      const panel = document.getElementById(`${cardType}-path-panel`);

      if (mode === "upload") {
        if (dropzone) dropzone.classList.remove("hidden");
        if (panel) panel.classList.add("hidden");
      } else {
        if (dropzone) dropzone.classList.add("hidden");
        if (panel) panel.classList.remove("hidden");
      }
    });
  });

  // Auto-Discover Workspace Files
  async function loadWorkspaceFiles() {
    try {
      const resp = await fetch("/api/files/browse");
      if (!resp.ok) return;
      const data = await resp.json();

      // Populate Model Selectors
      const modelSelects = [
        document.getElementById("model-project-select"),
      ];
      modelSelects.forEach((sel) => {
        if (!sel) return;
        data.models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.path;
          opt.textContent = `${m.filename} (${m.size}) - ${m.path}`;
          sel.appendChild(opt);
        });
      });

      // Populate Dataset Selectors
      const datasetSelects = [
        document.getElementById("dataset-project-select"),
        document.getElementById("ref-project-select"),
        document.getElementById("curr-project-select"),
      ];
      datasetSelects.forEach((sel) => {
        if (!sel) return;
        data.datasets.forEach((d) => {
          const opt = document.createElement("option");
          opt.value = d.path;
          opt.textContent = `${d.filename} (${d.size}) - ${d.path}`;
          sel.appendChild(opt);
        });
      });
    } catch (e) {
      console.warn("Could not auto-browse workspace files:", e);
    }
  }

  loadWorkspaceFiles();

  // ========================================================
  // MODULE 01 LOGIC: BASELINE DIAGNOSTICS
  // ========================================================
  const m1 = {
    modelFile: null,
    modelPath: null,
    datasetFile: null,
    datasetPath: null,
    datasetMeta: null,
    results: null,
    filterOnlyErrors: true,
  };

  const modelInput = document.getElementById("model-file-input");
  const modelDropzone = document.getElementById("model-dropzone");
  const modelInfo = document.getElementById("model-info");
  const modelFilename = document.getElementById("model-filename");
  const modelFilesize = document.getElementById("model-filesize");
  const modelMetaNote = document.getElementById("model-meta-note");
  const modelProjectSelect = document.getElementById("model-project-select");
  const modelPathInput = document.getElementById("model-path-input");
  const btnLoadModelPath = document.getElementById("btn-load-model-path");

  const datasetInput = document.getElementById("dataset-file-input");
  const datasetDropzone = document.getElementById("dataset-dropzone");
  const datasetStats = document.getElementById("dataset-stats");
  const statRows = document.getElementById("stat-rows");
  const statCols = document.getElementById("stat-cols");
  const statMissing = document.getElementById("stat-missing");
  const statDuplicates = document.getElementById("stat-duplicates");
  const datasetProjectSelect = document.getElementById("dataset-project-select");
  const datasetPathInput = document.getElementById("dataset-path-input");
  const btnLoadDatasetPath = document.getElementById("btn-load-dataset-path");

  const previewSection = document.getElementById("preview-section");
  const previewFilenameBadge = document.getElementById("preview-filename-badge");
  const previewThead = document.getElementById("preview-thead");
  const previewTbody = document.getElementById("preview-tbody");

  const targetSelect = document.getElementById("target-column-select");
  const averageSelect = document.getElementById("average-method-select");
  const missingSelect = document.getElementById("missing-strategy-select");

  const runBtn = document.getElementById("run-btn");
  const runBtnText = document.getElementById("run-btn-text");
  const runSpinner = document.getElementById("run-spinner");
  const resetBtn = document.getElementById("reset-btn");

  const resultsSection = document.getElementById("results-section");
  const summaryTotalRows = document.getElementById("summary-total-rows");
  const summaryCorrect = document.getElementById("summary-correct");
  const summaryIncorrect = document.getElementById("summary-incorrect");
  const summaryDropped = document.getElementById("summary-dropped");
  const summaryDroppedItem = document.getElementById("summary-dropped-item");

  const metricAccuracy = document.getElementById("metric-accuracy");
  const metricPrecision = document.getElementById("metric-precision");
  const metricRecall = document.getElementById("metric-recall");
  const metricF1 = document.getElementById("metric-f1");
  const metricPrecSub = document.getElementById("metric-prec-sub");
  const metricRecSub = document.getElementById("metric-rec-sub");
  const metricF1Sub = document.getElementById("metric-f1-sub");

  const cmContainer = document.getElementById("cm-container");
  const classReportTbody = document.getElementById("class-report-tbody");

  const inspectorTbody = document.getElementById("inspector-tbody");
  const incorrectCountBadge = document.getElementById("incorrect-count-badge");
  const allCountBadge = document.getElementById("all-count-badge");
  const toggleFilterErrors = document.getElementById("toggle-filter-errors");
  const toggleFilterAll = document.getElementById("toggle-filter-all");

  function checkCanRunM1() {
    const hasModel = !!(m1.modelFile || m1.modelPath);
    const hasDataset = !!(m1.datasetFile || m1.datasetPath);
    runBtn.disabled = !(hasModel && hasDataset && targetSelect.value);
  }

  // File Upload Handlers
  async function handleModelFile(file) {
    hideError();
    const validExts = [".joblib", ".pkl"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!validExts.includes(ext)) {
      showError("Invalid Model Artifact", `File '${file.name}' must be a .joblib or .pkl artifact.`);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("/api/model/inspect", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Failed to inspect model.");

      m1.modelFile = file;
      m1.modelPath = null;
      modelFilename.textContent = data.filename;
      modelFilesize.textContent = data.size_readable;
      modelMetaNote.textContent = data.security_warning;
      modelInfo.classList.remove("hidden");
      checkCanRunM1();
    } catch (err) {
      showError("Model Inspection Failed", err.message);
    }
  }

  async function handleModelPath(path) {
    hideError();
    if (!path || !path.trim()) {
      showError("Missing Path", "Please enter or select a model file path.");
      return;
    }
    try {
      const resp = await fetch("/api/model/inspect-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: path.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Failed to inspect model at path.");

      m1.modelPath = data.file_path;
      m1.modelFile = null;
      modelFilename.textContent = data.filename;
      modelFilesize.textContent = data.size_readable;
      modelMetaNote.textContent = "Path: " + data.file_path;
      modelInfo.classList.remove("hidden");
      checkCanRunM1();
    } catch (err) {
      showError("Model Path Inspection Failed", err.message);
    }
  }

  modelProjectSelect.addEventListener("change", () => {
    if (modelProjectSelect.value) {
      modelPathInput.value = modelProjectSelect.value;
      handleModelPath(modelProjectSelect.value);
    }
  });

  btnLoadModelPath.addEventListener("click", () => {
    handleModelPath(modelPathInput.value);
  });
  modelPathInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleModelPath(modelPathInput.value);
  });

  async function applyDatasetData(data, isPath = false) {
    m1.datasetMeta = data;
    statRows.textContent = data.row_count.toLocaleString();
    statCols.textContent = data.column_count;
    statMissing.textContent = data.total_missing_values;
    statDuplicates.textContent = data.duplicate_rows;
    datasetStats.classList.remove("hidden");

    targetSelect.innerHTML = "";
    let defaultTarget = "";
    const likelyTargets = ["target", "label", "outcome", "class", "diagnosis", "y"];

    data.columns.forEach((col) => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = `${col} (${data.column_types[col] || "object"})`;
      targetSelect.appendChild(opt);

      if (!defaultTarget && likelyTargets.includes(col.toLowerCase())) {
        defaultTarget = col;
      }
    });

    if (!defaultTarget && data.columns.length > 0) {
      defaultTarget = data.columns[data.columns.length - 1];
    }
    targetSelect.value = defaultTarget;
    targetSelect.disabled = false;

    renderPreviewTable(data);
    checkCanRunM1();
  }

  async function handleDatasetFile(file) {
    hideError();
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showError("Invalid Dataset Format", `File '${file.name}' must be a CSV (.csv) file.`);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("/api/dataset/inspect", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Failed to inspect CSV dataset.");

      m1.datasetFile = file;
      m1.datasetPath = null;
      applyDatasetData(data, false);
    } catch (err) {
      showError("Dataset Inspection Failed", err.message);
    }
  }

  async function handleDatasetPath(path) {
    hideError();
    if (!path || !path.trim()) {
      showError("Missing Path", "Please enter or select a dataset file path.");
      return;
    }
    try {
      const resp = await fetch("/api/dataset/inspect-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: path.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Failed to inspect dataset at path.");

      m1.datasetPath = data.file_path;
      m1.datasetFile = null;
      applyDatasetData(data, true);
    } catch (err) {
      showError("Dataset Path Inspection Failed", err.message);
    }
  }

  datasetProjectSelect.addEventListener("change", () => {
    if (datasetProjectSelect.value) {
      datasetPathInput.value = datasetProjectSelect.value;
      handleDatasetPath(datasetProjectSelect.value);
    }
  });

  btnLoadDatasetPath.addEventListener("click", () => {
    handleDatasetPath(datasetPathInput.value);
  });
  datasetPathInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleDatasetPath(datasetPathInput.value);
  });

  function renderPreviewTable(data) {
    previewFilenameBadge.textContent = data.filename;
    previewThead.innerHTML = "";
    previewTbody.innerHTML = "";

    const trHead = document.createElement("tr");
    data.columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      trHead.appendChild(th);
    });
    previewThead.appendChild(trHead);

    data.preview.forEach((row) => {
      const tr = document.createElement("tr");
      data.columns.forEach((col) => {
        const td = document.createElement("td");
        td.textContent = row[col] !== undefined ? String(row[col]) : "";
        tr.appendChild(td);
      });
      previewTbody.appendChild(tr);
    });

    previewSection.classList.remove("hidden");
  }

  targetSelect.addEventListener("change", checkCanRunM1);
  setupDropzone(modelDropzone, modelInput, handleModelFile);
  setupDropzone(datasetDropzone, datasetInput, handleDatasetFile);

  // Run Baseline Diagnostics (Handles both Uploaded files and Direct Paths!)
  runBtn.addEventListener("click", async () => {
    if ((!m1.modelFile && !m1.modelPath) || (!m1.datasetFile && !m1.datasetPath) || !targetSelect.value) return;

    hideError();
    runBtn.disabled = true;
    runSpinner.classList.remove("hidden");
    runBtnText.textContent = "Executing Inference...";

    try {
      let resp, data;

      if (m1.modelPath || m1.datasetPath) {
        // Path mode execution
        resp = await fetch("/api/diagnostics/run-path", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_path: m1.modelPath || m1.modelFile.name,
            dataset_path: m1.datasetPath || m1.datasetFile.name,
            target_column: targetSelect.value,
            average_method: averageSelect.value,
            missing_strategy: missingSelect.value,
          }),
        });
      } else {
        // Multipart upload execution
        const formData = new FormData();
        formData.append("model_file", m1.modelFile);
        formData.append("dataset_file", m1.datasetFile);
        formData.append("target_column", targetSelect.value);
        formData.append("average_method", averageSelect.value);
        formData.append("missing_strategy", missingSelect.value);
        resp = await fetch("/api/diagnostics/run", { method: "POST", body: formData });
      }

      data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Diagnostic run failed.");

      m1.results = data;
      renderDiagnosticResults(data);
      resultsSection.classList.remove("hidden");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError("Diagnostic Run Error", err.message);
    } finally {
      runBtn.disabled = false;
      runSpinner.classList.add("hidden");
      runBtnText.textContent = "Run Baseline Diagnostics";
      checkCanRunM1();
    }
  });

  function renderDiagnosticResults(data) {
    const s = data.summary;
    summaryTotalRows.textContent = s.total_evaluated_rows.toLocaleString();
    summaryCorrect.textContent = s.correct_predictions.toLocaleString();
    summaryIncorrect.textContent = s.incorrect_predictions.toLocaleString();
    summaryDropped.textContent = s.dropped_rows_count.toLocaleString();

    if (s.dropped_rows_count > 0) summaryDroppedItem.classList.remove("hidden");
    else summaryDroppedItem.classList.add("hidden");

    metricAccuracy.textContent = (s.accuracy * 100).toFixed(1) + "%";
    metricPrecision.textContent = (s.precision * 100).toFixed(1) + "%";
    metricRecall.textContent = (s.recall * 100).toFixed(1) + "%";
    metricF1.textContent = (s.f1_score * 100).toFixed(1) + "%";

    const avgLabel = `${s.average_method} average`;
    metricPrecSub.textContent = avgLabel;
    metricRecSub.textContent = avgLabel;
    metricF1Sub.textContent = avgLabel;

    renderConfusionMatrix(data.confusion_matrix);
    renderClassReport(data.per_class_metrics);

    incorrectCountBadge.textContent = s.incorrect_predictions;
    allCountBadge.textContent = s.total_evaluated_rows;
    m1.filterOnlyErrors = s.incorrect_predictions > 0;
    updateInspectorFilterButtons();
    renderRowInspector();
  }

  function renderConfusionMatrix(cmData) {
    cmContainer.innerHTML = "";
    if (!cmData || !cmData.matrix) return;

    const labels = cmData.labels;
    const matrix = cmData.matrix;
    const table = document.createElement("table");
    table.className = "cm-table";

    const topRow = document.createElement("tr");
    const emptyTh = document.createElement("th");
    emptyTh.innerHTML = "<span style='font-size:0.7rem; color:var(--text-muted);'>Actual \\ Pred</span>";
    topRow.appendChild(emptyTh);

    labels.forEach((label) => {
      const th = document.createElement("th");
      th.textContent = `Pred: ${label}`;
      topRow.appendChild(th);
    });
    table.appendChild(topRow);

    matrix.forEach((row, i) => {
      const tr = document.createElement("tr");
      const rowTh = document.createElement("th");
      rowTh.textContent = `True: ${labels[i]}`;
      tr.appendChild(rowTh);

      row.forEach((val, j) => {
        const td = document.createElement("td");
        td.className = "cm-cell";
        td.textContent = val;
        td.classList.add(i === j ? (val > 0 ? "correct" : "zero") : (val > 0 ? "error" : "zero"));
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });

    cmContainer.appendChild(table);
  }

  function renderClassReport(classes) {
    classReportTbody.innerHTML = "";
    if (!classes) return;

    classes.forEach((c) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="font-weight: 600; font-family: var(--font-mono);">${escapeHtml(c.class_label)}</td>
        <td>${(c.precision * 100).toFixed(1)}%</td>
        <td>${(c.recall * 100).toFixed(1)}%</td>
        <td>${(c.f1_score * 100).toFixed(1)}%</td>
        <td>${c.support.toLocaleString()}</td>
      `;
      classReportTbody.appendChild(tr);
    });
  }

  function renderRowInspector() {
    inspectorTbody.innerHTML = "";
    if (!m1.results || !m1.results.row_level_predictions) return;

    const rows = m1.results.row_level_predictions;
    const displayedRows = m1.filterOnlyErrors ? rows.filter((r) => !r.is_correct) : rows;

    if (displayedRows.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">
          ${m1.filterOnlyErrors ? "No prediction errors found! 100% accuracy on evaluated dataset." : "No rows to display."}
        </td>
      `;
      inspectorTbody.appendChild(tr);
      return;
    }

    displayedRows.forEach((item) => {
      const tr = document.createElement("tr");
      const badgeClass = item.is_correct ? "correct" : "incorrect";
      const badgeText = item.is_correct ? "Correct" : "Error";

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); color: var(--text-muted);">Row ${item.row_number}</td>
        <td style="font-family: var(--font-mono); font-weight: 600;">${escapeHtml(item.actual)}</td>
        <td style="font-family: var(--font-mono); color: ${item.is_correct ? "var(--color-success)" : "var(--color-danger)"}; font-weight: 600;">${escapeHtml(item.predicted)}</td>
        <td><span class="badge-status ${badgeClass}">${badgeText}</span></td>
      `;
      inspectorTbody.appendChild(tr);
    });
  }

  function updateInspectorFilterButtons() {
    if (m1.filterOnlyErrors) {
      toggleFilterErrors.classList.add("active");
      toggleFilterAll.classList.remove("active");
    } else {
      toggleFilterErrors.classList.remove("active");
      toggleFilterAll.classList.add("active");
    }
  }

  toggleFilterErrors.addEventListener("click", () => {
    m1.filterOnlyErrors = true;
    updateInspectorFilterButtons();
    renderRowInspector();
  });

  toggleFilterAll.addEventListener("click", () => {
    m1.filterOnlyErrors = false;
    updateInspectorFilterButtons();
    renderRowInspector();
  });

  resetBtn.addEventListener("click", () => {
    m1.modelFile = null;
    m1.modelPath = null;
    m1.datasetFile = null;
    m1.datasetPath = null;
    m1.datasetMeta = null;
    m1.results = null;

    modelInput.value = "";
    modelPathInput.value = "";
    modelProjectSelect.value = "";
    datasetInput.value = "";
    datasetPathInput.value = "";
    datasetProjectSelect.value = "";

    modelInfo.classList.add("hidden");
    datasetStats.classList.add("hidden");
    previewSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    hideError();

    targetSelect.innerHTML = "<option value=''>Upload or select a dataset first</option>";
    targetSelect.disabled = true;
    checkCanRunM1();
  });

  // ========================================================
  // MODULE 02 LOGIC: HEALTH & DATA DRIFT
  // ========================================================
  const m2 = {
    refFile: null,
    refPath: null,
    currFile: null,
    currPath: null,
    modelFile: null,
    modelPath: null,
  };

  const refInput = document.getElementById("ref-file-input");
  const refDropzone = document.getElementById("ref-dropzone");
  const refInfo = document.getElementById("ref-info");
  const refFilename = document.getElementById("ref-filename");
  const refFilesize = document.getElementById("ref-filesize");
  const refProjectSelect = document.getElementById("ref-project-select");
  const refPathInput = document.getElementById("ref-path-input");
  const btnLoadRefPath = document.getElementById("btn-load-ref-path");

  const currInput = document.getElementById("curr-file-input");
  const currDropzone = document.getElementById("curr-dropzone");
  const currInfo = document.getElementById("curr-info");
  const currFilename = document.getElementById("curr-filename");
  const currFilesize = document.getElementById("curr-filesize");
  const currProjectSelect = document.getElementById("curr-project-select");
  const currPathInput = document.getElementById("curr-path-input");
  const btnLoadCurrPath = document.getElementById("btn-load-curr-path");

  const driftModelInput = document.getElementById("drift-model-input");
  const driftTargetSelect = document.getElementById("drift-target-select");
  const driftPvalueSelect = document.getElementById("drift-pvalue-select");

  const runDriftBtn = document.getElementById("run-drift-btn");
  const runDriftText = document.getElementById("run-drift-text");
  const driftSpinner = document.getElementById("drift-spinner");
  const resetDriftBtn = document.getElementById("reset-drift-btn");

  const driftResultsSection = document.getElementById("drift-results-section");
  const driftHealthBanner = document.getElementById("drift-health-banner");
  const driftHealthBadge = document.getElementById("drift-health-badge");
  const driftHealthDesc = document.getElementById("drift-health-desc");

  const perfDegradationCard = document.getElementById("perf-degradation-card");
  const perfBaseAcc = document.getElementById("perf-base-acc");
  const perfCurrAcc = document.getElementById("perf-curr-acc");
  const perfAccDelta = document.getElementById("perf-acc-delta");
  const perfAccDeltaSub = document.getElementById("perf-acc-delta-sub");
  const perfF1Delta = document.getElementById("perf-f1-delta");
  const perfF1DeltaSub = document.getElementById("perf-f1-delta-sub");

  const driftFeaturesTbody = document.getElementById("drift-features-tbody");

  function checkCanRunM2() {
    const hasRef = !!(m2.refFile || m2.refPath);
    const hasCurr = !!(m2.currFile || m2.currPath);
    runDriftBtn.disabled = !(hasRef && hasCurr);
  }

  setupDropzone(refDropzone, refInput, (file) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showError("Invalid File", "Reference dataset must be a CSV file.");
      return;
    }
    m2.refFile = file;
    m2.refPath = null;
    refFilename.textContent = file.name;
    refFilesize.textContent = (file.size / 1024).toFixed(1) + " KB";
    refInfo.classList.remove("hidden");
    checkCanRunM2();
  });

  function handleRefPath(path) {
    if (!path || !path.trim()) return;
    m2.refPath = path.trim();
    m2.refFile = null;
    refFilename.textContent = path.split("/").pop().split("\\").pop();
    refFilesize.textContent = "Path: " + path;
    refInfo.classList.remove("hidden");
    checkCanRunM2();
  }

  refProjectSelect.addEventListener("change", () => {
    if (refProjectSelect.value) {
      refPathInput.value = refProjectSelect.value;
      handleRefPath(refProjectSelect.value);
    }
  });
  btnLoadRefPath.addEventListener("click", () => handleRefPath(refPathInput.value));

  setupDropzone(currDropzone, currInput, (file) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showError("Invalid File", "Current dataset must be a CSV file.");
      return;
    }
    m2.currFile = file;
    m2.currPath = null;
    currFilename.textContent = file.name;
    currFilesize.textContent = (file.size / 1024).toFixed(1) + " KB";
    currInfo.classList.remove("hidden");
    checkCanRunM2();
  });

  function handleCurrPath(path) {
    if (!path || !path.trim()) return;
    m2.currPath = path.trim();
    m2.currFile = null;
    currFilename.textContent = path.split("/").pop().split("\\").pop();
    currFilesize.textContent = "Path: " + path;
    currInfo.classList.remove("hidden");
    checkCanRunM2();
  }

  currProjectSelect.addEventListener("change", () => {
    if (currProjectSelect.value) {
      currPathInput.value = currProjectSelect.value;
      handleCurrPath(currProjectSelect.value);
    }
  });
  btnLoadCurrPath.addEventListener("click", () => handleCurrPath(currPathInput.value));

  driftModelInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      m2.modelFile = e.target.files[0];
    }
  });

  runDriftBtn.addEventListener("click", async () => {
    if ((!m2.refFile && !m2.refPath) || (!m2.currFile && !m2.currPath)) return;

    hideError();
    runDriftBtn.disabled = true;
    driftSpinner.classList.remove("hidden");
    runDriftText.textContent = "Analyzing Statistical Drift...";

    try {
      let resp, data;

      if (m2.refPath || m2.currPath) {
        resp = await fetch("/api/drift/analyze-path", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ref_path: m2.refPath || m2.refFile.name,
            curr_path: m2.currPath || m2.currFile.name,
            target_column: driftTargetSelect.value.trim() || null,
            model_path: m2.modelPath || (m2.modelFile ? m2.modelFile.name : null),
            p_value_threshold: parseFloat(driftPvalueSelect.value),
          }),
        });
      } else {
        const formData = new FormData();
        formData.append("reference_file", m2.refFile);
        formData.append("current_file", m2.currFile);
        if (driftTargetSelect.value.trim()) {
          formData.append("target_column", driftTargetSelect.value.trim());
        }
        if (m2.modelFile) {
          formData.append("model_file", m2.modelFile);
        }
        formData.append("p_value_threshold", driftPvalueSelect.value);
        resp = await fetch("/api/drift/analyze", { method: "POST", body: formData });
      }

      data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Drift analysis failed.");

      renderDriftResults(data);
      driftResultsSection.classList.remove("hidden");
      driftResultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError("Drift Analysis Error", err.message);
    } finally {
      runDriftBtn.disabled = false;
      driftSpinner.classList.add("hidden");
      runDriftText.textContent = "Analyze Data Drift & Health";
      checkCanRunM2();
    }
  });

  function renderDriftResults(data) {
    const health = data.dataset_health;

    driftHealthBadge.textContent = health.health_status;
    driftHealthBadge.className = `health-badge ${health.status_color}`;
    driftHealthDesc.textContent = `${health.drifted_features_count} of ${health.total_features} eligible feature(s) (${health.drift_share_pct}%) exhibit significant distribution drift (threshold p < ${health.p_value_threshold_used || 0.05}). Reference: ${health.ref_row_count} rows | Current: ${health.curr_row_count} rows.`;

    if (data.performance_comparison && data.performance_comparison.accuracy) {
      const perf = data.performance_comparison;
      perfBaseAcc.textContent = (perf.accuracy.baseline * 100).toFixed(1) + "%";
      perfCurrAcc.textContent = (perf.accuracy.current * 100).toFixed(1) + "%";

      const deltaAcc = perf.accuracy.delta_pct;
      perfAccDelta.textContent = (deltaAcc > 0 ? "+" : "") + deltaAcc.toFixed(1) + "%";
      perfAccDelta.style.color = deltaAcc < 0 ? "var(--color-danger)" : "var(--color-success)";
      perfAccDeltaSub.textContent = deltaAcc < -5.0 ? "Severe Degradation" : "Stable";

      const deltaF1 = perf.f1_score.delta_pct;
      perfF1Delta.textContent = (deltaF1 > 0 ? "+" : "") + deltaF1.toFixed(1) + "%";
      perfF1Delta.style.color = deltaF1 < 0 ? "var(--color-danger)" : "var(--color-success)";

      perfDegradationCard.classList.remove("hidden");
    } else {
      perfDegradationCard.classList.add("hidden");
    }

    driftFeaturesTbody.innerHTML = "";
    const features = data.features;

    Object.keys(features).forEach((featName) => {
      const f = features[featName];
      const tr = document.createElement("tr");

      const sevClass = f.severity.toLowerCase();
      const meanShift = f.mean_shift_pct !== undefined ? (f.mean_shift_pct > 0 ? `+${f.mean_shift_pct}%` : `${f.mean_shift_pct}%`) : "N/A";
      const warningsText = f.warnings && f.warnings.length > 0 ? f.warnings.join(" ") : "None";

      tr.innerHTML = `
        <td style="font-weight: 600; font-family: var(--font-mono); color: var(--text-primary);">${escapeHtml(featName)}</td>
        <td><span class="file-badge">${f.feature_type}</span></td>
        <td>${escapeHtml(f.test_used)}</td>
        <td style="font-family: var(--font-mono); color: ${f.drift_detected ? 'var(--color-danger)' : 'var(--text-secondary)'}; font-weight: ${f.drift_detected ? '600' : 'normal'};">${escapeHtml(f.p_value_formatted || (f.p_value < 0.0001 ? f.p_value.toExponential(2) : f.p_value.toFixed(4)))}</td>
        <td><span class="severity-pill ${sevClass}">${f.severity}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 600; color: ${Math.abs(f.mean_shift_pct || 0) > 15 ? "var(--color-warning)" : "inherit"};">${meanShift}</td>
        <td style="font-size: 0.75rem; color: ${f.warnings && f.warnings.length > 0 ? 'var(--color-warning)' : 'var(--text-muted)'}">${escapeHtml(warningsText)}</td>
      `;
      driftFeaturesTbody.appendChild(tr);
    });
  }

  resetDriftBtn.addEventListener("click", () => {
    m2.refFile = null;
    m2.refPath = null;
    m2.currFile = null;
    m2.currPath = null;
    m2.modelFile = null;
    m2.modelPath = null;

    refInput.value = "";
    refPathInput.value = "";
    refProjectSelect.value = "";
    currInput.value = "";
    currPathInput.value = "";
    currProjectSelect.value = "";
    driftModelInput.value = "";
    driftTargetSelect.value = "";

    refInfo.classList.add("hidden");
    currInfo.classList.add("hidden");
    driftResultsSection.classList.add("hidden");
    hideError();
    checkCanRunM2();
  });
});
