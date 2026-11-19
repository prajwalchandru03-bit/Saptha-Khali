/**
 * TruthLens Main Application Controller
 * Manages 7 workspace views, media uploads, URL scanning, Compare Real vs. AI,
 * Text Forensics, Evidence Dossier tabs, vis-network graphs, Grounded AI Assistant,
 * and Incident Ticket Reporting.
 */

const AppState = {
  currentView: "view-overview",
  activeInvestigation: null,
  activeSha256: null,
  assistantHistory: [],
  overviewData: null,
  incidents: [],
};

// Safe DOM utility helpers
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = (text !== null && text !== undefined) ? text : "";
}

function safeSetValue(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = (val !== null && val !== undefined) ? val : "";
}

// -----------------------------------------------------------------------------
// DOM Ready Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initUploadHandler();
  initScannerHandler();
  initCompareHandler();
  initTextForensicsHandler();
  initAssistantDrawer();
  initIncidentModal();
  initDossierTabs();

  // Load Overview by default
  navigateTo("view-overview");
});

// -----------------------------------------------------------------------------
// View Navigation Controller (7 Workspaces + Dossier)
// -----------------------------------------------------------------------------
function initNavigation() {
  const navButtons = document.querySelectorAll("[data-nav-target]");
  navButtons.forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const target = btn.getAttribute("data-nav-target");
      if (target) navigateTo(target);
    });
  });

  // Mobile drawer / menu toggle (☰)
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("mainSidebar");
  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      sidebar.classList.toggle("-translate-x-full");
    });
  }

  // Close sidebar on mobile when clicking outside
  document.addEventListener("click", (e) => {
    if (sidebar && window.innerWidth < 1024) {
      if (!sidebar.contains(e.target) && mobileMenuBtn && !mobileMenuBtn.contains(e.target)) {
        sidebar.classList.add("-translate-x-full");
      }
    }
  });
}

function navigateTo(viewId) {
  AppState.currentView = viewId;

  // Hide all view containers
  const views = document.querySelectorAll(".truthlens-view");
  views.forEach(v => v.classList.add("hidden"));

  // Show active view
  const activeView = document.getElementById(viewId);
  if (activeView) {
    activeView.classList.remove("hidden");
  }

  // Update sidebar active highlights
  const navButtons = document.querySelectorAll("[data-nav-target]");
  navButtons.forEach(btn => {
    const target = btn.getAttribute("data-nav-target");
    if (target === viewId) {
      btn.className = "flex items-center space-x-3 px-3.5 py-2.5 rounded-xl bg-cyan-500/10 text-cyan-400 font-semibold border border-cyan-500/20 text-xs transition w-full text-left";
    } else {
      btn.className = "flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 font-medium text-xs transition w-full text-left";
    }
  });

  // Close mobile sidebar if open
  const sidebar = document.getElementById("mainSidebar");
  if (sidebar && window.innerWidth < 1024) {
    sidebar.classList.add("-translate-x-full");
  }

  // Trigger view data loaders
  if (viewId === "view-overview") {
    loadOverviewData();
  } else if (viewId === "view-history") {
    loadHistoryData();
  } else if (viewId === "view-models") {
    loadModelsStatus();
  }
}

// -----------------------------------------------------------------------------
// Overview View Controller
// -----------------------------------------------------------------------------
async function loadOverviewData() {
  try {
    const res = await fetch("/api/overview");
    if (!res.ok) throw new Error("Failed to load overview");
    const data = await res.json();
    AppState.overviewData = data;

    // Update KPI counters safely
    safeSetText("kpiTotalScans", data.total_scans ?? 0);
    safeSetText("kpiDeepfakes", data.flagged_deepfakes ?? 0);
    safeSetText("kpiSuspicious", data.suspicious_count ?? 0);
    safeSetText("kpiAuthentic", data.authentic_count ?? 0);
    safeSetText("kpiIncidents", data.reported_incidents_count ?? 0);
    safeSetText("kpiAvgScore", `${(((data.average_manipulation_score || 0)) * 100).toFixed(1)}%`);

    // Render Doughnut chart
    const canvas = document.getElementById("overviewChartCanvas");
    if (canvas && window.TruthLensViz) {
      window.TruthLensViz.renderOverviewCharts(canvas, data);
    }

    // Populate recent scans table
    const tableBody = document.getElementById("overviewRecentTableBody");
    if (tableBody) {
      if (!data.recent_scans || data.recent_scans.length === 0) {
        tableBody.innerHTML = `
          <tr>
            <td colspan="5" class="py-6 text-center text-xs text-slate-500">
              No recent investigations recorded yet. Run your first scan to begin.
            </td>
          </tr>`;
      } else {
        tableBody.innerHTML = data.recent_scans.map(scan => {
          const badgeClass = getVerdictBadgeClass(scan.verdict);
          const scorePct = ((scan.score || 0) * 100).toFixed(1);
          return `
            <tr class="border-b border-slate-800/60 hover:bg-slate-800/30 transition text-xs">
              <td class="py-3 px-3 font-mono text-cyan-400 font-semibold truncate max-w-[140px]">
                ${escapeHtml(scan.source || scan.sha256_hash.slice(0, 12))}
              </td>
              <td class="py-3 px-3 uppercase font-semibold text-slate-400 text-[10px]">
                ${escapeHtml(scan.media_type || "Video")}
              </td>
              <td class="py-3 px-3">
                <span class="px-2 py-0.5 rounded-full text-[10px] font-bold border ${badgeClass}">
                  ${escapeHtml(scan.verdict || "UNKNOWN")}
                </span>
              </td>
              <td class="py-3 px-3 font-mono font-bold text-slate-200">
                ${scorePct}%
              </td>
              <td class="py-3 px-3 text-right">
                <button onclick="loadAndDisplayReport('${escapeHtml(scan.sha256_hash)}')" class="px-2.5 py-1 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 text-[11px] font-semibold transition border border-cyan-500/20 cursor-pointer">
                  Inspect
                </button>
              </td>
            </tr>
          `;
        }).join("");
      }
    }
  } catch (err) {
    console.error("Error loading overview:", err);
  }
}

// -----------------------------------------------------------------------------
// History & Incidents Controller
// -----------------------------------------------------------------------------
async function loadHistoryData() {
  const historyContainer = document.getElementById("historyTableBody");
  const incidentsContainer = document.getElementById("incidentsTableBody");

  try {
    const [histRes, incRes] = await Promise.all([
      fetch("/api/history"),
      fetch("/incidents")
    ]);

    const historyItems = histRes.ok ? await histRes.json() : [];
    const incidentItems = incRes.ok ? await incRes.json() : [];
    AppState.incidents = incidentItems;

    // Render History
    if (historyContainer) {
      if (historyItems.length === 0) {
        historyContainer.innerHTML = `
          <tr>
            <td colspan="5" class="py-8 text-center text-xs text-slate-500">
              No past scans found in workspace.
            </td>
          </tr>`;
      } else {
        historyContainer.innerHTML = historyItems.map(item => {
          const badgeClass = getVerdictBadgeClass(item.verdict);
          const scorePct = ((item.score || 0) * 100).toFixed(1);
          return `
            <tr class="border-b border-slate-800/60 hover:bg-slate-800/30 transition text-xs">
              <td class="py-3.5 px-3 font-mono text-cyan-400 font-semibold truncate max-w-[150px]">
                ${escapeHtml(item.source_identifier || item.sha256_hash.slice(0, 12))}
              </td>
              <td class="py-3.5 px-3 text-slate-400 uppercase text-[10px] font-semibold">
                ${escapeHtml(item.media_type || "Video")}
              </td>
              <td class="py-3.5 px-3">
                <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${badgeClass}">
                  ${escapeHtml(item.verdict || "UNKNOWN")}
                </span>
              </td>
              <td class="py-3.5 px-3 font-mono font-bold text-slate-200">
                ${scorePct}%
              </td>
              <td class="py-3.5 px-3 text-right">
                <button onclick="loadAndDisplayReport('${escapeHtml(item.sha256_hash)}')" class="px-3 py-1 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 font-semibold text-xs border border-cyan-500/20 transition cursor-pointer">
                  Open Dossier
                </button>
              </td>
            </tr>
          `;
        }).join("");
      }
    }

    // Render Incident Tickets
    if (incidentsContainer) {
      if (incidentItems.length === 0) {
        incidentsContainer.innerHTML = `
          <tr>
            <td colspan="5" class="py-8 text-center text-xs text-slate-500">
              No formal deepfake incident tickets logged yet.
            </td>
          </tr>`;
      } else {
        incidentsContainer.innerHTML = incidentItems.map(inc => {
          return `
            <tr class="border-b border-slate-800/60 hover:bg-slate-800/30 transition text-xs">
              <td class="py-3 px-3 font-mono font-bold text-rose-400">
                ${escapeHtml(inc.ticket_id)}
              </td>
              <td class="py-3 px-3 text-slate-300 truncate max-w-[160px]">
                ${escapeHtml(inc.source_identifier)}
              </td>
              <td class="py-3 px-3 font-semibold text-slate-200">
                ${escapeHtml(inc.manipulation_type)}
              </td>
              <td class="py-3 px-3 text-[10px] uppercase font-bold text-slate-400">
                ${escapeHtml(inc.platform)}
              </td>
              <td class="py-3 px-3 text-right">
                <button onclick="viewTakedownTicket('${escapeHtml(inc.ticket_id)}')" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs border border-slate-700 transition cursor-pointer">
                  Takedown Notice
                </button>
              </td>
            </tr>
          `;
        }).join("");
      }
    }
  } catch (err) {
    console.error("Error loading history:", err);
  }
}

function viewTakedownTicket(ticketId) {
  const inc = (AppState.incidents || []).find(i => i.ticket_id === ticketId);
  const text = inc?.takedown_report_text || "No takedown report text stored.";
  alert(`Incident Ticket: ${ticketId}\nPlatform: ${inc?.platform || "Web"}\nTarget: ${inc?.source_identifier || ""}\n\n${text}`);
}

// -----------------------------------------------------------------------------
// Model Reliability & Status Controller
// -----------------------------------------------------------------------------
async function loadModelsStatus() {
  const container = document.getElementById("modelsStatusTableBody");
  if (!container) return;

  try {
    const res = await fetch("/api/models/status");
    if (!res.ok) throw new Error("Failed to load model statuses");
    const data = await res.json();

    safeSetText("systemDeviceBadge", `Compute Device: ${data.device || "CPU"}`);

    container.innerHTML = (data.models || []).map(m => {
      const isOnline = m.status.includes("Loaded") || m.status.includes("Operational") || m.status.includes("Connected");
      const statusPill = isOnline
        ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">${escapeHtml(m.status)}</span>`
        : `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">${escapeHtml(m.status)}</span>`;

      return `
        <tr class="border-b border-slate-800/60 hover:bg-slate-800/30 transition text-xs">
          <td class="py-3.5 px-3">
            <span class="font-bold text-white block">${escapeHtml(m.name)}</span>
            <span class="text-[11px] text-slate-400">${escapeHtml(m.role)}</span>
          </td>
          <td class="py-3.5 px-3 text-slate-300 font-mono text-[11px]">
            ${escapeHtml(m.type)}
          </td>
          <td class="py-3.5 px-3">
            ${statusPill}
          </td>
          <td class="py-3.5 px-3">
            <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
              ${escapeHtml(m.benchmark_accuracy)}
            </span>
          </td>
          <td class="py-3.5 px-3 text-slate-400 text-[11px]">
            ${escapeHtml(m.calibration || "Standard")}
          </td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Error loading model status:", err);
  }
}

// -----------------------------------------------------------------------------
// Workspace 1: Direct File Upload
// -----------------------------------------------------------------------------
function initUploadHandler() {
  const dropZone = document.getElementById("uploadDropZone");
  const fileInput = document.getElementById("uploadFileInput");
  const analyzeBtn = document.getElementById("executeUploadBtn");
  const captionInput = document.getElementById("uploadCaptionInput");
  const previewContainer = document.getElementById("uploadPreviewContainer");
  const previewMedia = document.getElementById("uploadPreviewMedia");
  const previewName = document.getElementById("uploadPreviewName");
  const previewSize = document.getElementById("uploadPreviewSize");
  const clearBtn = document.getElementById("uploadClearBtn");

  let selectedFile = null;

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("border-cyan-500", "bg-cyan-500/5");
  });
  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("border-cyan-500", "bg-cyan-500/5");
  });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-cyan-500", "bg-cyan-500/5");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      selectedFile = null;
      fileInput.value = "";
      if (previewContainer) previewContainer.classList.add("hidden");
      dropZone.classList.remove("hidden");
    });
  }

  function handleFileSelected(file) {
    selectedFile = file;
    dropZone.classList.add("hidden");
    if (previewContainer) previewContainer.classList.remove("hidden");

    if (previewName) previewName.textContent = file.name;
    if (previewSize) previewSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

    // Preview
    if (previewMedia) {
      previewMedia.innerHTML = "";
      const type = file.type;
      if (type.startsWith("image/")) {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.className = "w-full h-full object-cover";
        previewMedia.appendChild(img);
      } else if (type.startsWith("video/")) {
        const video = document.createElement("video");
        video.src = URL.createObjectURL(file);
        video.controls = true;
        video.className = "w-full h-full object-cover";
        previewMedia.appendChild(video);
      } else if (type.startsWith("audio/")) {
        previewMedia.innerHTML = `<div class="text-cyan-400 text-3xl flex items-center justify-center h-full"><i class="fa-solid fa-file-audio"></i></div>`;
      }
    }
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      if (!selectedFile) {
        showToast("Please select a video, picture, or audio file first.", "warning");
        return;
      }

      const formData = new FormData();
      formData.append("file", selectedFile);
      const cap = captionInput ? captionInput.value.trim() : "";
      if (cap) formData.append("caption", cap);

      showLoadingModal("Ingesting file and extracting frames...");

      try {
        const res = await fetch("/analyze/upload", {
          method: "POST",
          body: formData
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: "Upload failed" }));
          throw new Error(errData.detail || "Analysis failed");
        }

        const data = await res.json();
        hideLoadingModal();
        renderInvestigationDossier(data);
      } catch (err) {
        hideLoadingModal();
        showToast(err.message || "Failed to analyze media file.", "error");
      }
    });
  }
}

// -----------------------------------------------------------------------------
// Workspace 2: Social Media Scanner (Instagram / YouTube)
// -----------------------------------------------------------------------------
function initScannerHandler() {
  const scanBtn = document.getElementById("executeScanUrlBtn");
  const urlInput = document.getElementById("scanUrlInput");
  const captionInput = document.getElementById("scanCaptionInput");

  if (!scanBtn || !urlInput) return;

  scanBtn.addEventListener("click", async () => {
    const url = urlInput.value.trim();
    if (!url) {
      showToast("Please enter an Instagram Reel or YouTube Shorts URL.", "warning");
      return;
    }

    showLoadingModal("Downloading social reel via yt-dlp & analyzing forensics...");

    try {
      const res = await fetch("/analyze/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: url,
          caption: captionInput ? captionInput.value.trim() : null
        })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Scan failed" }));
        throw new Error(err.detail || "URL scan failed");
      }

      const data = await res.json();
      hideLoadingModal();
      renderInvestigationDossier(data);
    } catch (err) {
      hideLoadingModal();
      showToast(err.message, "error");
    }
  });
}

// -----------------------------------------------------------------------------
// Workspace 3: Compare Real vs. AI
// -----------------------------------------------------------------------------
function initCompareHandler() {
  const origDrop = document.getElementById("compOrigDrop");
  const origInput = document.getElementById("compOrigInput");
  const origPreview = document.getElementById("compOrigPreview");
  const origName = document.getElementById("compOrigName");

  const suspDrop = document.getElementById("compSuspDrop");
  const suspInput = document.getElementById("compSuspInput");
  const suspPreview = document.getElementById("compSuspPreview");
  const suspName = document.getElementById("compSuspName");

  const runBtn = document.getElementById("executeCompareBtn");
  const resultsCard = document.getElementById("compareResultsCard");

  let origFile = null;
  let suspFile = null;

  if (origDrop && origInput) {
    origDrop.addEventListener("click", () => origInput.click());
    origInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        origFile = e.target.files[0];
        if (origName) origName.textContent = origFile.name;
        if (origPreview) origPreview.classList.remove("hidden");
        origDrop.classList.add("hidden");
      }
    });
  }

  if (suspDrop && suspInput) {
    suspDrop.addEventListener("click", () => suspInput.click());
    suspInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        suspFile = e.target.files[0];
        if (suspName) suspName.textContent = suspFile.name;
        if (suspPreview) suspPreview.classList.remove("hidden");
        suspDrop.classList.add("hidden");
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      if (!origFile || !suspFile) {
        showToast("Please provide both Original reference and Suspect media files.", "warning");
        return;
      }

      const fd = new FormData();
      fd.append("original_file", origFile);
      fd.append("suspect_file", suspFile);

      showLoadingModal("Running biometric alignment & ELA noise comparison...");

      try {
        const res = await fetch("/api/compare", {
          method: "POST",
          body: fd
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Comparison failed" }));
          throw new Error(err.detail || "Forensic comparison failed");
        }

        const data = await res.json();
        hideLoadingModal();
        renderCompareResults(data);
      } catch (err) {
        hideLoadingModal();
        showToast(err.message, "error");
      }
    });
  }
}

function renderCompareResults(comp) {
  const card = document.getElementById("compareResultsCard");
  if (!card) return;
  card.classList.remove("hidden");

  // Badges & Scores
  const verdictBadge = document.getElementById("compVerdictBadge");
  if (verdictBadge) {
    verdictBadge.textContent = comp.verdict || "UNKNOWN";
    verdictBadge.className = `px-3 py-1 rounded-lg text-xs font-bold border tracking-wide ${getVerdictBadgeClass(comp.verdict)}`;
  }

  const aiProb = (((comp.ai_probability || 0)) * 100).toFixed(1);
  const idMatch = (((comp.identity_match_score || 0)) * 100).toFixed(1);

  safeSetText("compAiProbText", `${aiProb}%`);
  safeSetText("compIdentityText", `${idMatch}%`);

  const aiBar = document.getElementById("compAiProbBar");
  if (aiBar) aiBar.style.width = `${aiProb}%`;

  const idBar = document.getElementById("compIdentityBar");
  if (idBar) idBar.style.width = `${idMatch}%`;

  safeSetText("compSummaryText", comp.summary || "");

  // Render ELA Heatmaps
  const origHeatmapImg = document.getElementById("compOrigHeatmapImg");
  const suspHeatmapImg = document.getElementById("compSuspHeatmapImg");
  if (origHeatmapImg && comp.local_forensics?.original?.ela_heatmap_base64) {
    origHeatmapImg.src = comp.local_forensics.original.ela_heatmap_base64;
  }
  if (suspHeatmapImg && comp.local_forensics?.suspect?.ela_heatmap_base64) {
    suspHeatmapImg.src = comp.local_forensics.suspect.ela_heatmap_base64;
  }

  // Key Differentiators
  const diffsList = document.getElementById("compDifferentiatorsList");
  if (diffsList && comp.key_differentiators) {
    diffsList.innerHTML = comp.key_differentiators.map(d => `
      <li class="flex items-start space-x-2 text-xs text-slate-300">
        <i class="fa-solid fa-circle-dot text-rose-400 mt-1 text-[10px]"></i>
        <span>${escapeHtml(d)}</span>
      </li>
    `).join("");
  }

  card.scrollIntoView({ behavior: "smooth" });
}

// -----------------------------------------------------------------------------
// Workspace 4: Text Forensics Controller
// -----------------------------------------------------------------------------
function initTextForensicsHandler() {
  const analyzeBtn = document.getElementById("executeTextAnalysisBtn");
  const textInput = document.getElementById("textForensicsInput");
  const resultsCard = document.getElementById("textForensicsResultsCard");

  if (!analyzeBtn || !textInput) return;

  analyzeBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) {
      showToast("Please paste text to analyze.", "warning");
      return;
    }

    try {
      const res = await fetch("/api/analyze/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Analysis failed" }));
        throw new Error(err.detail || "Text analysis failed");
      }

      const data = await res.json();
      if (resultsCard) resultsCard.classList.remove("hidden");

      safeSetText("textVerdictBadge", data.verdict);
      const scorePct = (((data.ai_likelihood_score || 0)) * 100).toFixed(1);
      safeSetText("textAiScoreText", `${scorePct}%`);
      const aiScoreBar = document.getElementById("textAiScoreBar");
      if (aiScoreBar) aiScoreBar.style.width = `${scorePct}%`;

      safeSetText("textWordCount", data.word_count ?? 0);
      safeSetText("textBurstiness", data.burstiness_score ?? 0);
      safeSetText("textLexicalDiv", data.lexical_diversity ?? 0);
      safeSetText("textRepetition", data.repetition_index ?? 0);
      safeSetText("textReadingEase", data.flesch_reading_ease ?? 0);
      safeSetText("textKincaidGrade", data.flesch_kincaid_grade ?? 0);

      const patternsList = document.getElementById("textPatternsList");
      if (patternsList) {
        if (!data.detected_patterns || data.detected_patterns.length === 0) {
          patternsList.innerHTML = `<li class="text-xs text-slate-400">No formulaic clichés or low-burstiness flags identified.</li>`;
        } else {
          patternsList.innerHTML = data.detected_patterns.map(p => `
            <li class="flex items-center space-x-2 text-xs text-amber-300">
              <i class="fa-solid fa-flag text-amber-400 text-[10px]"></i>
              <span>${escapeHtml(p)}</span>
            </li>
          `).join("");
        }
      }

      if (resultsCard) resultsCard.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      showToast(err.message, "error");
    }
  });
}

// -----------------------------------------------------------------------------
// Investigation Dossier Renderer & Tab Switcher
// -----------------------------------------------------------------------------
function initDossierTabs() {
  const tabs = document.querySelectorAll("[data-dossier-tab]");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-dossier-tab");

      tabs.forEach(t => {
        t.classList.remove("border-cyan-400", "text-cyan-400");
        t.classList.add("border-transparent", "text-slate-400");
      });
      tab.classList.add("border-cyan-400", "text-cyan-400");
      tab.classList.remove("border-transparent", "text-slate-400");

      document.querySelectorAll(".dossier-tab-content").forEach(c => c.classList.add("hidden"));
      const activeContent = document.getElementById(target);
      if (activeContent) {
        activeContent.classList.remove("hidden");
      }

      // If switching to Network Graph, fit vis-network
      if (target === "dossierTabNetwork" && AppState.activeInvestigation?.report?.correlation_graph) {
        const graphContainer = document.getElementById("correlationGraphContainer");
        if (graphContainer && window.TruthLensViz) {
          window.TruthLensViz.initCorrelationGraph(graphContainer, AppState.activeInvestigation.report.correlation_graph);
        }
      }
    });
  });
}

async function loadAndDisplayReport(sha256) {
  showLoadingModal("Retrieving forensic dossier...");
  try {
    const res = await fetch(`/reports/${sha256}`);
    if (!res.ok) throw new Error("Could not find report");
    const rep = await res.json();
    hideLoadingModal();

    const fullObj = {
      success: true,
      report: rep,
      artifacts: {
        ela_heatmap: `/artifacts/${sha256}/ela_heatmap.png`,
        audio_wav: `/artifacts/${sha256}/audio_16k_mono.wav`,
        flagged_frames: [],
        face_crops: []
      },
      pipeline_telemetry: { sha256_hash: sha256 }
    };
    renderInvestigationDossier(fullObj);
  } catch (err) {
    hideLoadingModal();
    showToast(err.message, "error");
  }
}

function renderInvestigationDossier(data) {
  if (!data || !data.report) {
    showToast("Invalid dossier data received.", "error");
    return;
  }

  AppState.activeInvestigation = data;
  const report = data.report;
  AppState.activeSha256 = report.sha256_hash;

  // Unhide Dossier Workspace
  navigateTo("view-dossier");

  // Header values
  safeSetText("dossierShaBadge", report.sha256_hash);
  const prov = report.provenance || {};
  safeSetText("dossierTargetName", prov.source_identifier || (report.sha256_hash ? report.sha256_hash.slice(0, 16) : "Media Target"));

  // Verdict Banner
  const verdictBadge = document.getElementById("dossierVerdictBadge");
  if (verdictBadge) {
    verdictBadge.textContent = report.verdict || "UNKNOWN";
    verdictBadge.className = `px-3.5 py-1.5 rounded-xl text-sm font-black border tracking-wide shadow-sm ${getVerdictBadgeClass(report.verdict)}`;
  }

  const normProb = normalizeProb(report.deepfake_probability || 0);
  const probPct = (normProb * 100).toFixed(1);
  safeSetText("dossierScoreText", `${probPct}%`);
  const scoreBar = document.getElementById("dossierScoreBar");
  if (scoreBar) scoreBar.style.width = `${probPct}%`;
  safeSetText("dossierConfidencePill", `Confidence: ${report.confidence_level || "Moderate"}`);

  // 3-Way Media Origin Assessment with distinct titles
  const origin = report.media_origin_assessment || {};
  renderOriginCard("dossierOriginFace", origin.deepfake_face_manipulation, "FACE MANIPULATION");
  renderOriginCard("dossierOriginVisual", origin.ai_generated_visuals, "AI-GENERATED VISUALS");
  renderOriginCard("dossierOriginVoice", origin.voice_manipulation_or_other_voice, "VOICE & DUBBING");

  // Sub-Tab 1: Executive Summary & Clean 2x2 Cross-Modal Grid
  safeSetText("dossierSummaryText", report.summary || "Forensic analysis completed.");

  // Populate 2x2 Cross-Modal Cards:
  const cm = report.cross_modal_findings || {};
  const cmTelemetry = data.pipeline_telemetry?.cross_modal_metrics || {};

  // Card a: Lip-Sync & Speech Dynamics
  safeSetText("dossierLipSyncText", cm.lip_sync || "Lip-sync correlation evaluated.");
  const lipSyncBadge = document.getElementById("cmLipSyncBadge");
  if (lipSyncBadge) {
    const rScore = cmTelemetry.lip_sync?.correlation_score;
    if (rScore !== null && rScore !== undefined && !isNaN(rScore)) {
      lipSyncBadge.textContent = `r = ${Number(rScore).toFixed(2)}`;
      if (rScore >= 0.30) {
        lipSyncBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
      } else if (rScore <= 0.10) {
        lipSyncBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-rose-500/20 text-rose-300 border-rose-500/30";
      } else {
        lipSyncBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-amber-500/20 text-amber-300 border-amber-500/30";
      }
    } else {
      lipSyncBadge.textContent = cmTelemetry.lip_sync?.dubbed_voice_detected ? "Dubbed Voice" : "Evaluated";
      lipSyncBadge.className = cmTelemetry.lip_sync?.dubbed_voice_detected
        ? "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-rose-500/20 text-rose-300 border-rose-500/30"
        : "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-slate-800 text-slate-300 border-slate-700";
    }
  }

  // Card b: Caption vs. Content Alignment
  safeSetText("dossierCaptionText", cm.caption_vs_content || "Visual semantics evaluated.");
  const captionBadge = document.getElementById("cmCaptionBadge");
  if (captionBadge) {
    const maxSim = cmTelemetry.caption_alignment?.max_similarity;
    const simPct = (maxSim !== null && maxSim !== undefined && !isNaN(maxSim)) ? `${(maxSim * 100).toFixed(0)}%` : null;
    const isMismatch = (cm.caption_vs_content || "").toLowerCase().includes("mismatch");
    if (isMismatch) {
      captionBadge.textContent = simPct ? `Score: ${simPct} (Mismatch)` : "Context Mismatch";
      captionBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-rose-500/20 text-rose-300 border-rose-500/30";
    } else {
      captionBadge.textContent = simPct ? `Score: ${simPct}` : "Aligned";
      captionBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-teal-500/20 text-teal-300 border-teal-500/30";
    }
  }

  // Card c: Acoustic Scene vs. Visual Setting
  safeSetText("dossierSceneText", cm.audio_vs_visual || "Scene consistency evaluated.");
  const sceneBadge = document.getElementById("cmSceneBadge");
  if (sceneBadge) {
    const isSceneMismatch = cmTelemetry.scene_consistency?.mismatch_detected || (cm.audio_vs_visual || "").toLowerCase().includes("mismatch");
    if (isSceneMismatch) {
      sceneBadge.textContent = "Scene Mismatch";
      sceneBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-rose-500/20 text-rose-300 border-rose-500/30";
    } else {
      sceneBadge.textContent = "Consistent";
      sceneBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    }
  }

  // Card d: Speech Transcript vs. Claimed Context
  safeSetText("dossierTranscriptText", cm.transcript_vs_caption || "Speech transcript evaluated against context.");
  const transBadge = document.getElementById("cmTranscriptBadge");
  if (transBadge) {
    const isTransMismatch = (cm.transcript_vs_caption || "").toLowerCase().includes("contradict") || (cm.transcript_vs_caption || "").toLowerCase().includes("mismatch");
    if (isTransMismatch) {
      transBadge.textContent = "Context Mismatch";
      transBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-rose-500/20 text-rose-300 border-rose-500/30";
    } else {
      transBadge.textContent = "Context Verified";
      transBadge.className = "text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    }
  }

  // Populate Detected Evidence Screenshots directly on Summary & Findings tab
  const screenshotsContainer = document.getElementById("dossierScreenshotsContainer");
  const screenshotsList = document.getElementById("dossierScreenshotsList");
  const detectionScreenshots = report.detection_screenshots || data.pipeline_telemetry?.visual_metrics?.detection_screenshots || [];

  if (screenshotsContainer && screenshotsList) {
    if (detectionScreenshots.length > 0) {
      screenshotsContainer.classList.remove("hidden");
      screenshotsList.innerHTML = detectionScreenshots.map((shot, idx) => {
        const isFlagged = shot.is_flagged;
        const normScore = normalizeProb(shot.score);
        const scorePct = (normScore * 100).toFixed(0);
        const badgeColor = isFlagged
          ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
          : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
        const badgeLabel = isFlagged
          ? `DETECTED ANOMALY (${scorePct}%)`
          : "VERIFIED AUTHENTIC";
        const icon = isFlagged ? "fa-triangle-exclamation text-rose-400" : "fa-shield-check text-emerald-400";

        const cropThumbnail = shot.crop_b64 ? `
          <div class="relative shrink-0 w-12 h-12 rounded-lg overflow-hidden border border-slate-700 bg-black/60 shadow" title="Target Face Crop">
            <img src="${shot.crop_b64}" class="w-full h-full object-cover">
          </div>
        ` : "";

        return `
          <div class="bg-dark-850 rounded-xl border border-slate-800 p-3 space-y-2.5 shadow-sm hover:border-slate-700 transition">
            <div class="relative rounded-lg overflow-hidden border border-slate-800/90 bg-black group">
              <img src="${shot.image_b64}" alt="Evidence Frame" class="w-full h-44 object-cover cursor-pointer transition transform group-hover:scale-[1.02]" onclick="window.open('${shot.image_b64}')" title="Click to view full size">
              <div class="absolute top-2 left-2 bg-dark-950/85 backdrop-blur-sm border border-slate-700/80 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-cyan-300 shadow">
                <i class="fa-regular fa-clock mr-1 text-[9px]"></i>${escapeHtml(shot.time_formatted || "00:00.0s")}
              </div>
              <div class="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-bold border backdrop-blur-sm ${badgeColor} shadow">
                <i class="fa-solid ${icon} mr-1 text-[9px]"></i>${badgeLabel}
              </div>
            </div>

            <div class="flex items-start space-x-2.5">
              ${cropThumbnail}
              <div class="flex-1 min-w-0">
                <p class="text-[11px] text-slate-300 leading-relaxed font-sans">${escapeHtml(shot.reason || "Sampled keyframe verified.")}</p>
              </div>
            </div>
          </div>
        `;
      }).join("");
    } else {
      screenshotsContainer.classList.add("hidden");
    }
  }

  // Sub-Tab 2: Evidence Timeline
  const timelineContainer = document.getElementById("timelineEventsContainer");
  if (timelineContainer && window.TruthLensViz) {
    window.TruthLensViz.renderTimeline(timelineContainer, report.timeline_events || []);
  }

  // Sub-Tab 3: Correlation Graph
  const graphContainer = document.getElementById("correlationGraphContainer");
  if (graphContainer && window.TruthLensViz && report.correlation_graph) {
    window.TruthLensViz.initCorrelationGraph(graphContainer, report.correlation_graph);
  }

  // Sub-Tab 4: Visuals & ELA
  const elaImg = document.getElementById("dossierElaHeatmap");
  if (elaImg) {
    elaImg.src = data.artifacts?.ela_heatmap || `/artifacts/${report.sha256_hash}/ela_heatmap.png`;
    elaImg.onerror = () => { if (elaImg.parentElement) elaImg.parentElement.classList.add("hidden"); };
  }

  // Sub-Tab 5: Provenance Card
  safeSetText("provHashText", report.sha256_hash);
  safeSetText("provMediaType", prov.media_type || "video");
  safeSetText("provDuration", prov.duration_seconds ? `${prov.duration_seconds}s` : "N/A");
  safeSetText("provCameraMake", prov.camera_make || "Stripped by platform");
  safeSetText("provCameraModel", prov.camera_model || "None detected");
  safeSetText("provSoftware", prov.software || "None detected");
  safeSetText("provPlatformNote", prov.platform_recompression_note || "Standard lossy recompression observed.");

  // Sub-Tab 6: Takedown Report
  safeSetValue("dossierTakedownText", report.draft_takedown_report || "");

  // Update Assistant Header with current Hash
  safeSetText("asstActiveShaBadge", report.sha256_hash ? report.sha256_hash.slice(0, 14) : "");
}

function normalizeProb(v) {
  let val = parseFloat(v);
  if (isNaN(val)) return 0.04;
  if (val > 1.0) val = val / 100.0;
  return Math.max(0.04, Math.min(0.94, val));
}

function renderOriginCard(containerId, item, defaultTitle) {
  const container = document.getElementById(containerId);
  if (!container || !item) return;

  const level = (item.indicator || "Low").toUpperCase();
  let badgeColor = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
  if (level === "MODERATE") badgeColor = "bg-amber-500/20 text-amber-300 border-amber-500/30";
  if (level === "HIGH") badgeColor = "bg-rose-500/20 text-rose-300 border-rose-500/30";

  const cardTitle = defaultTitle || item.title || "INDICATOR";
  let scoreBadge = "";
  if (item.score !== undefined && item.score !== null) {
    const calibratedScorePct = (normalizeProb(item.score) * 100).toFixed(0);
    scoreBadge = ` (${calibratedScorePct}%)`;
  }

  container.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <span class="text-xs font-bold text-white uppercase tracking-wider">${cardTitle}</span>
      <span class="text-[10px] font-extrabold px-2 py-0.5 rounded-full border ${badgeColor}">${level}${scoreBadge}</span>
    </div>
    <p class="text-xs text-slate-300 leading-relaxed">${escapeHtml(item.reason || "No anomalies flagged.")}</p>
  `;
}

// -----------------------------------------------------------------------------
// Grounded AI Forensic Assistant Slide-Over Drawer
// -----------------------------------------------------------------------------
function initAssistantDrawer() {
  const drawer = document.getElementById("assistantDrawer");
  const openBtn = document.getElementById("openAssistantBtn");
  const closeBtn = document.getElementById("closeAssistantBtn");
  const backdrop = document.getElementById("assistantBackdrop");
  const sendBtn = document.getElementById("asstSendBtn");
  const input = document.getElementById("asstChatInput");
  const chatMessages = document.getElementById("asstChatMessages");

  if (!drawer) return;

  function toggleDrawer(open) {
    if (open) {
      drawer.classList.remove("translate-x-full");
      if (backdrop) backdrop.classList.remove("hidden");
    } else {
      drawer.classList.add("translate-x-full");
      if (backdrop) backdrop.classList.add("hidden");
    }
  }

  if (openBtn) openBtn.addEventListener("click", () => toggleDrawer(true));
  if (closeBtn) closeBtn.addEventListener("click", () => toggleDrawer(false));
  if (backdrop) backdrop.addEventListener("click", () => toggleDrawer(false));

  // Quick Chips
  const chips = document.querySelectorAll("[data-asst-chip]");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-asst-chip");
      if (input) input.value = q;
      sendAssistantMessage(q);
    });
  });

  if (sendBtn && input) {
    sendBtn.addEventListener("click", () => {
      const q = input.value.trim();
      if (q) sendAssistantMessage(q);
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const q = input.value.trim();
        if (q) sendAssistantMessage(q);
      }
    });
  }

  async function sendAssistantMessage(question) {
    const sha = AppState.activeSha256 || (AppState.overviewData?.recent_scans?.[0]?.sha256_hash);
    if (!sha) {
      showToast("Please run or select an investigation first to query the assistant.", "warning");
      return;
    }

    if (input) input.value = "";

    // Append User Message
    appendChatMessage("user", question);

    // Typing indicator
    const typingId = appendTypingIndicator();

    try {
      const res = await fetch("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          investigation_id: sha,
          question: question,
          chat_history: AppState.assistantHistory
        })
      });

      removeTypingIndicator(typingId);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Query failed" }));
        throw new Error(err.detail || "Assistant query failed");
      }

      const data = await res.json();
      AppState.assistantHistory.push({ role: "user", content: question });
      AppState.assistantHistory.push({ role: "assistant", content: data.answer });

      appendChatMessage("assistant", data.answer, data.source);
    } catch (err) {
      removeTypingIndicator(typingId);
      appendChatMessage("assistant", `⚠️ Error: ${err.message}`);
    }
  }

  function appendChatMessage(role, text, source) {
    if (!chatMessages) return;
    const msgDiv = document.createElement("div");
    if (role === "user") {
      msgDiv.className = "flex justify-end";
      msgDiv.innerHTML = `
        <div class="bg-cyan-600 text-white rounded-2xl rounded-tr-sm p-3 max-w-[85%] text-xs leading-relaxed shadow-sm">
          ${escapeHtml(text)}
        </div>
      `;
    } else {
      msgDiv.className = "flex justify-start space-x-2";
      const sourceTag = source ? `<span class="text-[9px] font-mono uppercase px-1.5 py-0.2 rounded bg-slate-800 text-cyan-400 border border-slate-700">${escapeHtml(source)}</span>` : "";
      const parsedText = window.marked ? window.marked.parse(text) : `<p class="whitespace-pre-wrap">${escapeHtml(text)}</p>`;

      msgDiv.innerHTML = `
        <div class="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0 text-[10px] mt-1 border border-cyan-500/30">
          <i class="fa-solid fa-robot"></i>
        </div>
        <div class="bg-dark-850 border border-slate-800 text-slate-200 rounded-2xl rounded-tl-sm p-3.5 max-w-[88%] text-xs leading-relaxed space-y-2 shadow-sm">
          <div class="flex items-center justify-between pb-1 border-b border-slate-800">
            <span class="font-bold text-[11px] text-cyan-400">Forensic Assistant</span>
            ${sourceTag}
          </div>
          <div class="prose prose-invert prose-xs max-w-none text-slate-200">
            ${parsedText}
          </div>
        </div>
      `;
    }

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendTypingIndicator() {
    if (!chatMessages) return null;
    const id = `typing_${Date.now()}`;
    const div = document.createElement("div");
    div.id = id;
    div.className = "flex justify-start space-x-2";
    div.innerHTML = `
      <div class="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0 text-[10px] mt-1 border border-cyan-500/30">
        <i class="fa-solid fa-robot"></i>
      </div>
      <div class="bg-dark-850 border border-slate-800 rounded-xl px-4 py-2 text-xs text-slate-400 flex items-center space-x-2">
        <i class="fa-solid fa-circle-notch fa-spin text-cyan-400"></i>
        <span>Consulting forensic telemetry...</span>
      </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
  }

  function removeTypingIndicator(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
  }
}

// -----------------------------------------------------------------------------
// Incident Reporting Modal Controller
// -----------------------------------------------------------------------------
function initIncidentModal() {
  const modal = document.getElementById("incidentReportModal");
  const openBtn = document.getElementById("dossierLogIncidentBtn");
  const cancelBtn = document.getElementById("cancelIncidentBtn");
  const submitBtn = document.getElementById("submitIncidentBtn");

  if (!modal) return;

  function openModal() {
    const rep = AppState.activeInvestigation?.report;
    const sha = rep?.sha256_hash || AppState.activeSha256 || document.getElementById("dossierShaBadge")?.textContent?.trim() || "";
    const prov = rep?.provenance || {};
    const source = prov.source_identifier || document.getElementById("dossierTargetName")?.textContent?.trim() || (sha ? sha.slice(0, 16) : "");
    const takedown = rep?.draft_takedown_report || document.getElementById("dossierTakedownText")?.value || "";

    safeSetValue("incidentShaInput", sha);
    safeSetValue("incidentSourceInput", source);
    safeSetValue("incidentTakedownText", takedown);
    safeSetValue("incidentNotesInput", "");

    // Pre-select manipulation category based on analysis findings
    const typeSelect = document.getElementById("incidentTypeSelect");
    if (typeSelect) {
      if (rep) {
        const origin = rep.media_origin_assessment || {};
        const faceInd = (origin.deepfake_face_manipulation?.indicator || "").toUpperCase();
        const voiceInd = (origin.voice_manipulation_or_other_voice?.indicator || "").toUpperCase();
        const visualInd = (origin.ai_generated_visuals?.indicator || "").toUpperCase();

        if (faceInd === "HIGH" || faceInd === "MODERATE") {
          typeSelect.value = "Face-Swap Deepfake";
        } else if (voiceInd === "HIGH" || voiceInd === "MODERATE") {
          typeSelect.value = "Voice Clone / Dubbing";
        } else if (visualInd === "HIGH" || visualInd === "MODERATE") {
          typeSelect.value = "AI-Generated Visuals";
        } else {
          typeSelect.value = "Face-Swap Deepfake";
        }
      }
    }

    // Auto-select platform based on URL/source
    const platformSelect = document.getElementById("incidentPlatformSelect");
    if (platformSelect) {
      const srcLower = (source || "").toLowerCase();
      if (srcLower.includes("instagram.com") || srcLower.includes("instagr.am")) {
        platformSelect.value = "Instagram";
      } else if (srcLower.includes("youtube.com") || srcLower.includes("youtu.be")) {
        platformSelect.value = "YouTube";
      } else if (srcLower.includes("tiktok.com")) {
        platformSelect.value = "TikTok";
      } else if (srcLower.includes("twitter.com") || srcLower.includes("x.com")) {
        platformSelect.value = "X / Twitter";
      } else {
        platformSelect.value = "Direct Web";
      }
    }

    modal.classList.remove("hidden");
  }

  function closeModal() {
    modal.classList.add("hidden");
  }

  if (openBtn) {
    openBtn.addEventListener("click", openModal);
  }

  // Support any alternate report buttons
  const altButtons = document.querySelectorAll("[data-action='report-incident'], #dossierReportIncidentBtn");
  altButtons.forEach(b => b.addEventListener("click", openModal));

  if (cancelBtn) {
    cancelBtn.addEventListener("click", closeModal);
  }

  // Close on backdrop click
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      closeModal();
    }
  });

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) {
      closeModal();
    }
  });

  if (submitBtn) {
    submitBtn.addEventListener("click", async () => {
      const sha = document.getElementById("incidentShaInput")?.value?.trim() || "";
      const source = document.getElementById("incidentSourceInput")?.value?.trim() || "";
      const manipType = document.getElementById("incidentTypeSelect")?.value || "Face-Swap Deepfake";
      const takedown = document.getElementById("incidentTakedownText")?.value || "";
      const notes = document.getElementById("incidentNotesInput")?.value?.trim() || "";
      const platform = document.getElementById("incidentPlatformSelect")?.value || "Instagram";

      if (!sha) {
        showToast("No active media scan found. Please run an investigation first.", "warning");
        return;
      }

      submitBtn.disabled = true;
      const originalHtml = submitBtn.innerHTML;
      submitBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin mr-1"></i> Confirming...`;

      try {
        const res = await fetch("/incidents/report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sha256_hash: sha,
            source_identifier: source,
            manipulation_type: manipType,
            takedown_report_text: takedown,
            investigator_notes: notes,
            platform: platform
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: "Failed to record ticket" }));
          throw new Error(errData.detail || "Failed to record incident ticket");
        }
        const rec = await res.json();
        closeModal();
        showToast(`Incident ticket ${rec.ticket_id} logged successfully!`, "success");

        if (AppState.currentView === "view-history") {
          loadHistoryData();
        } else if (AppState.currentView === "view-overview") {
          loadOverviewData();
        }
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalHtml;
      }
    });
  }
}

// -----------------------------------------------------------------------------
// Helpers & Utilities
// -----------------------------------------------------------------------------
function getVerdictBadgeClass(verdict) {
  const v = (verdict || "").toUpperCase();
  if (v.includes("AUTHENTIC") || v.includes("REAL")) {
    return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
  }
  if (v.includes("SUSPICIOUS") || v.includes("REVIEW") || v.includes("INCONCLUSIVE")) {
    return "bg-amber-500/20 text-amber-400 border-amber-500/30";
  }
  return "bg-rose-500/20 text-rose-400 border-rose-500/30";
}

function showLoadingModal(message) {
  const modal = document.getElementById("loadingModal");
  const msgEl = document.getElementById("loadingModalText");
  if (msgEl) msgEl.textContent = message || "Processing forensic analysis...";
  if (modal) modal.classList.remove("hidden");
}

function hideLoadingModal() {
  const modal = document.getElementById("loadingModal");
  if (modal) modal.classList.add("hidden");
}

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  if (!container) return;

  const toast = document.createElement("div");
  let colors = "bg-slate-800 text-white border-slate-700";
  let icon = "fa-circle-info text-cyan-400";
  if (type === "error") {
    colors = "bg-rose-950 text-rose-100 border-rose-800";
    icon = "fa-circle-exclamation text-rose-400";
  } else if (type === "warning") {
    colors = "bg-amber-950 text-amber-100 border-amber-800";
    icon = "fa-triangle-exclamation text-amber-400";
  } else if (type === "success") {
    colors = "bg-emerald-950 text-emerald-100 border-emerald-800";
    icon = "fa-circle-check text-emerald-400";
  }

  toast.className = `flex items-center space-x-3 px-4 py-3 rounded-xl border shadow-xl text-xs transition duration-300 transform translate-y-2 opacity-0 ${colors}`;
  toast.innerHTML = `<i class="fa-solid ${icon} text-base"></i><span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.remove("translate-y-2", "opacity-0");
  });

  setTimeout(() => {
    toast.classList.add("opacity-0", "translate-y-2");
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function copyToClipboard(text) {
  if (!navigator.clipboard) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
      showToast("Copied to clipboard!", "success");
    } catch (err) {
      showToast("Could not copy to clipboard.", "error");
    }
    document.body.removeChild(textArea);
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    showToast("Copied to clipboard!", "success");
  }).catch(() => {
    showToast("Could not copy to clipboard.", "error");
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
