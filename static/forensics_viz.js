/**
 * TruthLens Forensics Visualization & Export Library
 * Handles:
 * 1. Interactive vis-network multi-modal correlation graph
 * 2. Chronological evidence timeline
 * 3. Chart.js analytics charts for Overview
 * 4. High-fidelity PDF dossier export via html2pdf.js
 */

window.TruthLensViz = (function () {
  let activeNetwork = null;
  let overviewChartInstance = null;

  /**
   * Initializes or updates the interactive vis-network correlation graph.
   * @param {HTMLElement} container - DOM element to render into
   * @param {Object} graphData - { nodes: [...], edges: [...] }
   */
  function initCorrelationGraph(container, graphData) {
    if (!container) return;

    if (!window.vis || !window.vis.Network) {
      container.innerHTML = `
        <div class="p-6 text-center text-slate-400 text-xs">
          <i class="fa-solid fa-circle-nodes text-2xl text-cyan-400 mb-2 block"></i>
          Graph visualization library loading or offline. Telemetry nodes: ${graphData?.nodes?.length || 0}.
        </div>`;
      return;
    }

    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
      container.innerHTML = `
        <div class="p-8 text-center text-slate-500 text-xs">
          No correlation nodes available for this media scan.
        </div>`;
      return;
    }

    // Color mapping by risk status
    const statusColors = {
      authentic: { background: "#064e3b", border: "#10b981", highlight: { background: "#047857", border: "#34d399" } },
      suspicious: { background: "#78350f", border: "#f59e0b", highlight: { background: "#b45309", border: "#fbbf24" } },
      deepfake: { background: "#881337", border: "#f43f5e", highlight: { background: "#be123c", border: "#fb7185" } },
      neutral: { background: "#0f172a", border: "#38bdf8", highlight: { background: "#1e293b", border: "#7dd3fc" } }
    };

    const formattedNodes = graphData.nodes.map(node => {
      const colors = statusColors[node.status] || statusColors.neutral;
      return {
        id: node.id,
        label: node.label,
        shape: node.group === "verdict" ? "box" : (node.group === "root" ? "hexagon" : "dot"),
        size: node.group === "verdict" ? 28 : (node.group === "root" ? 24 : 18),
        color: {
          background: colors.background,
          border: colors.border,
          highlight: colors.highlight
        },
        font: {
          color: "#f8fafc",
          size: node.group === "verdict" ? 13 : 11,
          face: "Inter, system-ui, sans-serif",
          bold: node.group === "verdict" || node.group === "root"
        },
        borderWidth: 2,
        shadow: { enabled: true, color: "rgba(0,0,0,0.5)", size: 6, x: 2, y: 2 }
      };
    });

    const formattedEdges = graphData.edges.map(edge => ({
      from: edge.from || edge.from_node,
      to: edge.to || edge.to_node,
      label: edge.label || "",
      arrows: { to: { enabled: true, scaleFactor: 0.7 } },
      color: { color: "#475569", highlight: "#38bdf8" },
      font: { color: "#94a3b8", size: 9, align: "middle" },
      smooth: { type: "cubicBezier", roundness: 0.4 },
      width: 1.5
    }));

    const data = {
      nodes: new vis.DataSet(formattedNodes),
      edges: new vis.DataSet(formattedEdges)
    };

    const options = {
      physics: {
        solver: "forceAtlas2Based",
        forceAtlas2Based: {
          gravitationalConstant: -35,
          centralGravity: 0.008,
          springLength: 90,
          springConstant: 0.08
        },
        stabilization: { iterations: 150 }
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        zoomView: true,
        navigationButtons: true,
        keyboard: false
      }
    };

    if (activeNetwork) {
      activeNetwork.destroy();
    }
    activeNetwork = new vis.Network(container, data, options);
  }

  /**
   * Renders the chronological evidence timeline.
   * @param {HTMLElement} container - DOM element
   * @param {Array} events - List of TimelineEvent items
   */
  function renderTimeline(container, events) {
    if (!container) return;

    if (!events || events.length === 0) {
      container.innerHTML = `
        <div class="p-8 text-center text-slate-500 text-xs">
          <i class="fa-solid fa-check-circle text-emerald-400 text-xl mb-2 block"></i>
          No timeline anomalies flagged across video timestamps.
        </div>`;
      return;
    }

    const icons = {
      visual: "fa-video text-cyan-400",
      audio: "fa-waveform text-indigo-400",
      cross_modal: "fa-code-branch text-purple-400",
      metadata: "fa-fingerprint text-amber-400"
    };

    const severityBadges = {
      high: "bg-rose-500/20 text-rose-300 border-rose-500/30",
      medium: "bg-amber-500/20 text-amber-300 border-amber-500/30",
      low: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
    };

    let html = `<div class="relative border-l-2 border-slate-800 ml-4 space-y-6 py-2">`;

    events.forEach(ev => {
      const icon = icons[ev.modality] || "fa-triangle-exclamation text-slate-400";
      const badge = severityBadges[ev.severity] || severityBadges.medium;

      html += `
        <div class="relative pl-6 group">
          <div class="absolute -left-[17px] top-1 w-8 h-8 rounded-full bg-dark-900 border-2 border-slate-700 group-hover:border-cyan-400 flex items-center justify-center text-xs transition">
            <i class="fa-solid ${icon}"></i>
          </div>
          <div class="bg-dark-850 hover:bg-dark-800/80 rounded-xl p-4 border border-slate-800 transition shadow-sm space-y-1.5">
            <div class="flex items-center justify-between flex-wrap gap-2">
              <div class="flex items-center space-x-2">
                <span class="font-mono text-xs font-bold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/60">${ev.time_formatted || "00:00"}</span>
                <span class="text-xs font-semibold text-white">${ev.indicator || "Anomaly"}</span>
              </div>
              <span class="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full border ${badge}">
                ${ev.severity || "Flag"}
              </span>
            </div>
            <p class="text-xs text-slate-300 leading-relaxed">${ev.description || ""}</p>
          </div>
        </div>
      `;
    });

    html += `</div>`;
    container.innerHTML = html;
  }

  /**
   * Renders Chart.js charts on the Overview page.
   * @param {HTMLCanvasElement} canvas - Canvas element
   * @param {Object} overviewData - Data from /api/overview
   */
  function renderOverviewCharts(canvas, overviewData) {
    if (!canvas || !window.Chart) return;

    if (overviewChartInstance) {
      overviewChartInstance.destroy();
    }

    const ctx = canvas.getContext("2d");
    const authentic = overviewData.authentic_count || 0;
    const suspicious = overviewData.suspicious_count || 0;
    const deepfakes = overviewData.flagged_deepfakes || 0;
    const total = overviewData.total_scans || (authentic + suspicious + deepfakes);

    overviewChartInstance = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Authentic / Likely Real", "Suspicious / Review", "Manipulated / Deepfake"],
        datasets: [{
          data: total === 0 ? [1, 0, 0] : [authentic, suspicious, deepfakes],
          backgroundColor: total === 0 ? ["#334155"] : ["#10b981", "#f59e0b", "#f43f5e"],
          borderColor: "#0f172a",
          borderWidth: 3,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              color: "#cbd5e1",
              font: { size: 11, family: "Inter, sans-serif" },
              padding: 12
            }
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                if (total === 0) return " No scans recorded yet";
                const label = context.label || "";
                const val = context.raw || 0;
                const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                return ` ${label}: ${val} (${pct}%)`;
              }
            }
          }
        },
        cutout: "70%"
      }
    });
  }

  /**
   * Generates a clean PDF dossier download using html2pdf.js.
   * @param {string} elementId - ID of element to print
   * @param {string} filename - Target PDF filename
   */
  function exportReportPDF(elementId, filename) {
    const el = document.getElementById(elementId);
    if (!el) {
      alert("Report container not found for PDF export.");
      return;
    }

    if (!window.html2pdf) {
      window.print();
      return;
    }

    const opt = {
      margin: [10, 10, 10, 10],
      filename: filename || "TruthLens_Forensic_Dossier.pdf",
      image: { type: "jpeg", quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, logging: false },
      jsPDF: { unit: "mm", format: "a4", orientation: "portrait" }
    };

    html2pdf().set(opt).from(el).save();
  }

  return {
    initCorrelationGraph,
    renderTimeline,
    renderOverviewCharts,
    exportReportPDF
  };
})();
