(() => {
  "use strict";
  const Core = globalThis.LitSyncScholarCore;
  const $ = id => document.getElementById(id);

  async function send(type) {
    const response = await chrome.runtime.sendMessage({type});
    if (!response || !response.ok) throw new Error(response && response.error || "Extension command failed");
    return response;
  }

  function render(snapshot) {
    const context = snapshot.context;
    const run = snapshot.run;
    $("sync").textContent = context ? "LitSync Scholar query synced ✓" : "No Scholar query synced. Generate a query in LitSync first.";
    $("version").textContent = context ? context.active_query_version : "—";
    $("query").textContent = context ? context.query : "—";
    if (run && run.reportedTotal != null) {
      const initial = run.initialReportedTotal;
      $("reported").textContent = initial != null && initial !== run.reportedTotal
        ? `${run.reportedTotal} (initial ${initial})`
        : String(run.reportedTotal);
    } else {
      $("reported").textContent = "—";
    }
    $("target").textContent = run && run.targetTotal != null ? String(run.targetTotal) : "auto ≤ 1000";
    const p = run ? Core.progress(run) : {saved: 0, percent: 0};
    $("saved").textContent = String(p.saved);
    $("pages").textContent = String(run && Array.isArray(run.pagesVisited) ? run.pagesVisited.length : 0);
    $("failed").textContent = String(run ? Core.unresolvedIds(run).length : 0);
    $("verified").textContent = run && run.labelVerifiedCount != null
      ? `${run.labelVerifiedCount}/${run.targetTotal ?? "?"}`
      : "—";
    $("csvRows").textContent = run && run.exportCsvRowCount != null
      ? `${run.exportCsvRowCount}/${run.exportCsvExpectedRows ?? run.targetTotal ?? "?"}${run.exportCsvVerified ? " ✓" : ""}`
      : "—";
    $("bar").style.width = `${p.percent}%`;
    $("status").textContent = run ? `${run.status} · ${run.message || run.stage}` : "No active run.";
    $("labelName").textContent = run ? `Label: ${run.label}` : "";

    const fullyVerifiedComplete = Boolean(run && run.status === Core.STATUS.COMPLETE && Core.fullExportIntegritySatisfied(run));
    // Starting again must never overwrite a checkpointed/incomplete run. A new
    // run is allowed only when there is no run or the previous run is truly final.
    $("start").disabled = !context || Boolean(run && !fullyVerifiedComplete);
    $("pause").disabled = !run || run.status !== Core.STATUS.RUNNING;
    // Resume also acts as a safe "wake" button for a RUNNING checkpoint after
    // an unpacked-extension reload; it never resets saved IDs or the run label.
    $("resume").disabled = !run || fullyVerifiedComplete;
    $("reset").disabled = !run;

    const logs = (snapshot.logs || []).slice(-20).map(entry => `${entry.at} ${entry.level.toUpperCase()} ${entry.event} ${JSON.stringify(entry.details || {})}`).join("\n");
    $("logs").textContent = logs || "No logs yet.";
  }

  async function refresh() {
    try {
      const snapshot = await chrome.runtime.sendMessage({type: "LSSC_GET_SNAPSHOT"});
      if (snapshot && snapshot.ok) render(snapshot);
    } catch (error) {
      $("status").textContent = String(error && error.message || error);
    }
  }

  $("start").addEventListener("click", async () => { try { await send("LSSC_START"); window.close(); } catch (e) { $("status").textContent = e.message; } });
  $("pause").addEventListener("click", async () => { try { await send("LSSC_PAUSE"); await refresh(); } catch (e) { $("status").textContent = e.message; } });
  $("resume").addEventListener("click", async () => { try { await send("LSSC_RESUME"); window.close(); } catch (e) { $("status").textContent = e.message; } });
  $("reset").addEventListener("click", async () => {
    if (!confirm("Reset only the current Scholar run? The synced LitSync query will be kept.")) return;
    try { await send("LSSC_RESET"); await refresh(); } catch (e) { $("status").textContent = e.message; }
  });

  refresh();
  setInterval(refresh, 750);
})();
