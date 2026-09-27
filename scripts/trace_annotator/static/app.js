(() => {
  const state = {
    meta: null,
    runs: [],
    currentId: null,
    detail: null,
    batch: "",
    taskNumber: null,
    statusFilter: "all",
    search: "",
    stepFilter: "all",
    inspectedIndex: null,
    saveAndNext: false,
    ignoreAutosave: false,
    dirty: false,
    lastSavedKey: "",
    saveTimer: 0,
    saveInFlight: null,
    overview: false,
  };

  const els = {
    progress: document.getElementById("header-progress"),
    prev: document.getElementById("prev-run"),
    next: document.getElementById("next-run"),
    overviewBtn: document.getElementById("overview-btn"),
    overviewClose: document.getElementById("overview-close"),
    overviewPanel: document.getElementById("overview-panel"),
    overviewBody: document.getElementById("overview-body"),
    overviewNote: document.getElementById("overview-note"),
    search: document.getElementById("run-search"),
    batchSelect: document.getElementById("batch-select"),
    taskFilters: document.getElementById("task-filters"),
    runCount: document.getElementById("run-count"),
    runList: document.getElementById("run-list"),
    timelineEmpty: document.getElementById("timeline-empty"),
    timeline: document.getElementById("timeline"),
    runMeta: document.getElementById("run-meta"),
    variantSummary: document.getElementById("variant-summary"),
    heading: document.getElementById("timeline-heading"),
    chips: document.getElementById("outcome-chips"),
    stepList: document.getElementById("step-list"),
    inspector: document.getElementById("step-inspector"),
    inspectorEmpty: document.getElementById("inspector-empty"),
    inspectorBody: document.getElementById("inspector-body"),
    inspectorMeta: document.getElementById("inspector-meta"),
    inspectorAction: document.getElementById("inspector-action"),
    inspectorBadge: document.getElementById("inspector-badge"),
    inspectorThought: document.getElementById("inspector-thought"),
    inspectorResult: document.getElementById("inspector-result"),
    inspectorGraph: document.getElementById("inspector-graph"),
    inspectorImageWrap: document.getElementById("inspector-image-wrap"),
    inspectorImage: document.getElementById("inspector-image"),
    inspectorImageCaption: document.getElementById("inspector-image-caption"),
    pddlPanel: document.getElementById("pddl-panel"),
    pddlFrame: document.getElementById("pddl-frame"),
    pddlLinks: document.getElementById("pddl-links"),
    formEmpty: document.getElementById("form-empty"),
    form: document.getElementById("annotator"),
    errorSummary: document.getElementById("error-summary"),
    subtaskCount: document.getElementById("subtask-count"),
    evalFillNote: document.getElementById("eval-fill-note"),
    errorFillNote: document.getElementById("error-fill-note"),
    criteriaList: document.getElementById("criteria-list"),
    failureFieldset: document.getElementById("failure-fieldset"),
    failureList: document.getElementById("failure-list"),
    comments: document.getElementById("comments"),
    failureError: document.getElementById("failure-error"),
    efficiencyFieldset: document.getElementById("efficiency-fieldset"),
    efficiencyList: document.getElementById("efficiency-list"),
    saveBtn: document.getElementById("save-btn"),
    saveNextBtn: document.getElementById("save-next-btn"),
    saveStatus: document.getElementById("save-status"),
    taskMetrics: document.getElementById("task-metrics"),
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function formatPercent(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
    return `${Math.round(value * 1000) / 10}%`;
  }

  function formatSeconds(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
    return `${value.toFixed(1)}s`;
  }

  function formatUsd(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "";
    return `$${value.toFixed(3)}`;
  }

  function formatCount(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "";
    return value.toLocaleString();
  }

  function varietyLabel(run) {
    if (run.axis && run.kind) return `${run.axis}-${run.kind}`;
    return run.episode_id || run.variant_folder || run.run_id;
  }

  function runLabel(run) {
    const variety = varietyLabel(run);
    if (state.taskNumber != null) return variety;
    if (run.task_number != null) return `T${run.task_number} · ${variety}`;
    return variety;
  }

  const HIDDEN_KINDS_BY_BATCH = {
    luna_high_states_images: new Set(["AMB", "ROOM", "CAND"]),
  };

  function isFrontendVisibleRun(run) {
    const hiddenKinds = HIDDEN_KINDS_BY_BATCH[run.batch];
    return !hiddenKinds || !hiddenKinds.has(run.kind);
  }

  function batchRuns() {
    const visible = state.runs.filter(isFrontendVisibleRun);
    if (!state.batch) return visible;
    return visible.filter((run) => run.batch === state.batch);
  }

  function taskNumbers() {
    const numbers = new Set();
    for (const run of batchRuns()) {
      if (typeof run.task_number === "number") numbers.add(run.task_number);
    }
    return [...numbers].sort((a, b) => a - b);
  }

  function filteredRuns() {
    const query = state.search.trim().toLowerCase();
    return batchRuns().filter((run) => {
      if (state.taskNumber != null && run.task_number !== state.taskNumber) return false;
      if (state.statusFilter === "annotated" && !run.annotated) return false;
      if (state.statusFilter === "unannotated" && run.annotated) return false;
      if (!query) return true;
      const haystack = [
        run.run_id,
        run.episode_id,
        run.variant_folder,
        run.batch,
        run.axis,
        run.kind,
        run.instruction,
        run.variant_summary,
        `t${run.task_number}`,
        `task ${run.task_number}`,
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }

  function currentIndex() {
    return filteredRuns().findIndex((run) => run.run_id === state.currentId);
  }

  function setSaveStatus(message, kind) {
    els.saveStatus.textContent = message;
    els.saveStatus.classList.toggle("is-ok", kind === "ok");
    els.saveStatus.classList.toggle("is-err", kind === "err");
  }

  function formatDuration(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
    if (value >= 60) return `${(value / 60).toFixed(1)}m`;
    return `${value.toFixed(0)}s`;
  }

  function mean(values) {
    const nums = values.filter((value) => typeof value === "number" && !Number.isNaN(value));
    if (!nums.length) return null;
    return nums.reduce((sum, value) => sum + value, 0) / nums.length;
  }

  function metricsScope() {
    let runs = batchRuns();
    const task = state.taskNumber;
    if (task != null) {
      runs = runs.filter((run) => run.task_number === task);
    }
    return { runs: runs.map(withLiveSubtasks), task };
  }

  function withLiveSubtasks(run) {
    if (run.run_id !== state.currentId || !state.detail || els.form.hidden) return run;
    const total = state.detail.criteria?.length || run.subtasks_total || 0;
    return {
      ...run,
      subtasks_done: checkedCriteria().length,
      subtasks_total: total,
      subtasks_source: "annotation",
    };
  }

  function criteriaCompletion(run) {
    const total = Number(run.subtasks_total) || 0;
    if (!total) return null;
    return (Number(run.subtasks_done) || 0) / total;
  }

  function isCriteriaSuccess(run) {
    const total = Number(run.subtasks_total) || 0;
    return total > 0 && (Number(run.subtasks_done) || 0) >= total;
  }

  function metricCard(label, value, sub) {
    return `
      <div class="metric">
        <span class="metric-label">${escapeHtml(label)}</span>
        <span class="metric-value">${escapeHtml(value)}</span>
        ${sub ? `<span class="metric-sub">${escapeHtml(sub)}</span>` : ""}
      </div>
    `;
  }

  function renderTaskMetrics() {
    if (!els.taskMetrics) return;
    const { runs, task } = metricsScope();
    if (!runs.length) {
      els.taskMetrics.hidden = true;
      els.taskMetrics.innerHTML = "";
      return;
    }
    const n = runs.length;
    const successes = runs.filter(isCriteriaSuccess).length;
    const completion = mean(runs.map(criteriaCompletion));
    const subDone = runs.reduce((sum, run) => sum + (Number(run.subtasks_done) || 0), 0);
    const subTotal = runs.reduce((sum, run) => sum + (Number(run.subtasks_total) || 0), 0);
    const annotated = runs.filter((run) => run.annotated).length;
    const runtime = mean(runs.map((run) => run.runtime));
    const replans = mean(runs.map((run) => run.replanning_count_0));
    const timeouts = runs.filter((run) => {
      if (Number(run.combined_time_limit_hit) >= 1) return true;
      const used = Number(run.combined_time_used_s);
      const limit = Number(run.combined_time_limit_s);
      return limit > 0 && used >= limit;
    }).length;
    const axes = ["ACC", "INC", "OUT"]
      .map((axis) => {
        const group = runs.filter((run) => run.axis === axis);
        if (!group.length) return "";
        const ok = group.filter(isCriteriaSuccess).length;
        return `${axis} ${ok}/${group.length}`;
      })
      .filter(Boolean)
      .join(" · ");
    const title = task != null ? `Task ${task}` : state.batch || "All runs";
    const annotatedSub = `${annotated} / ${n} annotated · ${n} variants`;
    const tagCounts = {};
    for (const run of runs) {
      const tags = new Set([
        ...(run.auto_efficiency_errors || []),
        ...(run.auto_failure_reasons || []),
      ]);
      for (const tag of tags) tagCounts[tag] = (tagCounts[tag] || 0) + 1;
    }
    const commonErrors = Object.entries(tagCounts)
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .map(([tag, count]) => `${tag.replaceAll("_", " ")} ${count}/${n}`)
      .join(" · ");
    els.taskMetrics.hidden = false;
    els.taskMetrics.innerHTML = [
      metricCard(title, `${n} variants`, annotatedSub),
      metricCard("Success", formatPercent(n ? successes / n : null), `${successes} / ${n}${axes ? ` · ${axes}` : ""}`),
      metricCard("Completion", formatPercent(completion), "mean spec criteria complete"),
      metricCard("Subtasks", subTotal ? `${subDone} / ${subTotal}` : "n/a", "spec criteria across variants"),
      metricCard("Runtime", formatDuration(runtime), "mean"),
      metricCard("Replans", replans == null ? "n/a" : String(Math.round(replans)), timeouts ? `${timeouts} timed out` : "mean"),
      metricCard("Common errors", commonErrors ? `${Object.keys(tagCounts).length} types` : "none", commonErrors || "no repeated skill errors"),
    ].join("");
    renderOverview();
  }

  const OVERVIEW_KIND_ORDER = ["BASE", "SUB", "ABS", "CON", "DIS"];
  const AXIS_ORDER = ["ACC", "INC", "OUT"];
  const AXIS_FULL = {
    ACC: "Accurate",
    INC: "Incomplete",
    OUT: "Outdated",
  };
  const KIND_FULL = {
    BASE: "Base",
    SUB: "Substitute",
    ABS: "Absent",
    CON: "Containment",
    DIS: "Distractor",
    AMB: "Underspecified",
    ROOM: "Room known",
    CAND: "Candidate rooms",
  };
  const OUTCOME_SEGMENTS = [
    { id: "complete", label: "Complete", className: "is-complete" },
    { id: "partial", label: "Partial", className: "is-partial" },
    { id: "noProgress", label: "No progress", className: "is-no-progress" },
  ];
  const ISSUE_SEGMENTS = [
    { id: "none", label: "No issue", className: "is-no-issue" },
    { id: "failureOnly", label: "Failure only", className: "is-failure-only" },
    {
      id: "inefficiencyOnly",
      label: "Inefficiency only",
      className: "is-inefficiency-only",
    },
    { id: "both", label: "Both", className: "is-both" },
  ];

  function overviewRuns() {
    const runs = batchRuns()
      .map(withLiveSubtasks)
      .filter((run) => OVERVIEW_KIND_ORDER.includes(run.kind));
    if (state.batch) return runs;
    const seen = new Map();
    for (const run of runs) {
      if (run.planner === "vlm_tamp_pddl") continue;
      const key = `${run.task_number}|${run.axis}|${run.kind}`;
      const existing = seen.get(key);
      if (!existing || (run.batch === "luna_high_states_images" && existing.batch !== "luna_high_states_images")) {
        seen.set(key, run);
      }
    }
    return [...seen.values()];
  }

  function outcomeBreakdown(runs) {
    const counts = {
      complete: 0,
      partial: 0,
      noProgress: 0,
    };
    let unknown = 0;
    for (const run of runs) {
      const total = Number(run.subtasks_total) || 0;
      const done = Math.max(0, Math.min(Number(run.subtasks_done) || 0, total));
      if (!total) {
        unknown += 1;
      } else if (done >= total) {
        counts.complete += 1;
      } else if (done > 0) {
        counts.partial += 1;
      } else {
        counts.noProgress += 1;
      }
    }
    return {
      counts,
      denominator: counts.complete + counts.partial + counts.noProgress,
      unknown,
    };
  }

  function runHasTag(run, listKey, countKey, ids) {
    if (ids.some((id) => Number(run[countKey]?.[id]) > 0)) return true;
    return (run[listKey] || []).some((id) => ids.includes(id));
  }

  function issueBreakdown(runs) {
    const failureIds = (state.meta?.failure_labels || []).map((item) => item.id);
    const inefficiencyIds = (state.meta?.efficiency_labels || []).map((item) => item.id);
    const annotated = runs.filter((run) => run.annotated);
    const counts = {
      none: 0,
      failureOnly: 0,
      inefficiencyOnly: 0,
      both: 0,
    };
    for (const run of annotated) {
      const hasFailure = runHasTag(
        run,
        "failure_reasons",
        "failure_reason_counts",
        failureIds
      );
      const hasInefficiency = runHasTag(
        run,
        "efficiency_errors",
        "efficiency_error_counts",
        inefficiencyIds
      );
      if (hasFailure && hasInefficiency) counts.both += 1;
      else if (hasFailure) counts.failureOnly += 1;
      else if (hasInefficiency) counts.inefficiencyOnly += 1;
      else counts.none += 1;
    }
    return {
      counts,
      denominator: annotated.length,
      unannotated: runs.length - annotated.length,
    };
  }

  function tagPrevalence(runs, labels, listKey, countKey) {
    const annotated = runs.filter((run) => run.annotated);
    return {
      denominator: annotated.length,
      unannotated: runs.length - annotated.length,
      rows: labels.map((item) => {
        const count = annotated.filter((run) =>
          runHasTag(run, listKey, countKey, [item.id])
        ).length;
        return {
          id: item.id,
          label: item.label,
          count,
          ratio: annotated.length ? count / annotated.length : 0,
        };
      }),
    };
  }

  function criteriaBreakdown(runs) {
    const total = runs.reduce(
      (sum, run) => sum + Math.max(0, Number(run.subtasks_total) || 0),
      0
    );
    const done = runs.reduce((sum, run) => {
      const runTotal = Math.max(0, Number(run.subtasks_total) || 0);
      return sum + Math.max(0, Math.min(Number(run.subtasks_done) || 0, runTotal));
    }, 0);
    return { done, total, ratio: total ? done / total : null };
  }

  function partialProgressBreakdown(runs) {
    const ratios = runs
      .map((run) => {
        const total = Number(run.subtasks_total) || 0;
        const done = Math.max(0, Math.min(Number(run.subtasks_done) || 0, total));
        return total > 0 && done > 0 && done < total ? done / total : null;
      })
      .filter((ratio) => ratio != null);
    if (!ratios.length) {
      return { count: 0, average: null, minimum: null, maximum: null };
    }
    return {
      count: ratios.length,
      average: mean(ratios),
      minimum: Math.min(...ratios),
      maximum: Math.max(...ratios),
    };
  }

  function runOccurrenceTotal(run, listKey, countKey, ids) {
    const selected = new Set(run[listKey] || []);
    return ids.reduce((sum, id) => {
      const count = Number(run[countKey]?.[id]);
      if (Number.isFinite(count) && count > 0) return sum + count;
      return sum + (selected.has(id) ? 1 : 0);
    }, 0);
  }

  function quantile(sortedValues, percentile) {
    if (!sortedValues.length) return null;
    const position = (sortedValues.length - 1) * percentile;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return sortedValues[lower];
    const weight = position - lower;
    return (
      sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight
    );
  }

  function distributionStats(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((left, right) => left - right);
    const average = mean(sorted);
    const variance =
      sorted.reduce((sum, value) => sum + (value - average) ** 2, 0) /
      sorted.length;
    return {
      count: sorted.length,
      mean: average,
      median: quantile(sorted, 0.5),
      q1: quantile(sorted, 0.25),
      q3: quantile(sorted, 0.75),
      standardDeviation: Math.sqrt(variance),
      minimum: sorted[0],
      maximum: sorted[sorted.length - 1],
    };
  }

  function overviewKpi(label, count, denominator, note) {
    const value = denominator ? formatPercent(count / denominator) : "n/a";
    const sample = denominator ? `${count}/${denominator} runs` : "no eligible runs";
    return `
      <article class="overview-kpi">
        <span class="overview-kpi-label">${escapeHtml(label)}</span>
        <strong class="overview-kpi-value">${escapeHtml(value)}</strong>
        <span class="overview-kpi-sub">${escapeHtml(sample)}</span>
        <span class="overview-kpi-note">${escapeHtml(note)}</span>
      </article>
    `;
  }

  function overviewGroupLabel(group) {
    if (group.taskNumber == null) return escapeHtml(group.label);
    return `<button type="button" class="overview-group-button" data-jump-task="${group.taskNumber}">${escapeHtml(group.label)}</button>`;
  }

  function stackedPercentageChart(title, description, groups, segments, breakdownFn) {
    const prepared = groups
      .map((group) => ({ ...group, breakdown: breakdownFn(group.runs) }))
      .filter((group) => group.breakdown.denominator > 0);
    if (!prepared.length) return "";
    const legend = segments
      .map(
        (segment) => `
          <span class="stacked-legend-item">
            <span class="stacked-legend-swatch ${segment.className}" aria-hidden="true"></span>
            ${escapeHtml(segment.label)}
          </span>
        `
      )
      .join("");
    const bars = prepared
      .map((group) => {
        const denominator = group.breakdown.denominator;
        const pieces = segments
          .map((segment) => {
            const count = group.breakdown.counts[segment.id] || 0;
            const ratio = count / denominator;
            const percentage = formatPercent(ratio);
            const directLabel =
              count > 0
                ? `<span class="stacked-segment-label">${escapeHtml(percentage)}</span>`
                : "";
            return `
              <span
                class="stacked-segment ${segment.className}"
                style="width: ${ratio * 100}%"
                title="${escapeHtml(`${segment.label}: ${percentage} (${count}/${denominator})`)}"
                aria-label="${escapeHtml(`${segment.label}: ${percentage}, ${count} of ${denominator} runs`)}"
              >${directLabel}</span>
            `;
          })
          .join("");
        const values = segments
          .map((segment) => {
            const count = group.breakdown.counts[segment.id] || 0;
            return `${segment.label}: ${formatPercent(count / denominator)} (${count}/${denominator})`;
          })
          .join(" · ");
        return `
          <div class="stacked-row">
            <div class="stacked-row-label">${overviewGroupLabel(group)}<small>n=${denominator}</small></div>
            <div class="stacked-track" role="img" aria-label="${escapeHtml(`${group.label}: percentage distribution`)}">
              ${pieces}
            </div>
            <div class="stacked-row-values">${escapeHtml(values)}</div>
          </div>
        `;
      })
      .join("");
    const tableHeads = segments
      .map((segment) => `<th>${escapeHtml(segment.label)}</th>`)
      .join("");
    const tableRows = prepared
      .map((group) => {
        const denominator = group.breakdown.denominator;
        const cells = segments
          .map((segment) => {
            const count = group.breakdown.counts[segment.id] || 0;
            return `<td>${escapeHtml(formatPercent(count / denominator))}<small>${count}/${denominator}</small></td>`;
          })
          .join("");
        return `<tr><th scope="row">${overviewGroupLabel(group)}</th><td>${denominator}</td>${cells}</tr>`;
      })
      .join("");
    return `
      <section class="overview-section percentage-card">
        <h3>${escapeHtml(title)}</h3>
        <p class="help">${escapeHtml(description)} Exact percentages and count/denominator values are shown below each bar.</p>
        <div class="stacked-legend" aria-label="Chart legend">${legend}</div>
        <div class="stacked-chart">${bars}</div>
        <details class="chart-data-details">
          <summary>View percentage table</summary>
          <div class="chart-table-wrap">
            <table class="overview-data-table">
              <thead><tr><th>Group</th><th>n</th>${tableHeads}</tr></thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function prevalenceChart(title, description, prevalence, toneClass) {
    if (!prevalence.denominator) return "";
    const rows = prevalence.rows
      .map(
        (row) => `
          <div class="prevalence-row">
            <div class="prevalence-label">${escapeHtml(row.label)}</div>
            <div
              class="prevalence-track"
              role="img"
              aria-label="${escapeHtml(`${row.label}: ${formatPercent(row.ratio)}, ${row.count} of ${prevalence.denominator} annotated runs`)}"
            >
              <span class="prevalence-fill ${toneClass}" style="width: ${row.ratio * 100}%"></span>
            </div>
            <div class="prevalence-value">
              <strong>${escapeHtml(formatPercent(row.ratio))}</strong>
              <small>${row.count}/${prevalence.denominator} runs</small>
            </div>
          </div>
        `
      )
      .join("");
    const tableRows = prevalence.rows
      .map(
        (row) => `
          <tr>
            <th scope="row">${escapeHtml(row.label)}</th>
            <td>${escapeHtml(formatPercent(row.ratio))}</td>
            <td>${row.count}/${prevalence.denominator}</td>
          </tr>
        `
      )
      .join("");
    return `
      <section class="overview-section percentage-card">
        <h3>${escapeHtml(title)}</h3>
        <p class="help">${escapeHtml(description)} Each row shows its exact percentage and count/denominator.</p>
        <div class="prevalence-chart">${rows}</div>
        <details class="chart-data-details">
          <summary>View percentage table</summary>
          <div class="chart-table-wrap">
            <table class="overview-data-table">
              <thead><tr><th>Type</th><th>Runs</th><th>Count</th></tr></thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function partialProgressChart(groups) {
    const rows = groups.map((group) => ({
      ...group,
      partial: partialProgressBreakdown(group.runs),
    }));
    const chartRows = rows
      .map((group) => {
        const { count, average, minimum, maximum } = group.partial;
        if (!count) {
          return `
            <div class="partial-progress-row">
              <div class="stacked-row-label">${overviewGroupLabel(group)}<small>no partial runs</small></div>
              <div class="partial-progress-empty">Not applicable</div>
              <div class="partial-progress-value">—</div>
            </div>
          `;
        }
        return `
          <div class="partial-progress-row">
            <div class="stacked-row-label">${overviewGroupLabel(group)}<small>n=${count} partial run${count === 1 ? "" : "s"}</small></div>
            <div
              class="partial-progress-track"
              role="img"
              aria-label="${escapeHtml(`${group.label}: ${formatPercent(average)} average completion across ${count} partial runs, range ${formatPercent(minimum)} to ${formatPercent(maximum)}`)}"
            >
              <span class="partial-progress-fill" style="width: ${average * 100}%"></span>
            </div>
            <div class="partial-progress-value">
              <strong>${escapeHtml(formatPercent(average))}</strong>
              <small>range ${escapeHtml(formatPercent(minimum))}–${escapeHtml(formatPercent(maximum))}</small>
            </div>
          </div>
        `;
      })
      .join("");
    const tableRows = rows
      .map((group) => {
        const { count, average, minimum, maximum } = group.partial;
        return `
          <tr>
            <th scope="row">${overviewGroupLabel(group)}</th>
            <td>${count}</td>
            <td>${average == null ? "—" : escapeHtml(formatPercent(average))}</td>
            <td>${minimum == null ? "—" : escapeHtml(formatPercent(minimum))}</td>
            <td>${maximum == null ? "—" : escapeHtml(formatPercent(maximum))}</td>
          </tr>
        `;
      })
      .join("");
    return `
      <section class="overview-section percentage-card">
        <h3>Average partial completion by task</h3>
        <p class="help">Shows how much of the task partial runs completed before stopping. For each partial run, completion is completed criteria divided by total criteria; the chart averages those run percentages equally within each task and excludes complete and no-progress runs.</p>
        <div class="partial-progress-chart">${chartRows}</div>
        <details class="chart-data-details">
          <summary>View percentage table</summary>
          <div class="chart-table-wrap">
            <table class="overview-data-table">
              <thead><tr><th>Task</th><th>Partial n</th><th>Average</th><th>Minimum</th><th>Maximum</th></tr></thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function issueStoryRows(groups, failureLabels, inefficiencyLabels) {
    const failureIds = failureLabels.map((item) => item.id);
    const inefficiencyIds = inefficiencyLabels.map((item) => item.id);
    return groups
      .map((group) => {
        const annotatedRuns = group.runs.filter((run) => run.annotated);
        if (!annotatedRuns.length) return null;
        const issueFree = annotatedRuns.filter(
          (run) =>
            !runHasTag(
              run,
              "failure_reasons",
              "failure_reason_counts",
              failureIds
            ) &&
            !runHasTag(
              run,
              "efficiency_errors",
              "efficiency_error_counts",
              inefficiencyIds
            )
        ).length;
        const failureOccurrences = annotatedRuns.reduce(
          (sum, run) =>
            sum +
            runOccurrenceTotal(
              run,
              "failure_reasons",
              "failure_reason_counts",
              failureIds
            ),
          0
        );
        const inefficiencyOccurrences = annotatedRuns.reduce(
          (sum, run) =>
            sum +
            runOccurrenceTotal(
              run,
              "efficiency_errors",
              "efficiency_error_counts",
              inefficiencyIds
            ),
          0
        );
        return {
          label: group.label,
          taskNumber: group.taskNumber,
          runTotal: annotatedRuns.length,
          issueFree,
          issueBearing: annotatedRuns.length - issueFree,
          failureOccurrences,
          inefficiencyOccurrences,
          occurrenceTotal:
            failureOccurrences + inefficiencyOccurrences,
        };
      })
      .filter(Boolean);
  }

  function issueStoryChart(
    plotId,
    groups,
    failureLabels,
    inefficiencyLabels
  ) {
    const rows = issueStoryRows(
      groups,
      failureLabels,
      inefficiencyLabels
    );
    if (!rows.length) return "";
    const tableRows = rows
      .map((row) => {
        const issueFreePct = formatPercent(row.issueFree / row.runTotal);
        const issuePct = formatPercent(row.issueBearing / row.runTotal);
        const failurePct = row.occurrenceTotal
          ? formatPercent(row.failureOccurrences / row.occurrenceTotal)
          : "0%";
        const inefficiencyPct = row.occurrenceTotal
          ? formatPercent(
              row.inefficiencyOccurrences / row.occurrenceTotal
            )
          : "0%";
        return `
          <tr>
            <th scope="row">${overviewGroupLabel(row)}</th>
            <td>${escapeHtml(issueFreePct)}<small>${row.issueFree}/${row.runTotal} runs</small></td>
            <td>${escapeHtml(issuePct)}<small>${row.issueBearing}/${row.runTotal} runs</small></td>
            <td>${escapeHtml(failurePct)}<small>${row.failureOccurrences}/${row.occurrenceTotal} occurrences</small></td>
            <td>${escapeHtml(inefficiencyPct)}<small>${row.inefficiencyOccurrences}/${row.occurrenceTotal} occurrences</small></td>
          </tr>
        `;
      })
      .join("");
    return `
      <section class="overview-section percentage-card issue-story-card">
        <h3>Issue prevalence and composition by task</h3>
        <p class="help">The left panel divides annotated runs into issue-free versus issue-bearing. The right panel divides recorded issue occurrences into failures versus inefficiencies; occurrence shares sum to 100% without forcing overlapping runs into one family.</p>
        <div id="${escapeHtml(plotId)}" class="plotly-issue-story-chart" role="img" aria-label="Issue-free and issue-bearing runs, plus failure and inefficiency occurrence shares, by task"></div>
        <details class="chart-data-details">
          <summary>View percentages and raw counts</summary>
          <div class="chart-table-wrap">
            <table class="overview-data-table">
              <thead><tr><th>Task</th><th>Issue-free runs</th><th>Issue-bearing runs</th><th>Failure occurrences</th><th>Inefficiency occurrences</th></tr></thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function renderIssueStoryPlot(
    plotId,
    groups,
    failureLabels,
    inefficiencyLabels
  ) {
    const plot = document.getElementById(plotId);
    if (!plot) return;
    if (!window.Plotly) {
      plot.textContent = "Plotly could not be loaded. Use the table below.";
      plot.classList.add("plotly-load-error");
      return;
    }
    const rows = issueStoryRows(
      groups,
      failureLabels,
      inefficiencyLabels
    );
    const labels = rows.map((row) => row.label);
    const issueFreePct = rows.map(
      (row) => (100 * row.issueFree) / row.runTotal
    );
    const issueBearingPct = rows.map(
      (row) => (100 * row.issueBearing) / row.runTotal
    );
    const failurePct = rows.map((row) =>
      row.occurrenceTotal
        ? (100 * row.failureOccurrences) / row.occurrenceTotal
        : 0
    );
    const inefficiencyPct = rows.map((row) =>
      row.occurrenceTotal
        ? (100 * row.inefficiencyOccurrences) / row.occurrenceTotal
        : 0
    );
    const visibleText = (values) =>
      values.map((value) => (value >= 5 ? `${value.toFixed(1)}%` : ""));
    const traces = [
      {
        type: "bar",
        name: "Issue-free runs",
        x: labels,
        y: issueFreePct,
        customdata: rows.map((row) => [row.issueFree, row.runTotal]),
        text: visibleText(issueFreePct),
        textposition: "inside",
        marker: { color: "#047857" },
        xaxis: "x",
        yaxis: "y",
        hovertemplate:
          "%{x}<br>Issue-free: %{y:.1f}%<br>%{customdata[0]}/%{customdata[1]} runs<extra></extra>",
      },
      {
        type: "bar",
        name: "Issue-bearing runs",
        x: labels,
        y: issueBearingPct,
        customdata: rows.map((row) => [row.issueBearing, row.runTotal]),
        text: visibleText(issueBearingPct),
        textposition: "inside",
        marker: { color: "#7c3aed" },
        xaxis: "x",
        yaxis: "y",
        hovertemplate:
          "%{x}<br>Issue-bearing: %{y:.1f}%<br>%{customdata[0]}/%{customdata[1]} runs<extra></extra>",
      },
      {
        type: "bar",
        name: "Failure occurrences",
        x: labels,
        y: failurePct,
        customdata: rows.map((row) => [
          row.failureOccurrences,
          row.occurrenceTotal,
        ]),
        text: visibleText(failurePct),
        textposition: "inside",
        marker: { color: "#b91c1c" },
        xaxis: "x2",
        yaxis: "y2",
        hovertemplate:
          "%{x}<br>Failures: %{y:.1f}%<br>%{customdata[0]}/%{customdata[1]} occurrences<extra></extra>",
      },
      {
        type: "bar",
        name: "Inefficiency occurrences",
        x: labels,
        y: inefficiencyPct,
        customdata: rows.map((row) => [
          row.inefficiencyOccurrences,
          row.occurrenceTotal,
        ]),
        text: visibleText(inefficiencyPct),
        textposition: "inside",
        marker: { color: "#1d4ed8" },
        xaxis: "x2",
        yaxis: "y2",
        hovertemplate:
          "%{x}<br>Inefficiencies: %{y:.1f}%<br>%{customdata[0]}/%{customdata[1]} occurrences<extra></extra>",
      },
    ];
    const axis = {
      range: [0, 100],
      dtick: 20,
      ticksuffix: "%",
      fixedrange: true,
      gridcolor: "#e2e8f0",
    };
    window.Plotly.newPlot(
      plot,
      traces,
      {
        height: 330,
        margin: { l: 42, r: 42, t: 36, b: 42 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {
          family: '"Fira Sans", system-ui, sans-serif',
          color: "#1e293b",
          size: 11,
        },
        barmode: "stack",
        xaxis: { domain: [0, 0.47], anchor: "y", fixedrange: true },
        yaxis: { ...axis, domain: [0, 1], anchor: "x" },
        xaxis2: {
          domain: [0.53, 1],
          anchor: "y2",
          fixedrange: true,
        },
        yaxis2: {
          ...axis,
          domain: [0, 1],
          anchor: "x2",
          side: "right",
        },
        annotations: [
          {
            text: "Run prevalence",
            x: 0.235,
            y: 1.12,
            xref: "paper",
            yref: "paper",
            showarrow: false,
            font: { size: 12 },
          },
          {
            text: "Occurrence composition",
            x: 0.765,
            y: 1.12,
            xref: "paper",
            yref: "paper",
            showarrow: false,
            font: { size: 12 },
          },
        ],
        legend: {
          orientation: "h",
          x: 0,
          y: 1.26,
          font: { size: 9 },
          traceorder: "normal",
        },
        hovermode: "closest",
      },
      {
        responsive: true,
        displayModeBar: false,
        displaylogo: false,
      }
    );
  }

  function occurrenceByTypeChart(
    plotId,
    title,
    description,
    runs,
    labels,
    listKey,
    countKey
  ) {
    const taskNumbers = [
      ...new Set(
        runs
          .map((run) => run.task_number)
          .filter((task) => typeof task === "number")
      ),
    ].sort((left, right) => left - right);
    const counts = taskNumbers.map((task) => {
      const taskRuns = runs.filter((run) => run.task_number === task);
      return Object.fromEntries(
        labels.map((item) => [
          item.id,
          taskRuns.reduce(
            (sum, run) =>
              sum +
              runOccurrenceTotal(run, listKey, countKey, [item.id]),
            0
          ),
        ])
      );
    });
    const total = counts.reduce(
      (sum, taskCounts) =>
        sum +
        Object.values(taskCounts).reduce(
          (taskSum, value) => taskSum + value,
          0
        ),
      0
    );
    const heads = labels
      .map((item) => `<th>${escapeHtml(shortTagLabel(item.label))}</th>`)
      .join("");
    const rows = taskNumbers
      .map((task, index) => {
        const taskTotal = Object.values(counts[index]).reduce(
          (sum, value) => sum + value,
          0
        );
        const cells = labels
          .map((item) => `<td>${counts[index][item.id] || 0}</td>`)
          .join("");
        return `<tr><th scope="row">Task ${task}</th>${cells}<td>${taskTotal}</td></tr>`;
      })
      .join("");
    return `
      <section class="overview-section percentage-card type-occurrence-card">
        <h3>${escapeHtml(title)}</h3>
        <p class="help">${escapeHtml(description)} Counts sum saved repeated occurrences; unannotated runs contribute one occurrence per auto-inferred tag.</p>
        <p class="distribution-summary"><strong>Total:</strong> ${total} occurrences across ${runs.length} runs</p>
        <div id="${escapeHtml(plotId)}" class="plotly-type-chart" role="img" aria-label="${escapeHtml(`${title}. ${total} total occurrences across ${runs.length} runs.`)}"></div>
        <details class="chart-data-details">
          <summary>View counts table</summary>
          <div class="chart-table-wrap">
            <table class="overview-data-table">
              <thead><tr><th>Task</th>${heads}<th>Total</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function renderOccurrenceByTypePlot(
    plotId,
    runs,
    labels,
    listKey,
    countKey,
    colors
  ) {
    const plot = document.getElementById(plotId);
    if (!plot) return;
    if (!window.Plotly) {
      plot.textContent = "Plotly could not be loaded. Use the counts table below.";
      plot.classList.add("plotly-load-error");
      return;
    }
    const taskNumbers = [
      ...new Set(
        runs
          .map((run) => run.task_number)
          .filter((task) => typeof task === "number")
      ),
    ].sort((left, right) => left - right);
    const taskLabels = taskNumbers.map((task) => `Task ${task}`);
    const rawSeries = labels.map((item) =>
      taskNumbers.map((task) =>
        runs
          .filter((run) => run.task_number === task)
          .reduce(
            (sum, run) =>
              sum +
              runOccurrenceTotal(run, listKey, countKey, [item.id]),
            0
          )
      )
    );
    const taskTotals = taskNumbers.map((_task, taskIndex) =>
      rawSeries.reduce((sum, values) => sum + values[taskIndex], 0)
    );
    const traces = labels.map((item, labelIndex) => ({
      type: "bar",
      name: shortTagLabel(item.label),
      x: taskLabels,
      y: rawSeries[labelIndex],
      text: rawSeries[labelIndex].map((value) => (value > 0 ? String(value) : "")),
      textposition: "inside",
      insidetextanchor: "middle",
      textfont: { color: "#ffffff", size: 9 },
      marker: { color: colors[labelIndex % colors.length] },
      hovertemplate:
        "%{x}<br>%{fullData.name}: %{y} occurrences<extra></extra>",
    }));
    const maxTaskTotal = Math.max(1, ...taskTotals);
    window.Plotly.newPlot(
      plot,
      traces,
      {
        height: 300,
        margin: { l: 44, r: 10, t: 8, b: 42 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {
          family: '"Fira Sans", system-ui, sans-serif',
          color: "#1e293b",
          size: 11,
        },
        barmode: "stack",
        xaxis: {
          fixedrange: true,
          gridcolor: "#f1f5f9",
          zeroline: false,
        },
        yaxis: {
          title: { text: "Occurrences", font: { size: 11 } },
          rangemode: "tozero",
          dtick: Math.max(1, Math.ceil(maxTaskTotal / 6)),
          fixedrange: true,
          gridcolor: "#e2e8f0",
          zerolinecolor: "#94a3b8",
        },
        legend: {
          orientation: "h",
          x: 0,
          y: 1.16,
          font: { size: 9 },
          traceorder: "normal",
        },
        hovermode: "closest",
      },
      {
        responsive: true,
        displayModeBar: false,
        displaylogo: false,
      }
    );
  }

  function occurrenceDistributionChart(
    plotId,
    title,
    description,
    runs,
    labels,
    listKey,
    countKey
  ) {
    const ids = labels.map((item) => item.id);
    const taskNumbers = [
      ...new Set(
        runs
          .map((run) => run.task_number)
          .filter((task) => typeof task === "number")
      ),
    ].sort((left, right) => left - right);
    const taskGroups = taskNumbers.map((task) => {
      const entries = runs
        .filter((run) => run.task_number === task)
        .map((run) => ({
          run,
          value: runOccurrenceTotal(run, listKey, countKey, ids),
        }));
      return {
        label: `Task ${task}`,
        taskNumber: task,
        entries,
        stats: distributionStats(entries.map((entry) => entry.value)),
      };
    });
    const allEntries = taskGroups.flatMap((group) => group.entries);
    const overall = distributionStats(allEntries.map((entry) => entry.value));
    if (!overall) return "";
    const summary = `n=${overall.count} · mean ${overall.mean.toFixed(2)} · median ${overall.median.toFixed(2)} · Q1 ${overall.q1.toFixed(2)} · Q3 ${overall.q3.toFixed(2)} · σ ${overall.standardDeviation.toFixed(2)} · range ${overall.minimum}–${overall.maximum}`;
    const tableGroups = [
      {
        label: "Overall",
        taskNumber: null,
        entries: allEntries,
        stats: overall,
      },
      ...taskGroups,
    ];
    const tableRows = tableGroups
      .map((group) => {
        const stats = group.stats;
        return `
          <tr>
            <th scope="row">${overviewGroupLabel(group)}</th>
            <td>${stats.count}</td>
            <td>${stats.mean.toFixed(2)}</td>
            <td>${stats.median.toFixed(2)}</td>
            <td>${stats.q1.toFixed(2)}</td>
            <td>${stats.q3.toFixed(2)}</td>
            <td>${stats.standardDeviation.toFixed(2)}</td>
            <td>${stats.minimum}</td>
            <td>${stats.maximum}</td>
            <td>${group.entries.map((entry) => entry.value).join(", ")}</td>
          </tr>
        `;
      })
      .join("");
    return `
      <section class="overview-section percentage-card distribution-card">
        <h3>${escapeHtml(title)}</h3>
        <p class="help">${escapeHtml(description)} Each point is one run’s summed occurrence counts; saved repeated counts are included, while unannotated runs use one count per auto-inferred tag.</p>
        <p class="distribution-summary"><strong>Overall:</strong> ${escapeHtml(summary)}</p>
        <div id="${escapeHtml(plotId)}" class="plotly-distribution-chart" role="img" aria-label="${escapeHtml(`${title}. ${summary}`)}"></div>
        <details class="chart-data-details">
          <summary>View statistics and run values</summary>
          <div class="chart-table-wrap">
            <table class="overview-data-table">
              <thead><tr><th>Group</th><th>n</th><th>Mean</th><th>Median</th><th>Q1</th><th>Q3</th><th>Population σ</th><th>Min</th><th>Max</th><th>Run values</th></tr></thead>
              <tbody>${tableRows}</tbody>
            </table>
          </div>
        </details>
      </section>
    `;
  }

  function renderOccurrenceDistributionPlot(
    plotId,
    runs,
    labels,
    listKey,
    countKey,
    color
  ) {
    const plot = document.getElementById(plotId);
    if (!plot) return;
    if (!window.Plotly) {
      plot.textContent = "Plotly could not be loaded. Use the values table below.";
      plot.classList.add("plotly-load-error");
      return;
    }
    const ids = labels.map((item) => item.id);
    const taskNumbers = [
      ...new Set(
        runs
          .map((run) => run.task_number)
          .filter((task) => typeof task === "number")
      ),
    ].sort((left, right) => left - right);
    const taskLabels = taskNumbers.map((task) => `Task ${task}`);
    const annotatedX = [];
    const annotatedY = [];
    const annotatedText = [];
    const autoX = [];
    const autoY = [];
    const autoText = [];
    const means = [];
    const deviations = [];
    const boxTraces = taskNumbers.map((task, index) => {
      const taskRuns = runs.filter((run) => run.task_number === task);
      const values = taskRuns.map((run, runIndex) => {
        const value = runOccurrenceTotal(run, listKey, countKey, ids);
        const jitter =
          taskRuns.length <= 1
            ? 0
            : (runIndex / (taskRuns.length - 1) - 0.5) * 0.48;
        const hover = `${run.run_id}<br>${value} occurrence${value === 1 ? "" : "s"}<br>${run.annotated ? "Saved annotation" : "Auto-inferred"}`;
        if (run.annotated) {
          annotatedX.push(index + jitter);
          annotatedY.push(value);
          annotatedText.push(hover);
        } else {
          autoX.push(index + jitter);
          autoY.push(value);
          autoText.push(hover);
        }
        return value;
      });
      const stats = distributionStats(values);
      means.push(stats.mean);
      deviations.push(stats.standardDeviation);
      return {
        type: "box",
        x: Array(values.length).fill(index),
        y: values,
        name: "Q1–Q3 / median",
        boxpoints: false,
        width: 0.42,
        quartilemethod: "linear",
        fillcolor: `${color}24`,
        line: { color, width: 1.5 },
        whiskerwidth: 0.65,
        hovertemplate: `${taskLabels[index]}<br>Q1: ${stats.q1.toFixed(2)}<br>Median: ${stats.median.toFixed(2)}<br>Q3: ${stats.q3.toFixed(2)}<extra></extra>`,
        showlegend: index === 0,
      };
    });
    const traces = [
      ...boxTraces,
      {
        type: "scatter",
        mode: "markers",
        name: "Saved annotation",
        x: annotatedX,
        y: annotatedY,
        text: annotatedText,
        hovertemplate: "%{text}<extra></extra>",
        marker: { color, size: 6, opacity: 0.82 },
      },
      {
        type: "scatter",
        mode: "markers",
        name: "Auto-inferred",
        x: autoX,
        y: autoY,
        text: autoText,
        hovertemplate: "%{text}<extra></extra>",
        marker: {
          color: "#ffffff",
          line: { color, width: 2 },
          size: 7,
          symbol: "circle-open",
        },
      },
      {
        type: "scatter",
        mode: "markers",
        name: "Mean ± 1σ",
        x: taskNumbers.map((_task, index) => index),
        y: means,
        error_y: {
          type: "data",
          array: deviations,
          visible: true,
          color: "#111827",
          thickness: 1.5,
          width: 4,
        },
        hovertemplate:
          "Mean: %{y:.2f}<br>Population σ: %{error_y.array:.2f}<extra></extra>",
        marker: { color: "#111827", size: 7, symbol: "diamond" },
      },
    ];
    const maxValue = Math.max(1, ...annotatedY, ...autoY);
    window.Plotly.newPlot(
      plot,
      traces,
      {
        height: 285,
        margin: { l: 44, r: 10, t: 8, b: 42 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {
          family: '"Fira Sans", system-ui, sans-serif',
          color: "#1e293b",
          size: 11,
        },
        xaxis: {
          tickmode: "array",
          tickvals: taskNumbers.map((_task, index) => index),
          ticktext: taskLabels,
          fixedrange: true,
          gridcolor: "#f1f5f9",
          zeroline: false,
        },
        yaxis: {
          title: { text: "Occurrences per run", font: { size: 11 } },
          range: [-0.25, maxValue + 0.45],
          dtick: 1,
          fixedrange: true,
          gridcolor: "#e2e8f0",
          zerolinecolor: "#94a3b8",
        },
        legend: {
          orientation: "h",
          x: 0,
          y: 1.12,
          font: { size: 10 },
        },
        hovermode: "closest",
        boxmode: "overlay",
      },
      {
        responsive: true,
        displayModeBar: false,
        displaylogo: false,
      }
    );
  }

  function shortTagLabel(label) {
    return String(label || "").split(" — ")[0];
  }

  function topPrevalence(prevalence) {
    return [...prevalence.rows].sort(
      (left, right) => right.count - left.count || left.label.localeCompare(right.label)
    )[0];
  }

  function resultsNarrative(outcomes, issues, criteria, failures, inefficiencies) {
    const sentences = [];
    if (outcomes.denominator) {
      sentences.push(
        `${formatPercent(outcomes.counts.complete / outcomes.denominator)} of eligible runs completed every criterion, ` +
          `${formatPercent(outcomes.counts.partial / outcomes.denominator)} made partial progress, and ` +
          `${formatPercent(outcomes.counts.noProgress / outcomes.denominator)} made no progress.`
      );
    }
    if (criteria.total) {
      sentences.push(
        `Across all criteria, ${formatPercent(criteria.ratio)} were completed.`
      );
    }
    if (issues.denominator) {
      sentences.push(
        `${formatPercent(issues.counts.none / issues.denominator)} of annotated runs had no recorded failure or inefficiency.`
      );
    }
    const topFailure = topPrevalence(failures);
    const topInefficiency = topPrevalence(inefficiencies);
    if (topFailure?.count) {
      sentences.push(
        `The most common failure was ${shortTagLabel(topFailure.label).toLowerCase()}, affecting ${formatPercent(topFailure.ratio)} of annotated runs.`
      );
    }
    if (topInefficiency?.count) {
      sentences.push(
        `The most common inefficiency was ${shortTagLabel(topInefficiency.label).toLowerCase()}, affecting ${formatPercent(topInefficiency.ratio)} of annotated runs.`
      );
    }
    return sentences.join(" ");
  }

  function renderOverview() {
    if (!els.overviewBody || !state.overview) return;
    const runs = overviewRuns();
    const batchName = state.batch || "all folders (deduped)";
    if (els.overviewNote) {
      els.overviewNote.textContent =
        `${batchName} · BASE/SUB/ABS/CON/DIS only · outcomes use runs with criteria · ` +
        "issue percentages use saved annotations · tags may overlap";
    }
    if (!runs.length) {
      els.overviewBody.innerHTML = `<p class="empty-state">No runs in this folder.</p>`;
      return;
    }

    const tasks = [
      ...new Set(
        runs
          .map((run) => run.task_number)
          .filter((item) => typeof item === "number")
      ),
    ].sort((left, right) => left - right);
    const taskGroups = tasks.map((task) => ({
      label: `Task ${task}`,
      taskNumber: task,
      runs: runs.filter((run) => run.task_number === task),
    }));
    const memoryGroups = AXIS_ORDER.map((axis) => ({
      label: AXIS_FULL[axis] || axis,
      runs: runs.filter((run) => run.axis === axis),
    })).filter((group) => group.runs.length);
    const uncertaintyGroups = OVERVIEW_KIND_ORDER.map((kind) => ({
      label: KIND_FULL[kind] || kind,
      runs: runs.filter((run) => run.kind === kind),
    })).filter((group) => group.runs.length);

    const outcomes = outcomeBreakdown(runs);
    const issues = issueBreakdown(runs);
    const criteria = criteriaBreakdown(runs);
    const failureLabels = state.meta?.failure_labels || [];
    const inefficiencyLabels = state.meta?.efficiency_labels || [];
    const failurePrevalence = tagPrevalence(
      runs,
      failureLabels,
      "failure_reasons",
      "failure_reason_counts"
    );
    const inefficiencyPrevalence = tagPrevalence(
      runs,
      inefficiencyLabels,
      "efficiency_errors",
      "efficiency_error_counts"
    );
    const anyProgress = outcomes.counts.complete + outcomes.counts.partial;
    const annotated = issues.denominator;
    const issueFree = issues.counts.none;
    const unknownNote = outcomes.unknown
      ? `${outcomes.unknown} zero-criterion run${outcomes.unknown === 1 ? "" : "s"} excluded`
      : "runs with one or more criteria";

    els.overviewBody.innerHTML = `
      <section class="overview-kpi-grid" aria-label="Results summary">
        ${overviewKpi(
          "Complete runs",
          outcomes.counts.complete,
          outcomes.denominator,
          unknownNote
        )}
        ${overviewKpi(
          "Runs with progress",
          anyProgress,
          outcomes.denominator,
          "completed at least one criterion"
        )}
        ${overviewKpi(
          "Issue-free runs",
          issueFree,
          annotated,
          "among annotated runs"
        )}
        ${overviewKpi(
          "Annotation coverage",
          annotated,
          runs.length,
          "saved human annotations"
        )}
      </section>

      <section class="overview-story" aria-labelledby="results-story-heading">
        <p class="eyebrow">Results story</p>
        <h3 id="results-story-heading">What the evaluation shows</h3>
        <p>${escapeHtml(
          resultsNarrative(
            outcomes,
            issues,
            criteria,
            failurePrevalence,
            inefficiencyPrevalence
          )
        )}</p>
      </section>

      <div class="overview-pair">
        ${stackedPercentageChart(
          "Run outcomes by task",
          "Shows whether runs completed all criteria, made partial progress, or made no progress. Each eligible run is classified once from its saved completed-versus-total criteria, then divided by eligible runs in that task.",
          taskGroups,
          OUTCOME_SEGMENTS,
          outcomeBreakdown
        )}
        ${stackedPercentageChart(
          "Issue overlap by task",
          "Shows whether annotated runs had no issue, only a failure reason, only an inefficiency, or both. Each run is classified once from its saved annotation tags, then divided by annotated runs in that task.",
          taskGroups,
          ISSUE_SEGMENTS,
          issueBreakdown
        )}
      </div>

      ${issueStoryChart(
        "issue-story-plot",
        taskGroups,
        failureLabels,
        inefficiencyLabels
      )}

      ${partialProgressChart(taskGroups)}

      <div class="overview-pair">
        ${stackedPercentageChart(
          "Run outcomes by memory",
          "Compares complete, partial, and no-progress outcomes under Accurate, Incomplete, and Outdated memory. Percentages use saved criteria results divided by eligible runs in each memory condition.",
          memoryGroups,
          OUTCOME_SEGMENTS,
          outcomeBreakdown
        )}
        ${stackedPercentageChart(
          "Run outcomes by uncertainty",
          "Compares run outcomes across Base, Substitute, Absent, Containment, and Distractor conditions. Percentages use saved criteria results divided by eligible runs in each uncertainty condition.",
          uncertaintyGroups,
          OUTCOME_SEGMENTS,
          outcomeBreakdown
        )}
      </div>

      <div class="overview-pair">
        ${prevalenceChart(
          "Why runs failed",
          "Shows how often each saved failure reason appears. Each bar is unique annotated runs with that tag divided by all annotated runs in scope; runs may have multiple reasons, so bars do not sum to 100%.",
          failurePrevalence,
          "is-failure"
        )}
        ${prevalenceChart(
          "Inefficiencies",
          "Shows how often each saved inefficiency appears. Each bar is unique annotated runs with that tag divided by all annotated runs in scope; runs may have multiple tags, so bars do not sum to 100%.",
          inefficiencyPrevalence,
          "is-inefficiency"
        )}
      </div>

      <div class="overview-pair distribution-pair">
        ${occurrenceByTypeChart(
          "failure-type-count-plot",
          "Failure types by task",
          "Shows the number and composition of failure occurrences for each task.",
          runs,
          failureLabels,
          "failure_reasons",
          "failure_reason_counts"
        )}
        ${occurrenceByTypeChart(
          "inefficiency-type-count-plot",
          "Inefficiency types by task",
          "Shows the number and composition of inefficiency occurrences for each task.",
          runs,
          inefficiencyLabels,
          "efficiency_errors",
          "efficiency_error_counts"
        )}
      </div>

      <div class="overview-pair distribution-pair">
        ${occurrenceDistributionChart(
          "failure-distribution-plot",
          "Failure occurrences by run and task",
          "Shows the spread of total failure occurrences per run, grouped by task.",
          runs,
          failureLabels,
          "failure_reasons",
          "failure_reason_counts"
        )}

        ${occurrenceDistributionChart(
          "inefficiency-distribution-plot",
          "Inefficiency occurrences by run and task",
          "Shows the spread of total inefficiency occurrences per run, grouped by task.",
          runs,
          inefficiencyLabels,
          "efficiency_errors",
          "efficiency_error_counts"
        )}
      </div>
    `;
    renderIssueStoryPlot(
      "issue-story-plot",
      taskGroups,
      failureLabels,
      inefficiencyLabels
    );
    renderOccurrenceByTypePlot(
      "failure-type-count-plot",
      runs,
      failureLabels,
      "failure_reasons",
      "failure_reason_counts",
      ["#b91c1c", "#ea580c", "#7c3aed", "#be185d", "#475569"]
    );
    renderOccurrenceByTypePlot(
      "inefficiency-type-count-plot",
      runs,
      inefficiencyLabels,
      "efficiency_errors",
      "efficiency_error_counts",
      ["#1d4ed8", "#0891b2", "#0f766e", "#7c3aed"]
    );
    renderOccurrenceDistributionPlot(
      "failure-distribution-plot",
      runs,
      failureLabels,
      "failure_reasons",
      "failure_reason_counts",
      "#b91c1c"
    );
    renderOccurrenceDistributionPlot(
      "inefficiency-distribution-plot",
      runs,
      inefficiencyLabels,
      "efficiency_errors",
      "efficiency_error_counts",
      "#1d4ed8"
    );
  }

  function setOverview(open, push) {
    state.overview = Boolean(open);
    document.body.classList.toggle("is-overview", state.overview);
    if (els.overviewPanel) els.overviewPanel.hidden = !state.overview;
    if (els.overviewBtn) {
      els.overviewBtn.setAttribute("aria-pressed", state.overview ? "true" : "false");
      els.overviewBtn.classList.toggle("is-active", state.overview);
    }
    if (state.overview) renderOverview();
    writeUrl(state.currentId, push);
  }

  function renderProgress() {
    const scoped = batchRuns();
    const total = scoped.length;
    const done = scoped.filter((run) => run.annotated).length;
    const batchName = state.batch || "all folders";
    const taskLabel = state.taskNumber != null ? ` · task ${state.taskNumber}` : "";
    els.progress.textContent = `${done} of ${total} annotated · ${batchName}${taskLabel}`;
    renderTaskMetrics();
  }

  function renderBatchSelect() {
    const batches = state.meta?.batches || [];
    const options = [`<option value="">All folders</option>`].concat(
      batches.map((batch) => {
        const selected = batch.id === state.batch ? "selected" : "";
        const visibleCount = state.runs.filter(
          (run) => run.batch === batch.id && isFrontendVisibleRun(run)
        ).length;
        return `<option value="${escapeHtml(batch.id)}" ${selected}>${escapeHtml(batch.label)} (${visibleCount})</option>`;
      })
    );
    els.batchSelect.innerHTML = options.join("");
  }

  function renderTaskFilters() {
    const tasks = taskNumbers();
    const chips = [
      {
        value: "",
        label: "All tasks",
        active: state.taskNumber == null,
      },
    ].concat(
      tasks.map((task) => ({
        value: String(task),
        label: `Task ${task}`,
        active: state.taskNumber === task,
      }))
    );
    els.taskFilters.innerHTML = chips
      .map(
        (chip) => `
          <button type="button" class="chip${chip.active ? " is-active" : ""}" data-task="${escapeHtml(chip.value)}" aria-pressed="${chip.active ? "true" : "false"}">${escapeHtml(chip.label)}</button>
        `
      )
      .join("");
  }

  function runItemHtml(run) {
    const active = run.run_id === state.currentId ? " is-active" : "";
    const status = run.annotated ? "Annotated" : "Unannotated";
    return `
      <a class="run-item${active}" href="?batch=${encodeURIComponent(state.batch)}&task=${state.taskNumber ?? ""}&run=${encodeURIComponent(run.run_id)}" data-run-id="${escapeHtml(run.run_id)}">
        <span class="run-item-id">${escapeHtml(runLabel(run))}</span>
        <span class="run-item-status">${escapeHtml(run.episode_id || run.variant_folder)} · ${status}</span>
      </a>
    `;
  }

  function renderRunList() {
    const runs = filteredRuns();
    els.runCount.textContent = `${runs.length} shown`;
    renderTaskFilters();
    if (!runs.length) {
      els.runList.innerHTML = `<p class="muted">No runs match this filter.</p>`;
      return;
    }
    if (state.taskNumber != null) {
      els.runList.innerHTML = runs.map(runItemHtml).join("");
      return;
    }
    const groups = [];
    let currentKey = null;
    for (const run of runs) {
      const taskTitle = typeof run.task_number === "number" ? `Task ${run.task_number}` : "Other";
      const key = state.batch ? taskTitle : `${run.batch || "(root)"} · ${taskTitle}`;
      if (key !== currentKey) {
        currentKey = key;
        groups.push(`<h3 class="run-group-title">${escapeHtml(key)}</h3>`);
      }
      groups.push(runItemHtml(run));
    }
    els.runList.innerHTML = groups.join("");
  }

  function renderChips(run) {
    const success = run.task_state_success === 1;
    const pddl = run.pddl || {};
    const metrics = pddl.metrics || {};
    const chips = [
      run.planner === "vlm_tamp_pddl"
        ? `<span class="outcome">VLM-TAMP PDDL</span>`
        : "",
      `<span class="outcome">${success ? "Evaluator success" : "Evaluator fail"} · ${formatPercent(run.task_state_success)}</span>`,
      `<span class="outcome">complete ${formatPercent(run.task_percent_complete)}</span>`,
      `<span class="outcome">${run.total_steps} steps</span>`,
      run.has_images
        ? `<span class="outcome">${run.image_count} camera frames</span>`
        : "",
      `<span class="outcome">runtime ${formatSeconds(run.runtime)}</span>`,
    ];
    if (metrics.llm_model) {
      const effort = metrics.llm_reasoning_effort
        ? ` ${metrics.llm_reasoning_effort}`
        : "";
      chips.push(
        `<span class="outcome">${escapeHtml(metrics.llm_model)}${escapeHtml(effort)}</span>`
      );
    }
    if (typeof metrics.llm_usd === "number") {
      chips.push(`<span class="outcome">${formatUsd(metrics.llm_usd)}</span>`);
    }
    if (typeof metrics.prompt_tokens === "number") {
      const cached =
        typeof metrics.cached_tokens === "number"
          ? ` · cached ${formatCount(metrics.cached_tokens)}`
          : "";
      chips.push(
        `<span class="outcome">${formatCount(metrics.prompt_tokens)} in / ${formatCount(metrics.completion_tokens)} out${cached}</span>`
      );
    }
    els.chips.innerHTML = chips.filter(Boolean).join("");
  }

  function renderPddlPanel(run) {
    const pddl = run?.pddl;
    if (!els.pddlPanel || !els.pddlFrame) return;
    if (!pddl || !pddl.graph_url) {
      els.timeline.classList.remove("has-pddl");
      els.pddlPanel.hidden = true;
      els.pddlFrame.removeAttribute("src");
      if (els.pddlLinks) els.pddlLinks.innerHTML = "";
      return;
    }
    els.timeline.classList.add("has-pddl");
    els.pddlPanel.hidden = false;
    const nextSrc = new URL(pddl.graph_url, window.location.origin).href;
    if (els.pddlFrame.src !== nextSrc) {
      els.pddlFrame.src = pddl.graph_url;
    }
    const links = [
      `<a class="outcome" href="${escapeHtml(pddl.graph_url)}" target="_blank" rel="noopener">Open tree</a>`,
    ];
    if (pddl.prompts_url) {
      links.push(
        `<a class="outcome" href="${escapeHtml(pddl.prompts_url)}" target="_blank" rel="noopener">VLM chat</a>`
      );
    }
    if (pddl.prompts_txt_url) {
      links.push(
        `<a class="outcome" href="${escapeHtml(pddl.prompts_txt_url)}" target="_blank" rel="noopener">Raw prompts</a>`
      );
    }
    if (els.pddlLinks) els.pddlLinks.innerHTML = links.join("");
  }

  function visibleSteps() {
    const steps = state.detail?.steps || [];
    if (state.stepFilter === "failed") return steps.filter((step) => !step.success);
    if (state.stepFilter === "ok") return steps.filter((step) => step.success);
    return steps;
  }

  function inspectStep(index) {
    const steps = visibleSteps();
    const step = steps.find((item) => item.index === index) || steps[0] || null;
    state.inspectedIndex = step ? step.index : null;
    els.stepList.querySelectorAll(".step").forEach((node) => {
      const active = Number(node.dataset.stepIndex) === state.inspectedIndex;
      node.classList.toggle("is-active", active);
      node.setAttribute("aria-selected", active ? "true" : "false");
    });
    if (!step) {
      els.inspectorEmpty.hidden = false;
      els.inspectorBody.hidden = true;
      els.inspector.classList.remove("is-fail");
      hideInspectorImage();
      return;
    }
    const ok = Boolean(step.success);
    const action = `${step.action}[${step.args}]`;
    els.inspectorEmpty.hidden = true;
    els.inspectorBody.hidden = false;
    els.inspector.classList.toggle("is-fail", !ok);
    const subgoal = step.subgoal ? ` · ${step.subgoal}` : "";
    els.inspectorMeta.textContent = `Step ${step.index} of ${state.detail?.total_steps ?? steps.length}${subgoal}`;
    els.inspectorAction.textContent = action;
    els.inspectorBadge.textContent = ok ? "Succeeded" : "Failed";
    els.inspectorBadge.className = `badge ${ok ? "badge-ok" : "badge-fail"}`;
    els.inspectorThought.textContent = step.thought || "No thought recorded.";
    els.inspectorResult.textContent = step.result || "No result recorded.";
    renderWorldGraph(step);
    showInspectorImage(step);
  }

  function formatObjectStates(states) {
    const entries = Object.entries(states || {});
    if (!entries.length) return "";
    return ` (${entries.map(([key, value]) => `${key}:${value}`).join(", ")})`;
  }

  function renderWorldGraph(step) {
    const el = els.inspectorGraph;
    if (!el) return;
    const graph = state.detail?.world_graph || {};
    const rooms = graph.rooms || [];
    const faucets = new Set(graph.faucets || []);
    const objects =
      step?.object_states && step.object_states.length
        ? step.object_states
        : graph.objects || [];
    if (!rooms.length && !objects.length) {
      el.innerHTML = `<p class="wg-empty">No world graph for this run.</p>`;
      return;
    }
    const byFurniture = new Map();
    const held = [];
    const loose = [];
    for (const obj of objects) {
      if (obj.held) {
        held.push(obj);
      } else if (obj.location) {
        if (!byFurniture.has(obj.location)) byFurniture.set(obj.location, []);
        byFurniture.get(obj.location).push(obj);
      } else {
        loose.push(obj);
      }
    }
    const usedFurniture = new Set();
    const groups = [];
    function roomGroup(title, body) {
      groups.push(
        `<div class="wg-group"><div class="wg-room">${escapeHtml(title)}</div>${body}</div>`
      );
    }
    function furnitureLine(names) {
      if (!names.length) return "";
      return `<div class="wg-furn-line">${names.join(", ")}</div>`;
    }
    function objectLine(obj, extra) {
      const loc = extra ? ` @ ${extra}` : "";
      const cls = obj.held ? "wg-obj is-held" : "wg-obj";
      return `<div class="${cls}">${escapeHtml(obj.name)}${escapeHtml(loc)}${escapeHtml(
        formatObjectStates(obj.states)
      )}</div>`;
    }
    for (const room of rooms) {
      const furnParts = [];
      const objLines = [];
      for (const furn of room.furniture || []) {
        usedFurniture.add(furn);
        const faucet = faucets.has(furn)
          ? ` <span class="wg-faucet">faucet</span>`
          : "";
        furnParts.push(`${escapeHtml(furn)}${faucet}`);
        for (const obj of byFurniture.get(furn) || []) {
          objLines.push(objectLine(obj, furn));
        }
      }
      roomGroup(room.name, furnitureLine(furnParts) + objLines.join(""));
    }
    if (held.length) {
      roomGroup("Held by agent", held.map((obj) => objectLine(obj, "")).join(""));
    }
    const leftover = [...loose];
    for (const [location, objs] of byFurniture) {
      if (!usedFurniture.has(location)) {
        leftover.push(...objs.map((obj) => ({ ...obj, location })));
      }
    }
    if (leftover.length) {
      roomGroup(
        "Other",
        leftover.map((obj) => objectLine(obj, obj.location || "")).join("")
      );
    }
    el.innerHTML = groups.join("");
  }

  function hideInspectorImage() {
    if (!els.inspectorImageWrap || !els.inspectorImage) return;
    els.inspectorImageWrap.hidden = true;
    els.inspectorImage.removeAttribute("src");
    els.inspectorImage.alt = "";
  }

  function showInspectorImage(step) {
    if (!els.inspectorImageWrap || !els.inspectorImage) return;
    if (!step.image_url) {
      hideInspectorImage();
      return;
    }
    const caption = `Camera · step ${step.index}`;
    if (els.inspectorImageCaption) els.inspectorImageCaption.textContent = caption;
    els.inspectorImage.alt = `Robot camera for step ${step.index}`;
    els.inspectorImage.onload = () => {
      els.inspectorImageWrap.hidden = false;
    };
    els.inspectorImage.onerror = () => {
      hideInspectorImage();
    };
    if (els.inspectorImage.getAttribute("src") !== step.image_url) {
      els.inspectorImage.src = step.image_url;
    }
    if (els.inspectorImage.complete && els.inspectorImage.naturalWidth > 0) {
      els.inspectorImageWrap.hidden = false;
    }
  }

  function renderSteps() {
    const steps = visibleSteps();
    const timelineList = document.querySelector(".timeline-list");
    if (timelineList) timelineList.scrollTop = 0;
    if (!steps.length) {
      els.stepList.innerHTML = `<li class="muted">No steps in this filter.</li>`;
      inspectStep(null);
      return;
    }
    els.stepList.innerHTML = steps
      .map((step) => {
        const ok = Boolean(step.success);
        const action = `${step.action}[${step.args}]`;
        const preview = step.thought || step.result || "";
        const camera = step.image_url
          ? `<span class="step-cam" title="Has camera frame">cam</span>`
          : "";
        return `
          <li class="step ${ok ? "is-ok" : "is-fail"}" data-step-index="${step.index}" tabindex="0" role="option" aria-selected="false">
            <div class="step-head">
              <span class="step-num">${step.index}</span>
              <span class="step-action">${escapeHtml(action)}</span>
              <span class="step-preview">${escapeHtml(preview)}</span>
              ${camera}
              <span class="badge ${ok ? "badge-ok" : "badge-fail"}">${ok ? "Succeeded" : "Failed"}</span>
            </div>
          </li>
        `;
      })
      .join("");
    const stillVisible = steps.some((step) => step.index === state.inspectedIndex);
    const fallback = steps.find((step) => !step.success) || steps[0];
    inspectStep(stillVisible ? state.inspectedIndex : fallback.index);
  }

  function checkedCriteria() {
    return Array.from(
      els.criteriaList.querySelectorAll("input[type=checkbox]:checked")
    ).map((input) => input.value);
  }

  function allCriteriaChecked() {
    const boxes = els.criteriaList.querySelectorAll("input[type=checkbox]");
    return boxes.length > 0 && Array.from(boxes).every((box) => box.checked);
  }

  function updateBranching() {
    const done = checkedCriteria().length;
    const total = state.detail?.criteria?.length || 0;
    els.subtaskCount.textContent = `${done} / ${total} subtasks done`;
    if (els.failureFieldset) els.failureFieldset.hidden = false;
    if (els.efficiencyFieldset) els.efficiencyFieldset.hidden = false;
    renderTaskMetrics();
  }

  function reasonCount(counts, id) {
    const value = Number(counts?.[id]);
    if (!Number.isFinite(value) || value < 1) return 1;
    return Math.min(99, Math.round(value));
  }

  function renderChecks(container, items, name, selected, counts) {
    const chosen = new Set(selected || []);
    container.innerHTML = items
      .map((item, index) => {
        const id = `${name}-${index}`;
        const checked = chosen.has(item.id) ? "checked" : "";
        const count = reasonCount(counts, item.id);
        return `
          <div class="check-item check-item-count">
            <input id="${id}" type="checkbox" name="${name}" value="${escapeHtml(item.id)}" ${checked} />
            <label for="${id}">${escapeHtml(item.label)}</label>
            <input class="reason-count" type="number" min="1" max="99" step="1" value="${count}" aria-label="Count for ${escapeHtml(item.label)}" />
          </div>
        `;
      })
      .join("");
  }

  function renderCriteria() {
    const criteria = state.detail.criteria || [];
    const selected = state.detail.annotation?.criteria || {};
    els.criteriaList.innerHTML = criteria
      .map((item, index) => {
        const id = `criterion-${index}`;
        const checked = selected[item] ? "checked" : "";
        return `
          <label class="check-item" for="${id}">
            <input id="${id}" type="checkbox" name="criterion" value="${escapeHtml(item)}" ${checked} />
            <span>${escapeHtml(item)}</span>
          </label>
        `;
      })
      .join("");
  }

  function renderForm() {
    if (!state.detail) {
      els.form.hidden = true;
      els.formEmpty.hidden = false;
      return;
    }
    els.form.hidden = false;
    els.formEmpty.hidden = true;
    renderCriteria();
    if (els.evalFillNote) {
      const autoFilled = Boolean(state.detail.annotation?.auto_filled);
      const explanation = (state.detail.task_explanation || "").trim();
      if (autoFilled && explanation) {
        els.evalFillNote.hidden = false;
        els.evalFillNote.textContent = `Pre-filled from the run evaluator and the last Objects snapshot in the trace.\n${explanation}`;
        els.evalFillNote.title = explanation;
      } else if (autoFilled) {
        els.evalFillNote.hidden = false;
        els.evalFillNote.textContent = "Pre-filled from the run evaluator and the last Objects snapshot in the trace. Change a box if it is wrong.";
        els.evalFillNote.title = "";
      } else {
        els.evalFillNote.hidden = true;
        els.evalFillNote.textContent = "";
        els.evalFillNote.title = "";
      }
    }
    if (els.errorFillNote) {
      if (state.detail.annotation?.auto_error_filled) {
        els.errorFillNote.hidden = false;
        els.errorFillNote.textContent =
          "Failed-step tags were filled from recurring skill errors in this trace. Change a box if it is wrong.";
      } else {
        els.errorFillNote.hidden = true;
        els.errorFillNote.textContent = "";
      }
    }
    renderChecks(
      els.failureList,
      state.meta.failure_labels,
      "failure",
      state.detail.annotation?.failure_reasons,
      state.detail.annotation?.failure_reason_counts
    );
    renderChecks(
      els.efficiencyList,
      state.meta.efficiency_labels,
      "efficiency",
      state.detail.annotation?.efficiency_errors,
      state.detail.annotation?.efficiency_error_counts
    );
    els.comments.value =
      state.detail.annotation?.comments ||
      state.detail.annotation?.failure_notes ||
      "";
    els.errorSummary.hidden = true;
    els.failureError.hidden = true;
    state.ignoreAutosave = true;
    setSaveStatus(
      state.detail.annotation?.updated_at
        ? `Saved ${state.detail.annotation.updated_at}`
        : "Saves automatically.",
      state.detail.annotation?.updated_at ? "ok" : ""
    );
    updateBranching();
    state.dirty = false;
    state.lastSavedKey = payloadKey(collectPayload());
    state.ignoreAutosave = false;
  }

  function renderDetail() {
    if (!state.detail) {
      els.timeline.hidden = true;
      els.timelineEmpty.hidden = false;
      state.inspectedIndex = null;
      renderPddlPanel(null);
      renderForm();
      return;
    }
    els.timeline.hidden = false;
    els.timelineEmpty.hidden = true;
    if (els.variantSummary) {
      els.variantSummary.textContent =
        state.detail.variant_summary ||
        [state.detail.axis, state.detail.kind].filter(Boolean).join("-") ||
        "";
    }
    els.runMeta.textContent = `${state.detail.episode_id || ""} · ${state.detail.run_id}`;
    els.heading.textContent = state.detail.instruction || state.detail.task || state.detail.episode_id;
    renderPddlPanel(state.detail);
    renderChips(state.detail);
    renderSteps();
    renderForm();
    renderTaskMetrics();
  }

  function collectPayload() {
    const criteria = {};
    for (const item of state.detail.criteria || []) {
      criteria[item] = false;
    }
    for (const value of checkedCriteria()) {
      criteria[value] = true;
    }
    const failureReasons = Array.from(
      els.failureList.querySelectorAll("input[type=checkbox]:checked")
    ).map((input) => input.value);
    const efficiencyErrors = Array.from(
      els.efficiencyList.querySelectorAll("input[type=checkbox]:checked")
    ).map((input) => input.value);
    const comments = (els.comments?.value || "").trim();
    return {
      run_id: state.currentId,
      episode_id: state.detail.episode_id,
      criteria,
      failure_reasons: failureReasons,
      failure_reason_counts: collectReasonCounts(els.failureList),
      failure_notes: comments,
      comments,
      efficiency_errors: efficiencyErrors,
      efficiency_error_counts: collectReasonCounts(els.efficiencyList),
    };
  }

  function collectReasonCounts(listEl) {
    const counts = {};
    if (!listEl) return counts;
    listEl.querySelectorAll(".check-item-count").forEach((row) => {
      const box = row.querySelector("input[type=checkbox]");
      if (!box || !box.checked) return;
      const number = row.querySelector("input[type=number]");
      const parsed = Number.parseInt(number?.value || "1", 10);
      counts[box.value] = Number.isFinite(parsed) && parsed >= 1 ? Math.min(99, parsed) : 1;
    });
    return counts;
  }

  function showClientErrors(errors) {
    els.errorSummary.hidden = false;
    els.errorSummary.innerHTML = `
      <h3 id="error-title">There is a problem</h3>
      <ul>
        ${errors
          .map(
            (error) =>
              `<li><a href="#${error.href}">${escapeHtml(error.message)}</a></li>`
          )
          .join("")}
      </ul>
    `;
    els.errorSummary.focus();
    const failure = errors.find((error) => error.field === "failure_reasons");
    if (failure) {
      els.failureError.hidden = false;
      els.failureError.id = "failure-error";
      els.failureError.textContent = failure.message;
    }
  }

  function clientValidate(payload) {
    const errors = [];
    const total = state.detail.criteria.length;
    const done = Object.values(payload.criteria).filter(Boolean).length;
    if (total === 0) {
      errors.push({
        field: "criteria",
        href: "criteria-list",
        message: "No success criteria were found for this run.",
      });
    } else if (done < total) {
      if (!payload.failure_reasons.length && !payload.comments && !payload.failure_notes) {
        errors.push({
          field: "failure_reasons",
          href: "comments",
          message: "Pick a failure reason or write why this run failed.",
        });
      }
    }
    return errors;
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error("Request failed");
      error.payload = data;
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function writeUrl(runId, push) {
    const url = new URL(window.location.href);
    if (state.batch) url.searchParams.set("batch", state.batch);
    else url.searchParams.delete("batch");
    if (state.taskNumber != null) url.searchParams.set("task", String(state.taskNumber));
    else url.searchParams.delete("task");
    if (state.overview) url.searchParams.set("view", "overview");
    else url.searchParams.delete("view");
    if (runId) url.searchParams.set("run", runId);
    else url.searchParams.delete("run");
    const method = push ? "pushState" : "replaceState";
    history[method]({ batch: state.batch, task: state.taskNumber, run: runId }, "", url);
  }

  function readUrlState() {
    const params = new URL(window.location.href).searchParams;
    const batch = params.get("batch") || "";
    const taskRaw = params.get("task");
    const taskNumber = taskRaw ? Number(taskRaw) : null;
    return {
      batch,
      taskNumber: Number.isFinite(taskNumber) ? taskNumber : null,
      run: params.get("run") || "",
      overview: params.get("view") === "overview",
    };
  }

  async function loadRun(runId, push) {
    if (!runId) return;
    const previous = state.loadQueue || Promise.resolve();
    const next = previous.catch(() => {}).then(() => loadRunNow(runId, push));
    state.loadQueue = next;
    return next;
  }

  async function loadRunNow(runId, push) {
    await flushAutosave();
    state.currentId = runId;
    state.inspectedIndex = null;
    state.dirty = false;
    state.lastSavedKey = "";
    writeUrl(runId, push);
    renderProgress();
    renderRunList();
    const detail = await fetchJson(`/api/run?id=${encodeURIComponent(runId)}`);
    if (state.currentId !== runId) return;
    state.detail = detail;
    const listed = state.runs.find((item) => item.run_id === runId);
    if (listed) {
      const criteria = detail.annotation?.criteria || {};
      listed.subtasks_done = Object.values(criteria).filter(Boolean).length;
      listed.subtasks_total = (detail.criteria || []).length;
      listed.task_percent_complete = detail.task_percent_complete;
      listed.task_state_success = detail.task_state_success;
    }
    renderDetail();
  }

  function ensureCurrentRun() {
    const visible = filteredRuns();
    if (visible.some((run) => run.run_id === state.currentId)) return;
    if (visible[0]) loadRun(visible[0].run_id, true);
  }

  function payloadKey(payload) {
    return JSON.stringify({
      run_id: payload.run_id,
      criteria: payload.criteria,
      failure_reasons: payload.failure_reasons,
      failure_reason_counts: payload.failure_reason_counts,
      comments: payload.comments,
      efficiency_errors: payload.efficiency_errors,
      efficiency_error_counts: payload.efficiency_error_counts,
    });
  }

  async function saveAnnotation(options = {}) {
    const autosave = Boolean(options.autosave);
    if (!state.detail || !state.currentId || els.form.hidden) return false;
    const runId = state.currentId;
    const payload = collectPayload();
    if (autosave && payloadKey(payload) === state.lastSavedKey) {
      state.dirty = false;
      return true;
    }
    if (!autosave) {
      els.errorSummary.hidden = true;
      els.failureError.hidden = true;
      els.saveBtn.disabled = true;
      els.saveNextBtn.disabled = true;
    }
    setSaveStatus("Saving…", "");
    try {
      const result = await fetchJson("/api/annotation", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (state.currentId !== runId) return true;
      const run = state.runs.find((item) => item.run_id === runId);
      if (run) {
        run.annotated = true;
        run.subtasks_done = Object.values(payload.criteria || {}).filter(Boolean).length;
        run.subtasks_total = (state.detail.criteria || []).length;
        run.subtasks_source = "annotation";
        run.failure_reasons = payload.failure_reasons || [];
        run.failure_reason_counts = payload.failure_reason_counts || {};
        run.efficiency_errors = payload.efficiency_errors || [];
        run.efficiency_error_counts = payload.efficiency_error_counts || {};
      }
      if (state.detail) state.detail.annotation = result.annotation;
      state.lastSavedKey = payloadKey(payload);
      state.dirty = false;
      renderProgress();
      renderRunList();
      setSaveStatus("Saved.", "ok");
      return true;
    } catch (error) {
      if (state.currentId !== runId) return false;
      if (!autosave) {
        const serverErrors = error.payload?.errors || [
          { field: "form", message: "Could not save this annotation.", href: "annotator" },
        ];
        showClientErrors(
          serverErrors.map((item) => ({
            ...item,
            href: item.field === "failure_reasons" ? "failure-list" : "annotator",
          }))
        );
      }
      setSaveStatus("Save failed.", "err");
      return false;
    } finally {
      if (!autosave) {
        els.saveBtn.disabled = false;
        els.saveNextBtn.disabled = false;
      }
    }
  }

  function scheduleAutosave(immediate) {
    if (state.ignoreAutosave || !state.detail || els.form.hidden) return;
    state.dirty = true;
    window.clearTimeout(state.saveTimer);
    if (immediate) {
      state.saveInFlight = saveAnnotation({ autosave: true });
      return;
    }
    state.saveTimer = window.setTimeout(() => {
      state.saveInFlight = saveAnnotation({ autosave: true });
    }, 400);
  }

  async function flushAutosave() {
    window.clearTimeout(state.saveTimer);
    if (state.saveInFlight) {
      try {
        await state.saveInFlight;
      } catch (_error) {
        /* keep going so navigation is not stuck */
      }
    }
    if (state.dirty) {
      await saveAnnotation({ autosave: true });
    }
  }

  function goRelative(delta) {
    const runs = filteredRuns();
    if (!runs.length) return;
    const index = currentIndex();
    const nextIndex =
      index < 0 ? 0 : Math.min(runs.length - 1, Math.max(0, index + delta));
    const next = runs[nextIndex];
    if (next && next.run_id !== state.currentId) {
      loadRun(next.run_id, true);
    }
  }

  function isTypingTarget(target) {
    if (!(target instanceof HTMLElement)) return false;
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  els.runList.addEventListener("click", (event) => {
    const link = event.target.closest("[data-run-id]");
    if (!link) return;
    event.preventDefault();
    loadRun(link.getAttribute("data-run-id"), true);
  });

  els.search.addEventListener("input", () => {
    state.search = els.search.value;
    renderRunList();
  });

  els.batchSelect.addEventListener("change", () => {
    state.batch = els.batchSelect.value;
    const tasks = taskNumbers();
    if (state.taskNumber != null && !tasks.includes(state.taskNumber)) {
      state.taskNumber = null;
    }
    renderProgress();
    renderRunList();
    ensureCurrentRun();
  });

  els.taskFilters.addEventListener("click", (event) => {
    const button = event.target.closest("[data-task]");
    if (!button) return;
    const raw = button.getAttribute("data-task") || "";
    state.taskNumber = raw === "" ? null : Number(raw);
    renderProgress();
    renderRunList();
    ensureCurrentRun();
  });

  document.querySelectorAll("[data-status]").forEach((button) => {
    button.addEventListener("click", () => {
      state.statusFilter = button.getAttribute("data-status");
      document.querySelectorAll("[data-status]").forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", active ? "true" : "false");
      });
      renderRunList();
    });
  });

  document.querySelectorAll("[data-step-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.stepFilter = button.getAttribute("data-step-filter");
      document.querySelectorAll("[data-step-filter]").forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", active ? "true" : "false");
      });
      renderSteps();
    });
  });

  function stepFromEvent(event) {
    const node = event.target.closest(".step[data-step-index]");
    if (!node || !els.stepList.contains(node)) return null;
    return Number(node.dataset.stepIndex);
  }

  els.stepList.addEventListener("mouseover", (event) => {
    const index = stepFromEvent(event);
    if (index == null || index === state.inspectedIndex) return;
    inspectStep(index);
  });

  els.stepList.addEventListener("click", (event) => {
    const index = stepFromEvent(event);
    if (index == null) return;
    inspectStep(index);
  });

  els.stepList.addEventListener("focusin", (event) => {
    const index = stepFromEvent(event);
    if (index == null) return;
    inspectStep(index);
  });

  els.stepList.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp" && event.key !== "Enter" && event.key !== " ") {
      return;
    }
    const steps = visibleSteps();
    if (!steps.length) return;
    const current = steps.findIndex((step) => step.index === state.inspectedIndex);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      event.stopPropagation();
      const delta = event.key === "ArrowDown" ? 1 : -1;
      const next = steps[Math.min(steps.length - 1, Math.max(0, (current < 0 ? 0 : current) + delta))];
      inspectStep(next.index);
      const node = els.stepList.querySelector(`[data-step-index="${next.index}"]`);
      if (node) node.focus();
      return;
    }
    event.preventDefault();
    const index = stepFromEvent(event);
    if (index != null) inspectStep(index);
  });

  els.form.addEventListener("change", () => {
    updateBranching();
    scheduleAutosave(true);
  });
  els.form.addEventListener("input", (event) => {
    if (!(event.target instanceof HTMLElement)) return;
    if (
      event.target.id === "comments" ||
      event.target.tagName === "TEXTAREA" ||
      event.target.classList.contains("reason-count")
    ) {
      scheduleAutosave(false);
    }
  });
  els.prev.addEventListener("click", () => goRelative(-1));
  els.next.addEventListener("click", () => goRelative(1));
  if (els.overviewBtn) {
    els.overviewBtn.addEventListener("click", () => setOverview(!state.overview, true));
  }
  if (els.overviewClose) {
    els.overviewClose.addEventListener("click", () => setOverview(false, true));
  }
  if (els.overviewBody) {
    els.overviewBody.addEventListener("click", (event) => {
      const button = event.target.closest("[data-jump-task]");
      if (!button) return;
      const task = Number(button.getAttribute("data-jump-task"));
      if (!Number.isFinite(task)) return;
      state.taskNumber = task;
      setOverview(false, false);
      renderProgress();
      renderRunList();
      const first = filteredRuns()[0];
      if (first) loadRun(first.run_id, true);
      else writeUrl(state.currentId, true);
    });
  }

  els.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const saved = await saveAnnotation();
    if (saved && state.saveAndNext) {
      state.saveAndNext = false;
      goRelative(1);
    }
  });

  els.saveNextBtn.addEventListener("click", async () => {
    state.saveAndNext = true;
    const saved = await saveAnnotation();
    if (saved) goRelative(1);
    state.saveAndNext = false;
  });

  function focusComments() {
    if (!els.comments) return;
    els.form.hidden = false;
    els.comments.focus();
    els.comments.scrollIntoView({ block: "nearest" });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isTypingTarget(event.target)) {
      event.target.blur();
      return;
    }
    if (event.key === "Escape" && state.overview) {
      event.preventDefault();
      setOverview(false, true);
      return;
    }
    if (isTypingTarget(event.target)) return;
    if (state.overview) return;
    if (
      event.target instanceof HTMLElement &&
      event.target.closest(".step-list") &&
      (event.key === "ArrowDown" || event.key === "ArrowUp")
    ) {
      return;
    }
    if (event.key === "n" || event.key === "N" || event.key === "j" || event.key === "ArrowDown") {
      event.preventDefault();
      goRelative(1);
    } else if (event.key === "p" || event.key === "P" || event.key === "k" || event.key === "ArrowUp") {
      event.preventDefault();
      goRelative(-1);
    } else if (event.key === "c" || event.key === "C") {
      event.preventDefault();
      focusComments();
    }
  });

  window.addEventListener("pagehide", () => {
    window.clearTimeout(state.saveTimer);
    if (!state.dirty || !state.detail || els.form.hidden) return;
    const payload = collectPayload();
    fetch("/api/annotation", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    });
  });

  window.addEventListener("popstate", () => {
    const parsed = readUrlState();
    state.batch = parsed.batch;
    state.taskNumber = parsed.taskNumber;
    els.batchSelect.value = state.batch;
    renderProgress();
    renderRunList();
    if (parsed.run && filteredRuns().some((run) => run.run_id === parsed.run)) {
      loadRun(parsed.run, false);
    } else {
      ensureCurrentRun();
    }
    setOverview(parsed.overview, false);
  });

  async function init() {
    const [meta, runsPayload] = await Promise.all([
      fetchJson("/api/meta"),
      fetchJson("/api/runs"),
    ]);
    state.meta = meta;
    state.runs = runsPayload.runs || [];
    state.meta.batches = meta.batches || runsPayload.batches || [];
    const parsed = readUrlState();
    const batchIds = new Set((state.meta.batches || []).map((batch) => batch.id));
    if (parsed.batch && batchIds.has(parsed.batch)) {
      state.batch = parsed.batch;
    } else if (!parsed.batch && batchIds.has("luna_high_states_images")) {
      state.batch = "luna_high_states_images";
    } else {
      state.batch = parsed.batch || "";
    }
    state.taskNumber = parsed.taskNumber;
    if (state.taskNumber != null && !taskNumbers().includes(state.taskNumber)) {
      state.taskNumber = null;
    }
    renderBatchSelect();
    els.batchSelect.value = state.batch;
    renderProgress();
    renderRunList();
    const requested = parsed.run;
    const first = requested && filteredRuns().some((run) => run.run_id === requested)
      ? requested
      : filteredRuns()[0]?.run_id;
    if (first) {
      await loadRun(first, !requested);
    }
    setOverview(parsed.overview, false);
  }

  init().catch((error) => {
    els.progress.textContent = "Could not load runs.";
    setSaveStatus(String(error), "err");
  });
})();
