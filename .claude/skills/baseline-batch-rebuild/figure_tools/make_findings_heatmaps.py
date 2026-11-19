"""Failure and inefficiency heat grids (paper figures 09/10) for the three baselines in this batch
(VLM-TAMP task 4 = the basket rerun). Same layout as
../../rail_end_of_plan_feedback_5reps_20261029/paper_figure/make_findings_heatmaps.py.

Both grids are % of all runs per baseline (n=450). A failure finding only counts on an unsuccessful run;
findings can overlap. Rows: <batch>/baseline_results.json. Findings: <batch>/uncertainty_classifier/source/findings.csv
(current rules, including the T3-OUT PowerOff exemption).

Writes 09_failure_findings_heatmap.{png,pdf,svg}, 10_inefficiency_findings_heatmap.{png,pdf,svg} and
findings_heatmaps.csv here.

Usage: run via ../rebuild_batch.sh (BATCH_DIR / FIG_DIR from the environment).
"""
import collections
import csv
import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(os.environ['FIG_DIR'])  # output folder
BATCH = Path(os.environ['BATCH_DIR'])

COLUMNS = [  # (id prefix, header)
    ("react", "ReAct"),
    ("vlm-tamp", "VLM-TAMP"),
    ("tru-pomdp", "TRU-POMDP"),
]
FAILURES = ["object_not_placed", "missing_action", "placement_error", "object_not_found",
            "insufficient_planning", "no_hypotheses", "wrong_object", "wrong_order", "failure_to_stop"]
INEFFICIENCIES = ["unnecessary_actions", "precondition_error", "repeated_actions", "failure_to_stop",
                  "hallucinated_object"]
CMAP = LinearSegmentedColormap.from_list("cells", [(0, "#f0f7fd"), (0.55, "#4aaef0"), (1, "#1569a8")])
INK, MUTED, EDGE = "#111827", "#64748b", "#d9e2ec"


def load():
    rows = json.loads((BATCH / "baseline_results.json").read_text())
    findings = list(csv.DictReader((BATCH / "uncertainty_classifier/source/findings.csv").open()))
    planner = lambda run_id: run_id.split("/", 1)[0]  # noqa: E731
    keys = {k for k, _ in COLUMNS}
    rows = [r for r in rows if planner(r["id"]) in keys]
    runs = {k: {r["id"] for r in rows if planner(r["id"]) == k} for k in keys}
    failed = {k: {r["id"] for r in rows if planner(r["id"]) == k and r.get("task_state_success") != 1} for k in keys}
    hits = collections.defaultdict(set)
    for f in findings:
        if planner(f["run_id"]) in keys:
            hits[(f["category"], f["label"])].add(f["run_id"])
    return runs, failed, hits


def grid(category, labels, counted, total, hits):
    """% of ``total`` runs that are in ``counted`` and carry the label."""
    return [[100 * len(hits[(category, lab)] & counted[k]) / len(total[k]) for k, _ in COLUMNS] for lab in labels]


def title_case(label):
    return "\n".join(textwrap.wrap(label.replace("_", " ").title(), 13))


def plot(stem, labels, values, subheads, legend, caption):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    ncol, nrow = len(COLUMNS), len(labels)
    label_w, cell_w, cell_h, gap, top_h, right = 3.3, 2.15, 0.72, 0.05, 0.6, 0.2
    head_h = 1.25 + 0.45 * max(s.count("\n") for s in subheads.values())
    W = label_w + ncol * cell_w + right
    H = top_h + head_h + nrow * cell_h
    fig = plt.figure(figsize=(W * 0.62, H * 0.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Legend: "% of ... runs  0% [ramp] 100%"
    ramp_x, ramp_w, ly = label_w + 2.0 * cell_w, 1.1, top_h * 0.55
    ax.text(ramp_x - 0.75, ly, legend, ha="right", va="center", fontsize=13, color=MUTED)
    ax.text(ramp_x - 0.12, ly, "0%", ha="right", va="center", fontsize=13, color=INK)
    ax.imshow([[i / 99 for i in range(100)]], cmap=CMAP, vmin=0, vmax=1, aspect="auto",
              extent=(ramp_x, ramp_x + ramp_w, ly + 0.09, ly - 0.09), zorder=2)
    ax.text(ramp_x + ramp_w + 0.1, ly, "100%", ha="left", va="center", fontsize=13, color=INK)

    y0 = top_h
    ax.text(0.15, y0 + head_h * 0.5, "Finding", ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    for j, (key, header) in enumerate(COLUMNS):
        cx = label_w + (j + 0.5) * cell_w
        ax.text(cx, y0 + 0.43, header, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color=INK, linespacing=1.0)
        ax.text(cx, y0 + 1.0, subheads[key], ha="center", va="top", fontsize=12, color=INK, linespacing=1.25)

    for i, (label, row) in enumerate(zip(labels, values)):
        y = y0 + head_h + i * cell_h
        ax.text(0.15, y + cell_h / 2, title_case(label), ha="left", va="center", fontsize=13.5,
                fontweight="bold", color=INK, linespacing=1.15)
        for j, v in enumerate(row):
            x = label_w + j * cell_w
            ax.add_patch(FancyBboxPatch((x + gap, y + gap), cell_w - 2 * gap, cell_h - 2 * gap,
                                        boxstyle="round,pad=0,rounding_size=0.04", facecolor=CMAP(v / 100),
                                        edgecolor=EDGE, linewidth=0.8))
            ax.text(x + cell_w / 2, y + cell_h / 2, f"{v:.1f}%", ha="center", va="center", fontsize=14.5,
                    fontweight="bold", color=INK)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(HERE / f"{stem}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    (HERE / f"{stem}.caption.txt").write_text(caption + "\n")


def main():
    runs, failed, hits = load()
    fail = grid("failure", FAILURES, failed, runs, hits)
    ineff = grid("inefficiency", INEFFICIENCIES, runs, runs, hits)
    heads = {k: f"n={len(runs[k])}\n{len(failed[k])} failed" for k in runs}
    plot("09_failure_findings_heatmap", FAILURES, fail, heads, "% of all runs",
         "Percentage of all evaluated episodes (n per baseline) that failed and exhibit each rule-based failure "
         "finding; the number of failed episodes is shown under n. Findings may overlap, so a column does not sum "
         "to its failure rate.")
    plot("10_inefficiency_findings_heatmap", INEFFICIENCIES, ineff, heads, "% of all runs",
         "Percentage of all evaluated episodes (n per baseline), successful or not, exhibiting each rule-based "
         "inefficiency finding; the number of failed episodes is shown under n. Findings may overlap.")
    with (HERE / "findings_heatmaps.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "label"] + [h.replace("\n", " ") for _, h in COLUMNS])
        for cat, labels, vals in (("failure", FAILURES, fail), ("inefficiency", INEFFICIENCIES, ineff)):
            for lab, row in zip(labels, vals):
                w.writerow([cat, lab] + [f"{v:.1f}" for v in row])
                print(f"{cat:12} {lab:22}", " ".join(f"{v:6.1f}" for v in row))
    print("n failed:", {k: len(v) for k, v in failed.items()})


if __name__ == "__main__":
    main()
