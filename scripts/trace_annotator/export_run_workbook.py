"""Export saved simulator-run metrics; no model requests or simulation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ACTION = r"([A-Za-z][A-Za-z0-9_]*)\[([^\n]*)\]"
SHEETS = {
    "vlm_tamp_no_images": "VLM-TAMP no images",
    "vlm_tamp_images": "VLM-TAMP images",
    "react_no_images": "PARTNR ReAct no images",
    "react_images": "PARTNR ReAct images",
    "tru_pomdp_no_images": "TruPOMDP no images",
}
PREFERRED_ACTIONS = [
    "Explore",
    "Navigate",
    "Open",
    "Close",
    "Pick",
    "Place",
    "Fill",
    "Pour",
    "Clean",
    "PowerOn",
    "PowerOff",
    "Wait",
]


def trace_actions(text, condition):
    text = ANSI.sub("", text)
    if condition.startswith("react_"):
        return re.findall(r"^" + ACTION + r"\s*$", text, re.M)
    return re.findall(
        r"^(?:Action: |High-level special-case: )" + ACTION + r"\s*$", text, re.M
    )


def console_actions(text, condition):
    """Return selected actions where logged, conservative lower bounds otherwise."""
    if condition.startswith("vlm_tamp"):
        actions = re.findall(
            r"^\s*(?:▶ \[\d+/\d+\]\s+|Action: )" + ACTION + r"\s*$", text, re.M
        )
        return actions, "Console selections; partial log"
    if condition.startswith("react_"):
        actions = []
        for block in re.findall(r"🤖 LLM RESPONSE\s*\n=+\s*\n(.*?)\n=+", text, re.S):
            choices = re.findall(r"^" + ACTION + r"\s*$", block, re.M)
            if choices:
                # The run's parser takes the first tool call in the response.
                actions.append(choices[0])
        return actions, "Console selections; partial log"
    # TruPOMDP does not print every skill start in older runs. World-graph
    # updates log successful completed high-level skills, not their inner moves.
    actions = re.findall(
        r"\[DEBUG\] - agent_uid=0: high_level_action=\('([^']+)', '([^']*)', None\), action_response=",
        text,
    )
    return actions, "Lower bound: successful skills logged only"


def summary_step_count(console):
    match = re.search(
        r"Action Simulation Steps \(total steps per action\) for instruction:\s*\n(.*?)(?=\n-+|\Z)",
        console,
        re.S,
    )
    if match:
        values = re.findall(r"^\s+\w+: (\d+)\s*$", match[1], re.M)
        if values:
            return sum(map(int, values))
    return None


def extract_metrics(run):
    root = Path(run["path"])
    console_path = root / "console.log"
    console = (
        ANSI.sub("", console_path.read_text(errors="replace"))
        if console_path.exists()
        else ""
    )
    traces = sorted(root.glob("dataset/traces/0/*.txt"))
    actions = []
    if traces:
        for p in traces:
            actions.extend(
                trace_actions(p.read_text(errors="replace"), run["condition"])
            )
        coverage = "Saved final trace"
        action_source = "; ".join(str(p) for p in traces)
    else:
        actions, coverage = console_actions(console, run["condition"])
        action_source = str(console_path)
        if not actions:
            coverage = "Unavailable: no logged action selections"
    counts = Counter(name for name, _ in actions)
    sim_steps = None
    step_source = "Unavailable: no saved simulator step counter"
    files = sorted(root.glob("dataset/planner-log/*.json"))
    planner_selection_counts = Counter()
    for p in files:
        data = json.loads(p.read_text())
        values = [
            s["sim_step_count"]
            for s in data.get("steps", [])
            if isinstance(s.get("sim_step_count"), (int, float))
        ]
        if values:
            sim_steps = (
                max(values) if sim_steps is None else max(sim_steps, max(values))
            )
            step_source = str(p)
        if run["condition"].startswith("react_"):
            for s in data.get("steps", []):
                for agent, action in s.get("high_level_actions", {}).items():
                    if s.get("replanned", {}).get(agent) and action and action[0]:
                        planner_selection_counts[action[0]] += 1
    console_steps = summary_step_count(console)
    checks = []
    if (
        sim_steps is not None
        and console_steps is not None
        and sim_steps != console_steps
    ):
        checks.append(
            {
                "kind": "sim_steps_disagreement",
                "planner": sim_steps,
                "console": console_steps,
            }
        )
    if sim_steps is None and console_steps is not None:
        sim_steps, step_source = (
            console_steps,
            str(console_path) + "#Action Simulation Steps",
        )
    if (
        traces
        and run["condition"].startswith("react_")
        and files
        and counts != planner_selection_counts
    ):
        checks.append(
            {
                "kind": "react_trace_vs_planner",
                "trace": dict(counts),
                "planner": dict(planner_selection_counts),
            }
        )
        # Structured selections exclude unexecuted extra calls in prose.
        counts = planner_selection_counts
        coverage = "Saved planner selections"
        action_source = str(files[0]) + "#replanned/high_level_actions"
    # User-requested nominal expansion, including attempts that terminated early.
    # This is a comparison convention, not a claim that every stage executed.
    rearranges = counts.pop("Rearrange", 0)
    if rearranges:
        counts.update(
            {"Navigate": 2 * rearranges, "Pick": rearranges, "Place": rearranges}
        )
    simulator = run["simulator"]
    return {
        "id": run["id"],
        "condition": run["condition"],
        "task": run["task"],
        "task_id": run["episode_id"],
        "memory": run["memory"],
        "uncertainty": run["uncertainty"],
        "iteration": run["repetition"],
        "run_id": run["run_id"],
        "execution_status": run["status"],
        "simulator_outcome": simulator["outcome"],
        "sim_steps": int(sim_steps) if sim_steps is not None else None,
        "num_actions": (
            sum(n for name, n in counts.items() if name != "Done")
            if actions or traces or planner_selection_counts
            else None
        ),
        "counts": dict(counts),
        "runtime_s": run["runtime_seconds"],
        "num_subtasks": simulator["total_count"],
        "subtasks_succeeded": simulator["satisfied_count"],
        "completion": simulator["metrics"].get("task_percent_complete"),
        "action_coverage": coverage,
        "sim_steps_coverage": (
            "Saved final counter" if sim_steps is not None else "Unavailable"
        ),
        "action_source": action_source,
        "sim_steps_source": step_source,
        "simulator_source": simulator["source"]["file"],
        "run_folder": str(root),
        "checks": checks,
    }


def write_workbook(rows, output, actions):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    fields = [
        ("sim_steps", "Sim steps"),
        ("num_actions", "Num actions"),
        *[("action:" + a, a) for a in actions],
        ("runtime_s", "Runtime (s)"),
        ("num_subtasks", "Num subtasks"),
        ("subtasks_succeeded", "Subtasks succeeded"),
        ("completion", "Completion %"),
        ("execution_status", "Execution status"),
        ("simulator_outcome", "Simulator outcome"),
        ("action_coverage", "Action count coverage"),
        ("sim_steps_coverage", "Sim step coverage"),
        ("run_id", "Run ID"),
        ("action_source", "Action source"),
        ("sim_steps_source", "Sim step source"),
        ("simulator_source", "Score source"),
        ("run_folder", "Run folder"),
    ]
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = "PARTNR baseline run comparison — all 1,350 runs"
    wb.properties.description = "Simulator scores and recorded high-level actions, grouped by task variant and iteration."
    colors = ["1D4ED8", "0F766E", "7C3AED"]
    for condition, title in SHEETS.items():
        ws = wb.create_sheet(title)
        subset = [r for r in rows if r["condition"] == condition]
        grouped = {}
        for r in subset:
            key = (r["task"], r["task_id"], r["memory"], r["uncertainty"])
            assert r["iteration"] not in grouped.setdefault(key, {})
            grouped[key][r["iteration"]] = r
        assert len(grouped) == 90 and all(set(g) == {1, 2, 3} for g in grouped.values())
        last_col = 4 + 3 * len(fields)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=16)
        ws.cell(1, 1, title + " | 90 variants × 3 iterations = 270 runs")
        ws.cell(1, 1).font = Font(size=17, bold=True, color="17365D")
        notes = [
            "Expanded action counts: each Rearrange selection adds 2 Navigate + 1 Pick + 1 Place, even if it failed before all stages executed. Done is excluded from Num actions.",
            "Explore counts once per room-sweep call; its internal navigation is not counted as Navigate. Num actions sums the expanded action columns.",
            "Blank = unavailable, not zero. Amber = partial/lower-bound action logs. Old TruPOMDP timeouts retain only successful skill counts.",
            "Runtime = measured process wall time including setup, planning, execution, and shutdown. Wall-clock limit: 10 min for Tasks 1–5, 30 min for Tasks 6 and 7. Task 7 uses relaxed containment (toys in one shared wardrobe, plate in any cabinet), rescored offline. Subtasks = simulator-scored propositions; unscored prose is excluded.",
        ]
        for idx, note in enumerate(notes, 2):
            ws.merge_cells(start_row=idx, start_column=1, end_row=idx, end_column=16)
            ws.cell(idx, 1, note).font = Font(size=10, color="475569")
        fixed = ["Task", "Task ID / variant", "Memory", "Uncertainty"]
        for col, value in enumerate(fixed, 1):
            ws.cell(7, col, value)
            ws.column_dimensions[get_column_letter(col)].width = [9, 22, 12, 14][
                col - 1
            ]
        for iteration, color in enumerate(colors, 1):
            start = 5 + (iteration - 1) * len(fields)
            ws.merge_cells(
                start_row=6,
                start_column=start,
                end_row=6,
                end_column=start + len(fields) - 1,
            )
            cell = ws.cell(6, start, f"iter_{iteration}")
            cell.fill = PatternFill("solid", fgColor=color)
            cell.font = Font(bold=True, color="FFFFFF", size=13)
            cell.alignment = Alignment(horizontal="left")
            for offset, (key, label) in enumerate(fields):
                col = start + offset
                c = ws.cell(7, col, label)
                c.fill = PatternFill("solid", fgColor=color)
                c.font = Font(bold=True, color="FFFFFF")
                c.alignment = Alignment(wrap_text=True, vertical="center")
                ws.column_dimensions[get_column_letter(col)].width = 13
                if key in ("action_coverage", "sim_steps_coverage"):
                    ws.column_dimensions[get_column_letter(col)].width = 26
                if key in (
                    "run_id",
                    "action_source",
                    "sim_steps_source",
                    "simulator_source",
                    "run_folder",
                ):
                    ws.column_dimensions[get_column_letter(col)].width = 45
                    ws.column_dimensions[get_column_letter(col)].hidden = True
                c.comment = Comment(
                    f"{label} for iter_{iteration}. See the notes above; source columns on the right of each iteration are hidden and can be expanded.",
                    "Run export",
                )
        for index, (key, iterations) in enumerate(sorted(grouped.items()), 8):
            for col, value in enumerate(key, 1):
                ws.cell(index, col, value)
            for iteration in (1, 2, 3):
                r = iterations[iteration]
                for offset, (field, _) in enumerate(fields):
                    col = 5 + (iteration - 1) * len(fields) + offset
                    value = (
                        r["counts"].get(field[7:], 0)
                        if field.startswith("action:") and r["num_actions"] is not None
                        else r.get(field)
                    )
                    c = ws.cell(index, col, value)
                    if isinstance(value, str):
                        c.data_type = "s"  # Never interpret log text as a formula.
                    c.fill = PatternFill(
                        "solid", fgColor="F1F5F9" if index % 2 else "FFFFFF"
                    )
                    c.alignment = Alignment(
                        vertical="center",
                        wrap_text=field in ("action_coverage", "sim_steps_coverage"),
                    )
                    if field in ("runtime_s",):
                        c.number_format = "0.00"
                    elif field == "completion":
                        c.number_format = "0.0%"
                    if field == "action_coverage" and not r[field].startswith("Saved"):
                        c.fill = PatternFill("solid", fgColor="FEF3C7")
                    if field in ("subtasks_succeeded", "sim_steps") and value is None:
                        c.fill = PatternFill("solid", fgColor="E2E8F0")
                        c.comment = Comment(
                            "Not saved in available terminal logs. This is unknown, not zero.",
                            "Run export",
                        )
                    if offset == 0:
                        c.border = Border(
                            left=Side(style="medium", color=colors[iteration - 1])
                        )
            ws.row_dimensions[index].height = 32
        for col in range(1, 5):
            ws.cell(7, col).fill = PatternFill("solid", fgColor="334155")
            ws.cell(7, col).font = Font(bold=True, color="FFFFFF")
        for iteration in (1, 2, 3):
            col = (
                5
                + (iteration - 1) * len(fields)
                + [k for k, _ in fields].index("completion")
            )
            ws.conditional_formatting.add(
                f"{get_column_letter(col)}8:{get_column_letter(col)}{ws.max_row}",
                ColorScaleRule(
                    start_type="num",
                    start_value=0,
                    start_color="FECACA",
                    end_type="num",
                    end_value=1,
                    end_color="BBF7D0",
                ),
            )
        ws.freeze_panes = "E8"
        ws.auto_filter.ref = f"A7:{get_column_letter(last_col)}{ws.max_row}"
        ws.row_dimensions[7].height = 42
        ws.sheet_view.showGridLines = False
        ws.sheet_view.zoomScale = 80
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.print_title_rows = "1:7"
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads((args.study / "study.json").read_text())["runs"]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(extract_metrics, runs):
            rows.append(row)
            if len(rows) % 150 == 0:
                print(f"Extracted {len(rows)}/{len(runs)} runs", flush=True)
    seen = {name for r in rows for name in r["counts"]}
    actions = (
        PREFERRED_ACTIONS + sorted(seen - set(PREFERRED_ACTIONS) - {"Done"}) + ["Done"]
    )
    write_workbook(rows, args.output, actions)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    keys = [k for k in rows[0] if k not in ("counts", "checks")] + [
        "count_" + a for a in actions
    ]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            record = {k: r.get(k) for k in keys}
            record.update(
                {
                    "count_"
                    + a: r["counts"].get(a, 0) if r["num_actions"] is not None else None
                    for a in actions
                }
            )
            writer.writerow(record)
    audit = {
        "counting_convention": "Nominal expansion: each recorded Rearrange selection becomes 2 Navigate + 1 Pick + 1 Place, regardless of early failure; Done excluded from total.",
        "runs": len(rows),
        "conditions": dict(Counter(r["condition"] for r in rows)),
        "action_coverage": dict(Counter(r["action_coverage"] for r in rows)),
        "missing_sim_steps": sum(r["sim_steps"] is None for r in rows),
        "missing_subtask_scores": sum(r["subtasks_succeeded"] is None for r in rows),
        "action_types": actions,
        "crosscheck_differences": [
            {"id": r["id"], "checks": r["checks"]} for r in rows if r["checks"]
        ],
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "crosscheck_differences"}, indent=2
        )
    )
    print("Crosscheck differences:", len(audit["crosscheck_differences"]))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
