"""Per-task (and optionally per-variant) success / completion / timeouts across row files, plus end-of-plan
and replan-trigger stats when the rows carry them (from batch_rows.py).

Usage (any python3):
    python3 .claude/skills/rail-batch-analysis/compare_tasks.py \
        rail=results/rail_planner_5reps_20261028/baseline_results.json \
        eop=results/rail_end_of_plan_feedback_5reps_20261029/baseline_results.json \
        trig=results/rail_replan_trigger_oracle_5reps_20261029/baseline_results.json [--task 5 --variants]
"""
import argparse
import collections
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="name=path/to/baseline_results.json")
    p.add_argument("--task", type=int, help="only this task")
    p.add_argument("--variants", action="store_true", help="break down by task_id instead of task")
    a = p.parse_args()
    sets = {}
    for spec in a.files:
        name, path = spec.split("=", 1)
        rows = json.load(open(path))
        sets[name] = [r for r in rows if a.task is None or int(r["task"]) == a.task]
    key = (lambda r: r["task_id"]) if a.variants else (lambda r: f"T{r['task']}")
    groups = sorted({key(r) for rs in sets.values() for r in rs}, key=lambda g: (len(g), g))
    print(f"{'group':14}" + "".join(f"{n:>26}" for n in sets))
    for g in groups + ["all"]:
        cells = []
        for rs in sets.values():
            sel = [r for r in rs if g == "all" or key(r) == g]
            if not sel:
                cells.append(f"{'-':>26}")
                continue
            ok = sum(r["task_state_success"] == 1 for r in sel)
            comp = sum(r.get("completion") or 0 for r in sel) / len(sel)
            to = sum(r.get("execution_status") == "timeout" for r in sel)
            cells.append(f"{ok:>4}/{len(sel):<4} c={comp:.2f} to={to:<3}")
        print(f"{g:14}" + "".join(cells))
    for name, rs in sets.items():
        if rs and "sent_back" in rs[0] and any(r["sent_back"] for r in rs):
            print(f"{name}: sent back {sum(r['sent_back'] for r in rs)}, recovered {sum(r['recovered'] for r in rs)}")
        if rs and "trigger_checks" in rs[0] and any(r["trigger_checks"] for r in rs):
            rules = collections.Counter(x.split("_")[0] for r in rs for x in r["trigger_rules"])
            fired = [r for r in rs if r["trigger_fired"]]
            print(f"{name}: {sum(r['trigger_fired'] for r in rs)} firings in {sum(r['trigger_checks'] for r in rs)} checks, "
                  f"{len(fired)} runs fired ({sum(r['task_state_success'] == 1 for r in fired)} succeeded), rules {dict(rules)}")


if __name__ == "__main__":
    main()
