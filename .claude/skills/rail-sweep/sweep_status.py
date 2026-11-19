"""Progress of a rail sweep made by launch_rail_sweep.py (or any batch with shards/*/attempt_*/runs/*/run.json
and submissions.txt): Slurm state per shard, then runs done / expected, successes and timeouts per task.

Usage: python3 .claude/skills/rail-sweep/sweep_status.py results/<batch> [--reps 5]
"""
import argparse
import collections
import glob
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def slurm_states(batch):
    subs = [line.split() for line in (batch / "submissions.txt").read_text().splitlines() if line.strip()]
    ids = {s[1]: s[0] for s in subs if s[1].isdigit()}
    if not ids:
        return {}
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(ids), "-o", "JobID,State,Elapsed"],
                         capture_output=True, text=True).stdout
    states = {}
    for line in out.splitlines():
        job, state, elapsed = line.split("|")
        if job in ids:
            states[ids[job]] = f"{state} {elapsed}"
    return states


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    batch = a.batch.resolve()
    states = slurm_states(batch)
    print("shards:", dict(collections.Counter(s.split()[0] for s in states.values())))
    for shard, s in sorted(states.items()):
        if not s.startswith("COMPLETED"):
            print(f"  {shard:8} {s}")
    runs = {}
    for f in glob.glob(str(batch / "shards/*/attempt_*/runs/*/run.json")):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        if r.get("status") in ("completed", "timeout"):
            runs[r["run_id"]] = r
    by = collections.defaultdict(lambda: [0, 0, 0])
    for rid, r in runs.items():
        t = rid.split("-")[0]
        by[t][0] += 1
        by[t][1] += ((r.get("metrics") or {}).get("task_state_success") or 0) >= 1
        by[t][2] += r["status"] == "timeout"
    total = 0
    for t in sorted({s.split("-")[0] for s in states} | set(by)):
        n = sum(1 for m in ("acc", "inc", "out") for _ in
                glob.glob(str(REPO / f"baseline_evaluation_v3/episodes/task_{t[1:]}/{t}-{m}-*"))
                if f"{t}-{m}" in states) * a.reps
        total += n
        d, s, to = by[t]
        print(f"{t}: {d}/{n} done, {s} success, {to} timeout")
    print(f"total {len(runs)}/{total}")
    print("Slurm FAILED with every run done = run_parallel exits 1 when any run timed out; not a crash.")


if __name__ == "__main__":
    main()
