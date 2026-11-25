"""Launch a RailPlanner sweep over baseline_evaluation_v3 on Slurm: one shard per task x memory, N reps
of every variant, frozen code snapshot, rail-lab first with overcap overflow, requeue-safe resume.

Usage (from the repo root, any python3):
    python3 .claude/skills/rail-sweep/launch_rail_sweep.py rail_<what>_5reps_<date> \
        --override evaluation.agents.agent_0.planner.plan_config.replan_trigger=oracle [--override ...] \
        [--tasks 1 2 3] [--memories acc inc out] [--reps 5] [--snapshot-from results/<batch>/snapshot] [--dry-run]

--dry-run prints the plan and writes nothing. Without it, this makes PAID API calls (about reps x variants runs).
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PYTHON_BIN = "/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin"
TIMEOUTS = {6: 1200, 7: 1200}  # seconds per run; T1-T5 use 600 (matches the baseline batch)
# One multi-partition request per shard: Slurm starts it on rail-lab (PriorityTier 2, no preemption) whenever
# rail-lab can run it, and only falls back to overcap (tier 1, preempt/requeue) when rail-lab is full.
ACCOUNT, PARTITIONS = "rail-lab", "rail-lab,overcap"

SHARD = r"""#!/bin/bash
# One shard = one task x memory (e.g. t3-inc), {reps} reps of every variant, planner={planner} with:
{override_comment}
# Usage (via sbatch): shard.sh <task_num> <memory> <timeout_s>
# On requeue, each new attempt reruns only runs without a completed/timeout result.
set -euo pipefail
TASK=$1 MEM=$2 TIMEOUT=$3
BATCH={batch}
REPO={repo}
SHARD=$BATCH/shards/t${{TASK}}-${{MEM}}
mkdir -p "$SHARD"
export PATH={python_bin}:$PATH
export LD_LIBRARY_PATH={python_lib}:${{LD_LIBRARY_PATH:-}}
cd "$REPO"
# The snapshot has no .env, so export the repo's key (stops here if it is missing).
OPENAI_API_KEY="$(python -c 'import sys; from dotenv import dotenv_values; k = dotenv_values(".env").get("OPENAI_API_KEY"); sys.exit("OPENAI_API_KEY missing from .env") if not k else print(k)')"
export OPENAI_API_KEY

N=$(find "$SHARD" -maxdepth 1 -name 'attempt_*' -type d | wc -l)
OUT=$SHARD/attempt_$(printf %02d "$N")
RUN_IDS=$(python - "$SHARD" <<'EOF'
import glob, json, os, sys
shard = sys.argv[1]
first = os.path.join(shard, "attempt_00", "manifest.json")
if not os.path.exists(first):
    sys.exit(0)  # first attempt (or it died before writing a manifest): run everything
all_ids = [j["run_id"] for j in json.load(open(first))["jobs"]]
done = set()
for rj in glob.glob(os.path.join(shard, "attempt_*", "runs", "*", "run.json")):
    try:
        r = json.load(open(rj))
    except Exception:
        continue
    if r.get("status") in ("completed", "timeout"):
        done.add(r["run_id"])
todo = [i for i in all_ids if i not in done]
print(" ".join(todo) if todo else "NONE")
EOF
)
if [ "$RUN_IDS" = "NONE" ]; then
    echo "All runs in $SHARD already have results."
    exit 0
fi
EXTRA=()
[ -n "$RUN_IDS" ] && EXTRA=(--run-ids $RUN_IDS)
echo "Attempt $OUT on $(hostname) partition=${{SLURM_JOB_PARTITION:-?}} job=${{SLURM_JOB_ID:-?}}"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python "$BATCH/snapshot/scripts/run_parallel.py" \
    --episodes "$REPO/baseline_evaluation_v3/episodes/task_${{TASK}}/t${{TASK}}-${{MEM}}-*" \
    --planner {planner} --runs-per-episode {reps} --workers 6 --timeout "$TIMEOUT" \
{budget_args}{override_args}    --output-dir "$OUT" "${{EXTRA[@]}}"
"""


def make_snapshot(dest, source=None):
    """Copy the code the runs will use; data and third_party are symlinked (same as run_parallel.workspace)."""
    if source:
        shutil.copytree(source, dest, symlinks=True)
        return
    dest.mkdir()
    for name in ("habitat_llm", "scripts", "dataset_generation"):
        shutil.copytree(REPO / name, dest / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "outputs", "logs"))
    for name in ("data", "third_party"):
        (dest / name).symlink_to(REPO / name, target_is_directory=True)


def checksums(snapshot):
    files = [snapshot / "habitat_llm/conf/planner/rail_planner.yaml"]
    files += sorted((snapshot / "habitat_llm/rail_planner").glob("*.py"))
    return "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(snapshot)}\n" for f in files)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="batch directory name under results/")
    p.add_argument("--override", action="append", default=[], help="Hydra override passed to every run")
    p.add_argument("--tasks", nargs="+", type=int, default=list(range(1, 8)))
    p.add_argument("--memories", nargs="+", default=["acc", "inc", "out"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--snapshot-from", type=Path, help="copy this existing snapshot/ (byte-identical code to another batch)")
    p.add_argument("--planner", default="rail", choices=["rail", "react", "vlm-tamp", "tru-pomdp"])
    p.add_argument("--match-baseline-budget", action="store_true",
                   help="also pass evaluation.max_combined_time_s=<timeout> and exclude_explore_from_time_budget=False"
                        " per run, as the all_baselines_5reps batch did")
    p.add_argument("--root", type=Path, default=REPO / "results", help="parent dir of the batch")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-submit", action="store_true",
                   help="write snapshot, shard.sh and submit.sh but do not sbatch; run submit.sh later")
    a = p.parse_args()

    batch = a.root.resolve() / a.name
    shards = [(t, m) for t in a.tasks for m in a.memories]
    variants = sum(len(list((REPO / f"baseline_evaluation_v3/episodes/task_{t}").glob(f"t{t}-{m}-*"))) for t, m in shards)
    print(f"batch {batch}\n{len(shards)} shards, {variants} variants x {a.reps} reps = {variants * a.reps} runs (paid API calls)")
    print("planner:", a.planner, "| overrides:", a.override or "(none: config defaults)")
    if batch.exists():
        sys.exit(f"{batch} already exists; pick a new name (shards resume by themselves on requeue)")
    if a.dry_run:
        for i, (t, m) in enumerate(shards):
            print(f"  t{t}-{m}  account={ACCOUNT} partitions={PARTITIONS}  timeout={TIMEOUTS.get(t, 600)}")
        return

    (batch / "slurm").mkdir(parents=True)
    (batch / "shards").mkdir()
    make_snapshot(batch / "snapshot", a.snapshot_from and a.snapshot_from.resolve())
    (batch / "snapshot_rail_planner.sha256").write_text(checksums(batch / "snapshot"))
    shard = batch / "shard.sh"
    shard.write_text(SHARD.format(
        reps=a.reps, planner=a.planner, batch=batch,
        budget_args=('    --override evaluation.max_combined_time_s="$TIMEOUT" \\\n'
                     '    --override ++evaluation.exclude_explore_from_time_budget=False \\\n')
        if a.match_baseline_budget else "", repo=REPO, python_bin=PYTHON_BIN, python_lib=str(Path(PYTHON_BIN).parent / "lib"),
        override_comment="\n".join(f"#   {o}" for o in a.override) or "#   (config defaults)",
        override_args="".join(f"    --override {o} \\\n" for o in a.override)))
    shard.chmod(0o755)
    lines = ["#!/bin/bash", "# Submits every shard; appends job ids to submissions.txt (sweep_status.py reads it).",
             "set -euo pipefail", f"cd {batch}"]
    for i, (t, m) in enumerate(shards):
        acc, timeout = ACCOUNT, TIMEOUTS.get(t, 600)
        cmd = ["sbatch", "--parsable", "-A", acc, "-p", PARTITIONS, "--requeue", "--gres=gpu:a40:1", "--cpus-per-task=12",
               "--mem=128G", "--time=4:00:00", f"--job-name={a.name[:20]}-t{t}-{m}",
               f"--output={batch}/slurm/t{t}-{m}-%j.out", str(shard), str(t), m, str(timeout)]
        lines.append(f'job=$({" ".join(cmd)}) && echo "t{t}-{m} $job account={acc} timeout={timeout}" | tee -a submissions.txt')
    submit = batch / "submit.sh"
    submit.write_text("\n".join(lines) + "\n")
    submit.chmod(0o755)
    if a.no_submit:
        print(f"prepared, not submitted. Launch with: bash {submit}")
        return
    subprocess.run(["bash", str(submit)], check=True)

if __name__ == "__main__":
    main()
