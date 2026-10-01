#!/usr/bin/env bash
# Replay every saved ground-truth sequence on Skynet A40s to measure timing.
#
# Run on sky1 from the repository root (it only checks files, builds the queue and submits;
# all Habitat work runs inside Slurm):
#
#   PARTITION=<your-lab> scripts/skynet/submit_gt_timing.sh            # all 90 variants, 8 GPUs
#   PARTITION=<your-lab> scripts/skynet/submit_gt_timing.sh --smoke    # shortest variant per task, 1 GPU, debug
#   PARTITION=<your-lab> REPEATS=3 scripts/skynet/submit_gt_timing.sh --variants T6-OUT-CON T6-ACC-BASE
#
# Settings (environment variables):
#   PARTITION       Slurm partition (Skynet: your lab's partition, or debug/short/long)  [required]
#   QOS             Slurm QoS                                     default: short (debug for --smoke)
#   ACCOUNT         Slurm account (e.g. overcap)                  default: none
#   SHARDS          array tasks = GPUs used at once               default: 8
#   REPEATS         replays per variant (median is reported)      default: 1
#   TIME_LIMIT      per array task                                default: 12:00:00 (01:00:00 for --smoke)
#   HABITAT_PYTHON  python of the habitat conda env               default: ~/miniconda3/envs/habitat/bin/python
#   TAG             batch folder name                             default: skynet_<timestamp>
#
# Results: baseline_evaluation_v3/ground_truth_timing/<TAG>/ (queue.json, results/, summary_*.csv, gpu/, logs/).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

SMOKE=""
VARIANTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE=1; shift ;;
        --variants) shift; while [[ $# -gt 0 && "$1" != --* ]]; do VARIANTS+=("$1"); shift; done ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

: "${PARTITION:?Set PARTITION to your Skynet partition (e.g. PARTITION=<lab-name>)}"
QOS="${QOS:-$([[ -n "$SMOKE" ]] && echo debug || echo short)}"
ACCOUNT="${ACCOUNT:-}"
SHARDS="${SHARDS:-$([[ -n "$SMOKE" ]] && echo 1 || echo 8)}"
REPEATS="${REPEATS:-1}"
TIME_LIMIT="${TIME_LIMIT:-$([[ -n "$SMOKE" ]] && echo 01:00:00 || echo 12:00:00)}"
HABITAT_PYTHON="${HABITAT_PYTHON:-$HOME/miniconda3/envs/habitat/bin/python}"
TAG="${TAG:-skynet_$(date +%Y%m%d_%H%M%S)$([[ -n "$SMOKE" ]] && echo _smoke)}"
OUT="$REPO/baseline_evaluation_v3/ground_truth_timing/$TAG"

# ---- preflight: files only, nothing heavy on the login node
fail() { echo "PREFLIGHT: $*" >&2; exit 1; }
[[ -x "$HABITAT_PYTHON" ]] || fail "no habitat python at $HABITAT_PYTHON (set HABITAT_PYTHON)"
[[ -f baseline_evaluation_v3/ground_truth.sqlite3 ]] || fail "missing baseline_evaluation_v3/ground_truth.sqlite3"
for path in data/hssd-hab data/robots data/humanoids data/objects data/versioned_data; do
    [[ -e "$path" ]] || fail "missing $path (rsync the data/ folder; see scripts/skynet/README.md)"
done
broken="$(find data -maxdepth 2 -xtype l 2>/dev/null || true)"
[[ -z "$broken" ]] || fail "broken symlinks under data/ (they point at the local machine):"$'\n'"$broken"
[[ -e "$OUT" ]] && fail "$OUT already exists; choose another TAG"

queue_args=(--out "$OUT" --shards "$SHARDS" --repeats "$REPEATS")
[[ -n "$SMOKE" ]] && queue_args+=(--smoke)
[[ ${#VARIANTS[@]} -gt 0 ]] && queue_args+=(--variants "${VARIANTS[@]}")
shards="$("$HABITAT_PYTHON" scripts/skynet/gt_timing_replay.py queue "${queue_args[@]}")"
mkdir -p "$OUT/logs"

slurm_args=(--partition="$PARTITION" --qos="$QOS" --time="$TIME_LIMIT")
[[ -n "$ACCOUNT" ]] && slurm_args+=(--account="$ACCOUNT")
export REPO OUT HABITAT_PYTHON

array_job="$(sbatch --parsable "${slurm_args[@]}" --array="0-$((shards - 1))" \
    --output="$OUT/logs/shard_%a.out" --export=ALL scripts/skynet/gt_timing_shard.sbatch)"
summary_job="$(sbatch --parsable --partition="$PARTITION" --qos="$QOS" ${ACCOUNT:+--account="$ACCOUNT"} \
    --time=00:10:00 --cpus-per-task=1 --mem=2G --job-name=gt-timing-summary \
    --dependency="afterany:$array_job" --output="$OUT/logs/summary.out" --export=ALL \
    --wrap="cd '$REPO' && '$HABITAT_PYTHON' scripts/skynet/gt_timing_replay.py summarize --out '$OUT'")"

cat <<EOF
Submitted $shards shard(s) as array job $array_job; summary job $summary_job runs after it.
Batch folder: $OUT

Progress (check occasionally, not in a loop):
  squeue -u \$USER
  tail -n 3 $OUT/logs/shard_*.out
Results when finished:
  cat $OUT/logs/summary.out
Then copy the folder back and publish to the Metrics page (see scripts/skynet/README.md).
EOF
