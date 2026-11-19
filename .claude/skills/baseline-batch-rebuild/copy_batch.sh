#!/bin/bash
# Make a new baseline batch folder from an existing one without touching it:
# top-level files, uncertainty_classifier/ and logs/ are real copies (scripts rewrite them); run folders and the
# code snapshot are hardlinked (no extra disk; never edit a file in them in place - it would change the original).
# Paper-figure folders go to archive/. Variants listed in MOVE are hardlinked into archive/<planner>_<tag>/ instead.
# Usage (repo root): copy_batch.sh results/<original> results/<new> [<planner>/<variant-glob> ...]
#   e.g. copy_batch.sh results/all_baselines_5reps_20261005_001408 results/all_baselines_x 'vlm-tamp/T4-*'
# Slow on /coc/flash5 (~30-40 min for 1,350 runs); run it in the background.
set -euo pipefail
OLD=$(cd "$1" && pwd); NEW=$2; shift 2
[ -e "$NEW" ] && { echo "$NEW exists"; exit 1; }
mkdir -p "$NEW/archive"; NEW=$(cd "$NEW" && pwd)
cd "$OLD"
find . -maxdepth 1 -type f ! -name '*.pid' -exec cp -p {} "$NEW/" \;
cp -rp uncertainty_classifier logs "$NEW/"
for d in paper_figures_*; do [ -d "$d" ] && cp -rp "$d" "$NEW/archive/${d}_original"; done
cp -al snapshot "$NEW/"
for planner in react tru-pomdp vlm-tamp; do
    mkdir -p "$NEW/$planner"
    for d in "$planner"/*; do
        moved=""
        for pat in "$@"; do [[ $d == $pat ]] && moved=1; done
        if [ -n "$moved" ]; then
            mkdir -p "$NEW/archive/${planner}_replaced"; cp -al "$d" "$NEW/archive/${planner}_replaced/"
        else
            cp -al "$d" "$NEW/$planner/"
        fi
    done
done
# Hardlinked attempts' summary.json files name the original folder: make status.py read them from this copy.
python3 - "$NEW/status.py" "$OLD" <<'EOF'
import sys
p, old = sys.argv[1], sys.argv[2]
s = open(p).read()
if "ORIGINAL" not in s:
    s = s.replace('ROOT = Path(__file__).resolve().parent\n',
                  f'ROOT = Path(__file__).resolve().parent\nORIGINAL = "{old}"\n', 1)
    a = '                        record["experiment"] = str(attempt)\n'
    assert s.count(a) == 1
    s = s.replace(a, a + '''                        # Runs are hardlinked from the original batch, whose summaries
                        # still name its folder; read them from this copy instead.
                        if record.get("output", "").startswith(ORIGINAL + "/"):
                            record["output"] = str(ROOT) + record["output"][len(ORIGINAL):]
''')
    open(p, "w").write(s)
EOF
echo "copied to $NEW; replaced: $*"
