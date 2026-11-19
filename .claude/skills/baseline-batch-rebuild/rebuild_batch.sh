#!/bin/bash
# Recompute everything derived from a baseline batch's runs (all_baselines_5reps_* layout):
#   rows      status.py + build_results_dashboard.py -> summary.json, baseline_results.json, dashboard
#   findings  rule-based classifier on all runs, viewer, import into uncertainty_classifier/ (old one archived)
#   metrics   scripts/dashboard_tc_spl/metrics.py -> tc_spl_data.json + dashboard TC-SPL section
#   figures   paper figures 01-04, 06-10 (+ copied 11) into <batch>/<figures_dir>
# Usage (repo root): rebuild_batch.sh results/<batch> paper_figures_<date> [stage ...]   (default: all stages)
# Offline: no simulator or API calls. Takes ~15 min (classifier ~10).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SKILL=$(cd "$(dirname "$0")" && pwd)
BATCH=$(cd "$1" && pwd); FIGNAME=$2; shift 2
STAGES=${*:-rows findings metrics figures}
PY=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python
PY310=/usr/bin/python3  # classifier needs Python >= 3.10
cd "$REPO"
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has rows; then
    $PY "$BATCH/status.py"
    $PY "$BATCH/build_results_dashboard.py" | tail -30
fi

if has findings; then
    STAGE=$BATCH/uncertainty_classifier_build
    rm -rf "$STAGE"; mkdir -p "$STAGE/viewer"
    SRC=$BATCH/uncertainty_classifier/source
    cp -p "$SRC"/{classify_uncertainties.py,build_viewer.py,PSEUDOCODE.md,README.md} "$STAGE/"
    cp -p "$SRC/viewer/template.html" "$STAGE/viewer/"
    (cd "$STAGE" && $PY310 classify_uncertainties.py "$BATCH" --out findings.csv \
        && $PY310 build_viewer.py "$BATCH" --findings findings.csv --out viewer/rule_viewer.html)
    mkdir -p "$BATCH/archive"
    mv "$BATCH/uncertainty_classifier" "$BATCH/archive/uncertainty_classifier_$(date +%Y%m%d_%H%M%S)"
    $PY scripts/dashboard_tc_spl/import_rules.py "$STAGE" "$BATCH"
    $PY scripts/dashboard_tc_spl/exempt_t3_out_poweroff.py "$BATCH"  # no-op: the classifier applies it already
    rm -r "$STAGE"
fi

if has metrics; then
    $PY scripts/dashboard_tc_spl/metrics.py "$BATCH" | tail -5
fi

if has figures; then
    export BATCH_DIR=$BATCH FIG_DIR=$BATCH/$FIGNAME
    mkdir -p "$FIG_DIR"
    # Figure 11 and task_definitions.json do not depend on results: copy from the newest earlier figure folder.
    PREV=$(ls -d "$BATCH"/paper_figures_* "$BATCH"/archive/paper_figures_* 2>/dev/null | grep -v "/$FIGNAME$" | tail -1 || true)
    if [ -n "$PREV" ] && [ ! -e "$FIG_DIR/task_definitions.json" ]; then
        cp -p "$PREV"/11_task_descriptions_and_goals.* "$PREV"/task_definitions.json "$FIG_DIR/"
    fi
    # Headless Chromium on sky2 lacks libasound; point it at the conda package copy only.
    LIBS=$(mktemp -d); ln -s "$(ls /nethome/vmuriki3/miniconda3/pkgs/alsa-lib-*/lib/libasound.so.2 | head -1)" "$LIBS/"
    LD_LIBRARY_PATH=$LIBS $PY "$SKILL/figure_tools/export_figures.py" "$BATCH" "$FIG_DIR" --overwrite | tail -3
    rm -r "$LIBS"
    for t in refresh_paper_action_spl export_split_overall export_task_spl_bars add_paper_captions make_findings_heatmaps; do
        $PY "$SKILL/figure_tools/$t.py" | tail -1
    done
fi
