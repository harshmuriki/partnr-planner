# Evidence-linked planner annotation

The study imports **1,350 existing runs**, 270 in each condition:

| Condition | Original results directory |
| --- | --- |
| VLM-TAMP without images | `image_comparison_20260927_232820/vlm_tamp_no_images` |
| PARTNR/ReAct without images | `image_comparison_20260927_232820/react_no_images` |
| PARTNR/ReAct with images | `image_comparison_20260927_232820/react_images` |
| TRU-POMDP without images | `tru_pomdp_no_images_20260928_002709` |
| VLM-TAMP with images | `vlm_tamp_images_10min_20260928_111039` |

Tasks 1–5 use a 600-second (10-minute) outer deadline. **Tasks 6 and 7 use a
1800-second (30-minute) deadline**: their `task_6/` and `task_7/` folders hold the
reruns from `results/tasks67_30min_20260930`, and the earlier 10-minute runs are in
each directory's `archive_10min/`. Import enforces this per-task limit
(`LONG_TIMEOUT_TASKS` in `study.py`). Task 7 is scored with relaxed containment
(both toys in the same wardrobe, plate in any cabinet); existing runs were rescored
offline in `results/t7_relaxed_rescore/`. Each condition contains
90 task/memory/uncertainty variants, three repetitions each. The supported axes are
BASE, SUB, ABS, CON, DIS; not every task defines every axis. Earlier 30-minute
VLM-TAMP image runs, test runs, and retry batches are excluded. Original experiment files are read-only.
The experiments used GPT-5.6 Luna high; the **annotation judge is GPT-6 Luna high**.

## Run the reviewer

From the repository root:

```bash
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python scripts/trace_annotator/study_cli.py \
  --study results/annotation_study_20260928 serve --port 5056
```

Open `http://localhost:5056` using VS Code port forwarding. The server binds to
localhost. The separate legacy annotator remains available through `app.py`.
Runtime dependencies are in `scripts/trace_annotator/requirements.txt`.

Choose **Review runs**, select a pilot run, and inspect the source evidence beside
the AI draft. Search an evidence ID to see its source file, line range or step.
Images come only from that exact run. The judge receives text, not images; images
are available to the human reviewer. No screenshots are borrowed across repetitions.

1. Review each required criterion and constraint as met, unmet, unknown, or not
   applicable. A met/unmet decision requires evidence IDs and an explanation.
2. Correct incident labels, causal explanations, evidence, and affected criteria.
   One incident is one distinct high-level attempt or decision, not a simulation tick.
3. Approve or reject proposed categories. Approving adds a versioned codebook entry
   and an editable incident; check its evidence and affected criteria before saving.
4. Set the whole-run issue assessment to **assessed** only with sufficient evidence.
   Zero incidents plus assessed explicitly means issue-free. Partial evidence can
   retain confirmed incidents while leaving whole-run assessment unknown.
5. Enter your reviewer name and save. Revisions are retained; concurrent stale saves
   are rejected. AI drafts never become human reviews automatically.

The pilot is **50 runs**, ten per condition, covering all seven tasks in each
condition with additional memory/uncertainty and timeout coverage. Review every
pilot run before approving the pilot. The remaining 1,300 drafts are gated on that
approval, so rubric problems can be caught before the full annotation batch.

## Draft commands

The API model is `gpt-6-luna` with `reasoning_effort="high"`, corresponding to the
requested `gpt-6-luna-high`. The judge never falls back to a different model.
Credentials come from the repository `.env` or existing environment variables.

```bash
# Resume the pilot; existing drafts and human reviews are skipped.
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python scripts/trace_annotator/study_cli.py \
  --study results/annotation_study_20260928 draft --pilot --workers 4

# After all 50 human reviews, approve in the UI or use:
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python scripts/trace_annotator/study_cli.py \
  --study results/annotation_study_20260928 approve-pilot --reviewer "YOUR NAME"

# Draft the remainder, preserving reviewed and drafted work.
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python scripts/trace_annotator/study_cli.py \
  --study results/annotation_study_20260928 draft --all --workers 4

# Current coverage (does not call the model).
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python scripts/trace_annotator/study_cli.py \
  --study results/annotation_study_20260928 status
```

## Scoring rules

Execution completion, timeout/crash, benchmark scores, criteria completion, and
causal issues are separate fields. A successful process is not necessarily a
successful task. A timeout is not automatically a planning failure.

The archived `## Success criteria` sections are audited against the supplied
criteria attachment, including absence reporting, accepted substitutes, temporal
order, and distractor constraints. The only display normalization is `hand_towel`
to “hand towel”; entity IDs remain unchanged. Task 5 and Task 6 ordering conditions
remain separate criteria. A required transient state such as bread in the microwave
can hold earlier, before its required final placement. Distractor prohibitions do
not inflate the subtask denominator.

Missing evidence stays unknown. Failure causes need an unmet criterion and causal
evidence. Failed skill calls are not automatically precondition failures. A missing
world-graph object or stale alias is not automatically a hallucinated object.
Reference scene truth is explicitly separated from information the robot knew.
Object aliases and geometric relations that cannot be established remain unknown.
The human review is essential: structural validation cannot prove a judge's causal
interpretation correct.

The initial codebook preserves the five requested failure labels and four
inefficiency labels. The judge can propose additional categories with definitions,
examples and evidence. Proposals do not contribute to statistics until approved
and saved in a human review.

## Charts and denominators

- Outcomes use reviewed criteria and explicitly show unknown and unreviewed runs.
  A violated distractor constraint is displayed separately from partial progress.
- Issue prevalence is unique reviewed, fully issue-assessed runs with a tag divided
  by all such runs in scope. Multiple tags can overlap; percentages need not sum to 100%.
- Category occurrence counts count incident-label assignments. Per-run distributions
  count distinct incidents per family, so two failure labels on the same incident
  do not double that run's failure count.
- Partial progress averages apply only to reviewed partial outcomes. Unknown
  criteria are reported separately, never converted to zero.
- Comparisons retain all repetitions. Per-variant averages and task-balanced
  success display coverage; pilot estimates are not final cohort results.
- Filter by planner/image setting, task, memory, uncertainty, repetition, execution
  status, or review state. Export runs, criteria, incidents as CSV, or complete JSON.

## Saved files

Everything new is under `results/annotation_study_20260928/`:

| Path | Contents |
| --- | --- |
| `study.json` | Frozen run catalog, settings, specs, pilot membership, model and review gate |
| `attached_criteria.md`, `criteria_audit.json` | User reference and audit of all archived variants |
| `codebook.json` | Approved category definitions and version |
| `evidence/<condition>/<run>.json` | Source-linked evidence, file hashes, same-run artifact inventory |
| `packets/<condition>/<run>.json` | Actual text evidence packet sent to the judge |
| `packets_by_hash/*.json` | Immutable packet copies, including superseded drafts |
| `attempts/<condition>/<run>/*.json` | Raw responses, usage, model, errors, prompt version; retains retries |
| `drafts/<condition>/<run>.json` | Validated AI draft and provenance |
| `reviewed/<condition>/<run>.json` | Latest explicit human review |
| `history/<condition>/<run>.rN.json` | Review history and original draft hash |
| `last_draft_batch.json` | Results of the most recent drafting invocation |
| `draft_batches/*.json` | Per-invocation progress and results |
| `superseded_drafts/` | Earlier pilot drafts retained when packet construction was improved |

Raw response usage includes repair attempts; count all attempt files when auditing
API token usage. Evidence packets remove unchanged simulation ticks and large API
image/envelope fields. Repeated navigation diagnostics retain their exact count;
repeated long text can be represented by shared blocks without losing its contents.
Any substantive omitted events are listed, and whole-run
issue assessment is forced to unknown. The first pilot preflight used prompt v1;
v2 adds exact-ID formatting and cautions about stale object aliases; v3 explains
shared text blocks and diagnostic counts. Affected pilot drafts were regenerated
with improved packets and their earlier versions retained. Prompt version is
recorded per draft. No version's drafts are treated as human labels.

## Future experiments

The evaluation runner now appends `evidence-<episode>.jsonl` to its existing output
directory at initialization, changed high-level action/state boundaries and termination. This saves
planner-observed information and separately named evaluator-only measurements. It
also saves evaluator reference graph states and simulator handles to resolve runtime
object and furniture aliases against the archived inputs. This reference truth
stays separate from the robot's knowledge. It
does not add model calls or expose evaluator state to the planner. Existing measures
are read, not advanced. Completed lines survive external termination; a truncated
last line is ignored by the importer. Set `PARTNR_ANNOTATION_EVIDENCE=0` to disable.
Logging errors issue a warning and disable the writer rather than terminating the run.
This cannot reconstruct missing state from old interrupted experiments.

## Validation

```bash
/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python -m pytest \
  scripts/trace_annotator/test_study.py \
  habitat_llm/tests/test_trace_annotator_catalog.py \
  habitat_llm/tests/test_combined_time_budget.py -q
```

Tests cover unknown outcomes, causal links, evidence ownership, review revisions,
unique incident counts, category approval, draft resumption, exact judge settings,
API failure handling, the pilot gate, safe artifact access, and partial JSONL logs.
