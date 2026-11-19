# Simulator results and Astra annotations

The revised study is `results/annotation_astra_20260929`. It covers all 1,350
saved runs: 270 each for VLM-TAMP with/without images, PARTNR ReAct with/without
images, and TruPOMDP without images. The judge is **gpt-6-astra, high effort**.
The original runs and `results/annotation_study_20260928` remain separate.

## Success criteria

Success is an immutable simulator result. Astra cannot create, change, or replace
criterion scores. Criterion IDs `p0`, `p1`, etc. correspond to propositions in
the archived run's `input/dataset.json.gz`.

When structured terminal evaluation is available, a criterion is satisfied only
if its proposition was satisfied and all applicable constraints pass. This
matches the simulator's evaluator: temporal prerequisites can be satisfied
earlier in the episode, while terminal constraints enforce required final states.
Checking only a final object snapshot would incorrectly reject some ordered tasks.

For the existing cohort, saved aggregate metrics establish **425 complete,
405 partial, and 127 no-progress runs**. **393 runs have no saved final evaluation**
and remain unavailable, rather than being assigned an AI success/failure score.

Older runs do not always retain per-proposition arrays. For these runs:

- A score of 100% establishes that every scored criterion passed; 0% establishes
  that none passed.
- Partial scores and exact simulator failure explanations resolve criteria only
  when the mapping is unambiguous. The extractor uses the simulator's category
  metadata and message formatting, including terminal and temporal violations.
- If two object instances have the same class description, the extractor retains
  every assignment consistent with the saved explanation and aggregate score.
  It never guesses which instance failed. Groups `g0`, etc. identify uncertain
  criteria and the known number of failures among them.
- Missing scores stay unknown. Unscored specification notes, including absence
  reporting, are available to Astra as notes but cannot change the simulator score.

Each saved simulator record includes its source, recovery method, original
metrics, proposition definitions, constraints, and archived dataset hash.

## Failure and inefficiency definitions

The complete definitions are in
[`scripts/trace_annotator/astra_rubric.md`](../scripts/trace_annotator/astra_rubric.md)
and copied into every request. The confirmed distinction is:

- **Insufficient planning:** getting stuck in unproductive execution rather than
  advancing or giving up. A timeout alone is not evidence of this cause.
- **Failure to stop:** continuing planning/search after evidence that the subgoal
  should be abandoned.

Failure labels require an unsuccessful simulator-scored subtask and must link to
an unmet criterion or a documented ambiguous failure group. Other failure labels
are missing action, placement error, and wrong object. Successful runs cannot
receive failure labels.

Inefficiencies can occur with any outcome: hallucinated object, extra subgoals,
repeated actions, and precondition error. An outdated-memory object's navigation
attempts and initial failed pick are exempt from hallucination counts. A later
misuse requires citations to both the earlier failed pick/removal and later use.
Distinct high-level attempts are counted, not simulation ticks or motor diagnostics.

Astra returns a summary, failure incidents, inefficiency incidents, important
notes, and possible failure notes. Each incident has a category, criterion/group
links, source evidence IDs, explanation, and confidence. Missing final evaluation
permits possible failure notes but not asserted failure labels. Unclassified issues
can be described in notes without silently changing the taxonomy.

## Full saved history and provenance

Each request includes the task specification, simulator result, reference maps,
saved traces, planner steps, PDDL events, console history, separately saved planner
prompts, and revised rubric. Reference truth is explicitly distinguished from
the robot's knowledge. Prior Luna annotations are not supplied.

There is no event-budget truncation. Identical navigation diagnostics are folded
with counts; unchanged simulation ticks are removed; repeated text is represented
by lossless text-block references. All saved substantive history is retained.
An externally killed episode may still lack a final snapshot. Packets too large
for the model's context are recorded as errors rather than silently shortened.
The judge receives text histories, not the planner's image payloads.

Requests save the exact packet and hash, rubric hash, system prompt, returned
model name, raw response, token usage, validation result, and evidence fingerprint.
Validation enforces simulator authority, category families, evidence IDs, and the
stale-memory citation rule. These are AI annotations, not human-verified labels.

## Running and inspecting

From the repository root:

```bash
PYTHON=/nethome/vmuriki3/miniconda3/envs/habitat-llm/bin/python
STUDY=results/annotation_astra_20260929

# Inspect progress without sending requests.
"$PYTHON" scripts/trace_annotator/astra_study.py --study "$STUDY" status

# Resume missing annotations after the existing batch has stopped.
# Successfully saved analyses are skipped; failed requests can be retried.
"$PYTHON" -u scripts/trace_annotator/astra_study.py --study "$STUDY" run --workers 8

# Start the dashboard if its existing server is not running.
"$PYTHON" scripts/trace_annotator/astra_study.py --study "$STUDY" serve --port 5057
```

The launched batch and dashboard have `batch.pid`, `batch.log`, `server.pid`, and
`server.log` in the study directory. Requests load credentials from the repository
`.env`. Run locks prevent concurrent processes from annotating the same run twice.
Batch manifests list every selected run and each completed request's status.

Open <http://localhost:5057> (forward port 5057 when using a remote IDE). The
dashboard supports condition/task/memory/uncertainty/repetition filters, simulator
outcomes, failure and inefficiency prevalence, individual criteria and notes,
clickable source evidence, exact input downloads, and CSV export. Refresh progress
to load newly saved annotations. Tag denominators count annotated runs only;
simulator outcomes remain independent of annotation progress. Tags may overlap.

Study artifacts are separated by condition and run ID:

```text
study.json                       all runs and immutable simulator results
rubric.md, codebook.json          revised definitions
simulator/<condition>/<run>.json  simulator result and recovery provenance
astra_packets/<condition>/...    exact full-history model inputs
astra_attempts/<condition>/...   requests, raw responses, usage, validation
analyses/<condition>/<run>.json   validated Astra labels and fixed simulator result
errors/<condition>/<run>.json     context, API, or validation failures
batches/*.json                   durable queue/progress records
```

After a successful retry, an older error file may remain as a record of that
attempt; the saved analysis takes precedence. No original experiment output is
overwritten, and this workflow does not launch new simulator runs.

## Validation

Regression tests cover simulator authority, missing metrics, duplicate object
classes, temporal constraints, failure/subtask links, outdated-memory exceptions,
and full-history retention beyond the old packet limit:

```bash
"$PYTHON" -m pytest -q scripts/trace_annotator/test_astra.py scripts/trace_annotator/test_study.py
```

The initial real Astra request passed validation. Browser checks exercised run
selection, fixed simulator criteria, incident evidence, and condition filtering.
