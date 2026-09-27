# Current generation run

Source: `source_tasks.xlsx` is the exported Google workbook; `source_cells.json`
records the nonempty cells of Tasks and Tasks Reformated. `audit.json` records
source hashes, all parsed specs, acknowledged user overrides and active issues.
Original specs and CSV are not edited by this workflow.

The audit separates blocking spec/task disagreements from runtime/evaluation
limitations. Blocked variants are not generated pending review. Failed physical
placements are not moved elsewhere or replaced with different assets.

## Pipeline used

Fixed Markdown specs are parsed into exact inputs by
`dataset_generation.benchmark_generation.generate_verified_specs`. It uses the
repository's `generate_episodes.generate_episode` scripting API, its object and
receptacle samplers, physics settling, and dataset serializer. It does not call
the episode-generator GUI service. The existing sampler's class/instance union
is restricted with explicit excluded asset hashes to pin each object.

The upstream instruction-generation LLM and the evaluation-generation LLM are
not run: instructions and success predicates already exist in the specs. The
adapter binds those explicit predicates to exact spawned entity handles, creates
standard evaluation constraints, and runs the existing dependency inference.
This uses the existing instantiation and evaluation components; it is not a
claim that every stage of the natural-language benchmark-generation pipeline ran.

Each successful task/scenario sample is reused unchanged across its eligible
ACC/INC/OUT variants. Each episode retains its own spec and memory text. **The
memory text is not connected to the runtime planner. These are placement-review
episodes, not fully implemented uncertainty benchmarks.** Non-predicate success
requirements (such as reporting absent objects) are retained as unscored text.

## Reproduce

From the repository root:

```bash
python3 -m dataset_generation.benchmark_generation.audit_variant_specs
conda activate habitat
python -m dataset_generation.benchmark_generation.generate_verified_specs
python -m dataset_generation.benchmark_generation.verify_dataset \
  --dataset-path baseline_evaluation_v3/generation/review_dataset.json.gz \
  --save-results-dir baseline_evaluation_v3/generation/runtime_verification \
  --num-proc 1
```

The generator resumes completed groups and records explicit failures in
`episodes.json` / `logs/`. One object is sampled per spec entity; no object-asset
replacement is performed after sampling. Seeds and exact sampler inputs are
saved under `inputs/`. Placement retries retain the same constraints.

The GUI fetches `episodes.json`, per-episode `review.json`, and real rendered
camera views. Cameras do not move episode objects or open closed containers.
`--render-only` reloads saved transforms to regenerate images. Interior camera
angles allow inspecting objects inside closed furniture without changing joints.

## Verification limits

Generation checks exact asset identity, explicit initial boolean states, sampled
furniture and room, spatial predicates, specified initial adjacency, and closed
articulated joints. Physics settling is enabled. The existing `verify_dataset`
module additionally checks saved episodes can initialize and resolve evaluation
handles. These checks do not prove navigation, manipulation, or task success;
oracle-skill rollouts were run on the ACC-BASE episodes. Open issues, if any, are listed in `audit.json`.
