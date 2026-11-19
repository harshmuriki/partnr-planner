# Judge progress

The requested outputs are JSONL records from the judge in this conversation. Each output uses findings_<dataset>.csv and the corresponding results index under /coc/flash5/vmuriki3/tamp/partnr-planner/results. The initial audit counted the intersection of findings run IDs and index IDs.

| Dataset and output suffix | Results directory | Expected runs | Findings | Initially completed |
| --- | --- | ---: | ---: | ---: |
| luna_all_planners | all_baselines_5reps_20261005_001408 | 1153 | 6037 | 0 |
| luna_all_planners_t4basket | all_baselines_5reps_vlmtamp_t4basket_20261030 | 1142 | 6035 | 0 |
| astra_react | react_gpt6astra_medium_3reps_20261110 | 197 | 459 | 0 |
| astra_react_placement_oracle | react_gpt6astra_placement_oracle_t3t5_3reps_20261111 | 56 | 115 | 0 |

Resume by checking each judge_<dataset>.jsonl against its own index and findings. Count unique validated run IDs and continue from the missing IDs. Reuse judgments across datasets when the judge inputs match. Compare the actions, trace, starting memory, outcome, task, variant and ordered findings since run IDs alone can recur across different runs.

The batch runner is .judge_work/run.py. It uses the original script's prompt builder and validation, dispatches the prompt to Codex, and caches each valid answer under the SHA-256 digest of the full prompt. It skips run IDs already in their destination output. The initial batch was launched as session 15452. Check the live process before restarting. If it has exited, run python3 .judge_work/run.py from the workspace. The runner takes an exclusive batch lock.

## Latest verified counts

- judge_astra_react.jsonl — 197 distinct runs with attributions
- judge_astra_react_placement_oracle.jsonl — 56 distinct runs with attributions
- judge_luna_all_planners.jsonl — 1153 distinct runs with attributions
- judge_luna_all_planners_t4basket.jsonl — 1142 distinct runs with attributions

## Completed November 18, 2026

All four files passed the source-input audit and field checks. The 2,548 output records cover 12,646 observed findings across the four CSVs. The runner saved 1,321 distinct Codex judgments and reused them for identical prompts. All expected run IDs are present exactly once per output. No work remains.
