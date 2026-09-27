# Current episode validation — 2026-09-27

90 / 90 retained variants have saved datasets and pass the ground-truth provenance gate.

| Task | Saved | Expected |
|---|---:|---:|
| T1 | 15 | 15 |
| T2 | 15 | 15 |
| T3 | 12 | 12 |
| T4 | 12 | 12 |
| T5 | 15 | 15 |
| T6 | 15 | 15 |
| T7 | 6 | 6 |

All 90 episodes passed fresh Habitat initialization, evaluation-handle and constraint-argument checks. Source/copied specs, provenance, current compiled scoring rules, and memory agree.

Regenerated 15 T2 episodes. Rebuilt 30 T5/T6 evaluations while preserving their initial placements. Source specs and saved ground truth were not changed.

No full task rollouts were performed. Details: [regeneration_report.json](regeneration_report.json), [current_scan.json](current_scan.json).
