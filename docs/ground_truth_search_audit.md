# Saved ground-truth search audit

Audit date: 2026-09-29. Read-only check of all completed SQLite recordings, cross-checked against the archive index, run.json, and actions.csv. No saved recordings were changed.

Scope: Navigate to the same room immediately before Explore; also checked all other room-target Navigate actions and repeated Explore calls. Furniture-target navigation (including OUT stale-location checks) is not classified as this violation. This is a sequence audit, not a simulator replay or full semantic validation.

- Expected / saved: 90 / 90.
- Archive integrity findings: 1.
- Rule violations: 8 variants, 20 extra room-navigation actions.
- Historical exceptions: 2 variants (T7-OUT-BASE and T7-OUT-DIS).
- Other room-target Navigate actions: 0.
- Variants with repeated Explore of the same room: 0.

## Findings

| Variant | Navigate steps before Explore | Rooms (same order) | Classification |
| --- | --- | --- | --- |
| T1-INC-ABS | 3, 8, 10, 12 | kitchen_0, living_room_0, bedroom_0, bedroom_1 | Violates current rule |
| T1-INC-BASE | 1 | kitchen_0 | Violates current rule |
| T1-INC-CON | 3 | kitchen_0 | Violates current rule |
| T1-INC-DIS | 1 | kitchen_0 | Violates current rule |
| T1-INC-SUB | 3, 8, 10, 12 | kitchen_0, living_room_0, bedroom_0, bedroom_1 | Violates current rule |
| T1-OUT-ABS | 4, 9, 11, 13 | kitchen_0, living_room_0, bedroom_0, bedroom_1 | Violates current rule |
| T1-OUT-CON | 4 | kitchen_0 | Violates current rule |
| T1-OUT-SUB | 4, 9, 11, 13 | kitchen_0, living_room_0, bedroom_0, bedroom_1 | Violates current rule |
| T7-OUT-BASE | 2 | living_room_0 | Historical exception |
| T7-OUT-DIS | 2 | living_room_0 | Historical exception |

## Archive integrity findings

- T5-ACC-CON unreadable run.json: Expecting value: line 1 column 1 (char 0)

T5-ACC-CON run.json is empty (0 bytes). Its SQLite actions match actions.csv and were included in the sequence audit. All other run.json action sequences and all archive index run IDs match SQLite.

## Historical exception

Repository memory records an explicit request to add Navigate living_room_0 to T7-OUT-BASE/DIS, followed by a later instruction to leave those existing recordings as they are. These are reported separately rather than silently marked for correction. See memory/baseline_evaluation_v3.md, “ACC BASE/DIS for T4-T6 and OUT pattern” and “T7-OUT ground truth”.

## Other room navigation

None.

## Repeated exploration

None.

## All recordings

| Variant | Actions | Matching Navigate → Explore pairs |
| --- | ---: | ---: |
| T1-ACC-ABS | 3 | 0 |
| T1-ACC-BASE | 10 | 0 |
| T1-ACC-CON | 11 | 0 |
| T1-ACC-DIS | 10 | 0 |
| T1-ACC-SUB | 10 | 0 |
| T1-INC-ABS | 14 | 4 |
| T1-INC-BASE | 12 | 1 |
| T1-INC-CON | 13 | 1 |
| T1-INC-DIS | 12 | 1 |
| T1-INC-SUB | 21 | 4 |
| T1-OUT-ABS | 15 | 4 |
| T1-OUT-BASE | 11 | 0 |
| T1-OUT-CON | 13 | 1 |
| T1-OUT-DIS | 11 | 0 |
| T1-OUT-SUB | 22 | 4 |
| T2-ACC-ABS | 9 | 0 |
| T2-ACC-BASE | 12 | 0 |
| T2-ACC-CON | 14 | 0 |
| T2-ACC-DIS | 13 | 0 |
| T2-ACC-SUB | 12 | 0 |
| T2-INC-ABS | 21 | 0 |
| T2-INC-BASE | 15 | 0 |
| T2-INC-CON | 24 | 0 |
| T2-INC-DIS | 14 | 0 |
| T2-INC-SUB | 24 | 0 |
| T2-OUT-ABS | 23 | 0 |
| T2-OUT-BASE | 15 | 0 |
| T2-OUT-CON | 26 | 0 |
| T2-OUT-DIS | 15 | 0 |
| T2-OUT-SUB | 26 | 0 |
| T3-ACC-ABS | 7 | 0 |
| T3-ACC-BASE | 10 | 0 |
| T3-ACC-CON | 12 | 0 |
| T3-ACC-DIS | 10 | 0 |
| T3-INC-ABS | 13 | 0 |
| T3-INC-BASE | 12 | 0 |
| T3-INC-CON | 14 | 0 |
| T3-INC-DIS | 12 | 0 |
| T3-OUT-ABS | 15 | 0 |
| T3-OUT-BASE | 15 | 0 |
| T3-OUT-CON | 17 | 0 |
| T3-OUT-DIS | 15 | 0 |
| T4-ACC-ABS | 17 | 0 |
| T4-ACC-BASE | 20 | 0 |
| T4-ACC-CON | 22 | 0 |
| T4-ACC-DIS | 20 | 0 |
| T4-INC-ABS | 21 | 0 |
| T4-INC-BASE | 21 | 0 |
| T4-INC-CON | 23 | 0 |
| T4-INC-DIS | 21 | 0 |
| T4-OUT-ABS | 22 | 0 |
| T4-OUT-BASE | 22 | 0 |
| T4-OUT-CON | 24 | 0 |
| T4-OUT-DIS | 22 | 0 |
| T5-ACC-ABS | 19 | 0 |
| T5-ACC-BASE | 22 | 0 |
| T5-ACC-CON | 26 | 0 |
| T5-ACC-DIS | 22 | 0 |
| T5-ACC-SUB | 22 | 0 |
| T5-INC-ABS | 24 | 0 |
| T5-INC-BASE | 23 | 0 |
| T5-INC-CON | 27 | 0 |
| T5-INC-DIS | 23 | 0 |
| T5-INC-SUB | 27 | 0 |
| T5-OUT-ABS | 25 | 0 |
| T5-OUT-BASE | 24 | 0 |
| T5-OUT-CON | 28 | 0 |
| T5-OUT-DIS | 24 | 0 |
| T5-OUT-SUB | 28 | 0 |
| T6-ACC-ABS | 18 | 0 |
| T6-ACC-BASE | 21 | 0 |
| T6-ACC-CON | 27 | 0 |
| T6-ACC-DIS | 21 | 0 |
| T6-ACC-SUB | 21 | 0 |
| T6-INC-ABS | 29 | 0 |
| T6-INC-BASE | 25 | 0 |
| T6-INC-CON | 36 | 0 |
| T6-INC-DIS | 25 | 0 |
| T6-INC-SUB | 36 | 0 |
| T6-OUT-ABS | 30 | 0 |
| T6-OUT-BASE | 26 | 0 |
| T6-OUT-CON | 39 | 0 |
| T6-OUT-DIS | 26 | 0 |
| T6-OUT-SUB | 39 | 0 |
| T7-ACC-BASE | 21 | 0 |
| T7-ACC-DIS | 21 | 0 |
| T7-INC-BASE | 22 | 0 |
| T7-INC-DIS | 22 | 0 |
| T7-OUT-BASE | 25 | 1 |
| T7-OUT-DIS | 25 | 1 |

## Correction implications

Removing the flagged room navigations requires fresh recordings to obtain valid simulator-step counts and matching videos. Subtracting old navigation step totals is not a valid replacement for rerunning the remaining actions, because trajectories and Explore behavior can change. No recordings have been modified by this audit.
