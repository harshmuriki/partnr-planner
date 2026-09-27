# Ground-truth archive

Open index.html through the scenario viewer server, or use index.csv/index.json for analysis.

Layout: T<task>/<variant>/run.json, actions.csv, video.mp4, initial.jpg, clips/, room_choices.json,
furniture_choices.json (ranked likely furniture per target; shared by variants in the same apartment).
There is exactly one ground truth per variant. Recording again replaces it; no history is kept.
Ground truth is recorded by replaying the checked sandbox steps from the episode's exact start.
Sandbox (practice) actions are never saved.
Run JSON includes action results, evaluation state, spec/dataset hashes, video size/SHA-256,
and room choices at the time it was saved. Reports add zero simulator steps.

The live source of truth is ../ground_truth.sqlite3 (actions, counts, video checksum).
Keep this folder and that database together when backing up. Use sqlite3's backup API
for a live database backup, or stop the viewer before copying the database.
