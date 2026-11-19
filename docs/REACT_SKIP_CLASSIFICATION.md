# ReAct missing-object picks for new runs

New ReAct runs record `ReAct skip policy: robot_sg_before_action` in their trace. For these runs, `Unexpected failure! - Failed to pick! Object does not exist.` is a blue **Skipped** attempt only when the exact object target existed in the robot's scene graph before dispatch. An absent target or unavailable prior evidence remains a red failure. The optional `Should be skipped` suffix alone does not grant an exemption.

The planner records `Robot SG before action:` followed by a JSON list of remembered object names before executing each new action. This uses the robot world graph, not simulator truth. The snapshot survives a failed skill removing a stale-memory object, and the next attempt gets a fresh snapshot. This metadata does not enter the model prompt.

The HTML viewer, terminal parser, annotation catalog, and judge use the same rule. Skips retain raw feedback but are excluded from error counts and the success-rate denominator. HTML records the policy, prior membership, and evidence source as data attributes.

Existing runs without the new policy/snapshot markers retain their previous classification. Their artifacts and judge responses are not retroactively updated.
