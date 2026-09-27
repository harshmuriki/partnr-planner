from scripts.trace_annotator.catalog import (
    EFFICIENCY_LABELS,
    FAILURE_LABELS,
    parse_success_criteria_from_dataset,
    parse_success_criteria_from_spec,
)


def test_distractor_note_in_spec_is_not_a_subtask():
    spec = """\
## Success criteria
- is_filled(jug_0)
- Distractors (vase_0, spray_bottle_0) must not be used in place of the task objects.

## Spawn / planner notes
- Keep both distractors in the scene.
"""

    assert parse_success_criteria_from_spec(spec) == ["is_filled(jug_0)"]


def test_distractor_note_is_not_a_subtask():
    dataset = {
        "episodes": [
            {
                "evaluation_propositions": [],
                "info": {
                    "variant_spec": {
                        "unscored_success_text": [
                            "- Distractors (vase_0, spray_bottle_0) must not be used "
                            "in place of the task objects; they have no placement goals.",
                            "- The missing jug should be reported as absent.",
                        ]
                    }
                },
            }
        ]
    }

    assert parse_success_criteria_from_dataset(dataset) == [
        "The missing jug should be reported as absent."
    ]


def test_annotation_labels_are_clear_and_stable():
    failure_labels = {item["id"]: item["label"] for item in FAILURE_LABELS}
    efficiency_labels = {item["id"]: item["label"] for item in EFFICIENCY_LABELS}

    assert failure_labels["no_action_vlm"] == (
        "Missing action — VLM omitted a required action for an object"
    )
    assert failure_labels["placement"] == (
        "Placement error — Placed objects in the wrong location"
    )
    assert efficiency_labels["precondition_failures"] == (
        "Precondition error — Tried an action before its requirements were met"
    )
