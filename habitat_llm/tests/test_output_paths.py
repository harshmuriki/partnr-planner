#!/usr/bin/env python3

from habitat_llm.vlm_tamp.output_paths import aggregate_outputs_dir_from_results_dir


def test_aggregate_outputs_dir_from_run_results_dir(tmp_path):
    run_dir = tmp_path / "results" / "vlm_tamp_pddl_v5" / "Task_5_Sg" / "Task_5_Sg_1"
    expected = tmp_path / "results" / "outputs_vlm_tamp_pddl_v5"
    assert aggregate_outputs_dir_from_results_dir(str(run_dir)) == expected.resolve()


def test_aggregate_outputs_dir_from_experiment_results_dir(tmp_path):
    experiment_dir = tmp_path / "results" / "vlm_tamp_pddl_v5"
    expected = tmp_path / "results" / "outputs_vlm_tamp_pddl_v5"
    assert (
        aggregate_outputs_dir_from_results_dir(str(experiment_dir))
        == expected.resolve()
    )
