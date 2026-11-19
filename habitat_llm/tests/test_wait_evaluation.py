"""Wait evidence, lifecycle, scoring order, and the serialized Task 6 contract."""

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationProposition, EvaluationPropositionDependency,
)
from habitat_llm.agent.env.evaluation.predicate_wrappers import (
    PropositionResult, SimBasedPredicates,
)
from habitat_llm.agent.env.evaluation.wait_tracker import WaitEvaluationTracker
from habitat_llm.agent.env.measures import AutoEvalPropositionTracker
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.tools.motor_skills.wait.wait_skill import WaitSkill


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sim(monkeypatch):
    sim = SimpleNamespace(inside={('bread', 'microwave')}, delivered=False)
    monkeypatch.setattr(SimBasedPredicates, 'is_inside', staticmethod(
        lambda sim, object_handles, receptacle_handles, **kw: PropositionResult(
            (object_handles[0], receptacle_handles[0]) in sim.inside,
            {'object_handles': object_handles[0], 'receptacle_handles': receptacle_handles[0]},
        )))
    props = [EvaluationProposition('is_waited_inside', {
        'object_handles': ['bread'], 'receptacle_handles': ['microwave'], 'number': 1,
    })]
    sim._wait_evaluation_tracker = WaitEvaluationTracker(sim, props)
    return sim


def satisfied(sim):
    return SimBasedPredicates.is_waited_inside(sim, ['bread'], ['microwave']).is_satisfied


def make_skill(sim):
    skill = WaitSkill.__new__(WaitSkill)
    skill.env = SimpleNamespace(sim=sim)
    skill.agent_uid = 0
    skill._config = SimpleNamespace(sim_freq=1)
    skill._batch_size = 1
    skill.steps_elapsed = torch.zeros(1)
    skill.failed = False
    skill.set_target('', skill.env)
    return skill


def tick(skill):
    masks = torch.ones(1)
    done = skill._is_skill_done(None, None, None, masks, [0])
    if not bool(done):
        skill._internal_act(None, None, torch.zeros((1, 8)), masks, 0)
    return bool(done)


def test_completed_wait_and_persistent_evidence(sim, monkeypatch):
    skill = make_skill(sim)
    for _ in range(6):
        assert not tick(skill)
        assert not satisfied(sim)
    assert tick(skill)
    assert satisfied(sim)
    monkeypatch.setattr(SkillPolicy, 'reset', lambda *args: None)
    skill.reset([0])
    sim.inside.clear()  # Delivery must not undo the intermediate wait step.
    assert satisfied(sim)


@pytest.mark.parametrize('scenario', ['never_inside', 'removed_midway', 'removed_at_end',
                                      'inserted_midway', 'wrong_container', 'wrong_object',
                                      'failed', 'cancelled'])
def test_invalid_waits_do_not_count(sim, monkeypatch, scenario):
    if scenario in ('never_inside', 'inserted_midway'):
        sim.inside.clear()
    if scenario == 'wrong_container':
        sim.inside = {('bread', 'cabinet')}
    if scenario == 'wrong_object':
        sim.inside = {('bottle', 'microwave')}
    skill = make_skill(sim)
    assert not tick(skill)
    if scenario == 'removed_midway':
        sim.inside.clear()
        assert not tick(skill)
        sim.inside.add(('bread', 'microwave'))
    if scenario == 'inserted_midway':
        sim.inside.add(('bread', 'microwave'))
    if scenario == 'failed':
        skill.failed = True
    if scenario == 'cancelled':
        monkeypatch.setattr(SkillPolicy, 'reset', lambda *args: None)
        skill.reset([0])
    while skill.steps_elapsed[0] <= skill.step_threshold:
        assert not tick(skill)
    if scenario == 'removed_at_end':
        sim.inside.clear()
    assert tick(skill)
    assert not satisfied(sim)


def test_idle_time_and_other_agents_do_not_complete_wait(sim):
    tracker = sim._wait_evaluation_tracker
    tracker.complete(0)  # No explicit start.
    assert not satisfied(sim)
    tracker.begin(0)
    tracker.complete(1)
    assert not satisfied(sim)
    tracker.cancel(0)
    tracker.complete(0)
    assert not satisfied(sim)


def test_metric_reset_and_delivery_dependency(sim, monkeypatch):
    import habitat_llm.agent.env.measures as measures

    monkeypatch.setattr(measures.sim_utilities, 'get_ao_link_id_map', lambda sim: {})
    monkeypatch.setattr(SimBasedPredicates, 'get_state_snapshot_if_none', lambda sim: {})
    monkeypatch.setattr(SimBasedPredicates, 'is_on_top', staticmethod(
        lambda sim, **kw: PropositionResult(sim.delivered)))
    wait_prop = EvaluationProposition('is_waited_inside', {
        'object_handles': ['bread'], 'receptacle_handles': ['microwave'], 'number': 1})
    episode = SimpleNamespace(
        evaluation_propositions=[wait_prop, EvaluationProposition('is_on_top', {
            'object_handles': ['bread'], 'receptacle_handles': ['desk'], 'number': 1})],
        evaluation_proposition_dependencies=[EvaluationPropositionDependency(
            proposition_indices=[1], depends_on=[0], relation_type='after_satisfied')],
        evaluation_constraints=[],
    )
    metric = AutoEvalPropositionTracker.__new__(AutoEvalPropositionTracker)
    metric._sim = sim
    metric.reset_metric(episode=episode, task=None)
    sim.delivered = True
    metric.update_metric()
    assert metric._proposition_satisfied_at == [-1, -1]
    sim.delivered = False
    skill = make_skill(sim)
    while not tick(skill):
        metric.update_metric()
    metric.update_metric()
    assert metric._proposition_satisfied_at[0] >= 0
    assert metric._proposition_satisfied_at[1] == -1
    sim.inside.clear()
    sim.delivered = True
    metric.update_metric()
    assert metric._proposition_satisfied_at[1] > metric._proposition_satisfied_at[0]
    metric.reset_metric(episode=episode, task=None)
    assert metric._proposition_satisfied_at == [-1, -1]
    assert not satisfied(sim)


def test_all_task6_datasets_and_spec_compiler():
    from dataset_generation.benchmark_generation.add_task6_wait_check import add_check
    from dataset_generation.benchmark_generation.audit_variant_specs import parse_spec
    from dataset_generation.benchmark_generation.generate_verified_specs import compile_success

    root = ROOT / 'baseline_evaluation_v3'
    paths = sorted((root / 'episodes/task_6').glob('*/dataset.json'))
    assert len(paths) == 15
    aggregate = {e['episode_id']: e for e in json.loads(
        (root / 'generation/review_dataset.json').read_text())['episodes']}
    for path in paths:
        data = json.loads(path.read_text())
        with gzip.open(path.with_suffix('.json.gz'), 'rt') as f:
            assert json.load(f) == data
        episode = data['episodes'][0]
        before = copy.deepcopy(episode)
        add_check(episode)
        assert episode == before  # Migration is idempotent.
        props = episode['evaluation_propositions']
        wait_indices = [i for i, p in enumerate(props) if p['function_name'] == 'is_waited_inside']
        assert len(wait_indices) == 1
        for c in episode['evaluation_constraints']:
            assert c['args']['n_propositions'] == len(props)
            if c['type'] == 'TerminalSatisfactionConstraint':
                assert wait_indices[0] not in c['args']['proposition_indices']
        assert any(d['depends_on'] == wait_indices and d['relation_type'] == 'after_satisfied'
                   for d in episode['evaluation_proposition_dependencies'])
        spec_path = root / 'specs/T6' / (episode['episode_id'] + '.md')
        spec = parse_spec(spec_path)
        assert spec_path.read_text() == (path.parent / 'spec.md').read_text()
        assert spec['sha256'] == episode['info']['variant_spec']['sha256']
        furniture = json.loads((path.parent / 'scene_info.json').read_text())['receptacle_to_handle']
        compiled, _, order = compile_success(spec, episode['info']['variant_spec']['entity_handles'], furniture)
        assert [vars(p) for p in compiled] == props
        assert any(a == wait_indices[0] for a, b in order)
        assert aggregate[episode['episode_id']]['evaluation_propositions'] == props
