import json

from episode_context import episode_context


def packet():
    return {'instruction':'Move both packages.', 'provenance':{'run_id':'react/T2-ACC-ABS/repeat_1'},
            'reference_truth_not_robot_knowledge':{'spec.md':'''# T2-ACC-ABS

## Task instruction / prompt given
"Move both packages."

## Success criteria
- is_on_top(box_0, table_16)
- The scissors and substitutes are absent: report and give up on that subgoal.

## Initial robot memory
- box_0: floor entryway, is_clean
- scissors_0: known absent

## Spawn / planner notes
Do not count this as another memory entry.
'''}}


def test_viewer_summary_and_replay_are_distinct_from_baseline(tmp_path):
    (tmp_path/'variant_object_assets.csv').write_text('Variant_ID,Object,Label,Start_Clean_Dirty,End_Clean_Dirty\nT2-ACC-ABS,box_0 (box),target,is_clean,is_clean\nT2-ACC-ABS,box_1 (box),target,is_dirty,is_clean\n')
    (tmp_path/'ground_truth_timing').mkdir()
    (tmp_path/'ground_truth_timing/replay_measurements.json').write_text(json.dumps({'T2-ACC-ABS':{'status':'replayed','execution_elapsed_seconds':27.2,'sim_steps':387,'action_count':9}}))
    context=episode_context(packet(),tmp_path)
    assert context['episode_id']=='T2-ACC-ABS'
    assert context['memory_condition']=={'code':'ACC','label':'Accurate'}
    assert context['availability_condition']=='ABS'
    assert len(context['success_criteria'])==2 and 'absent' in context['success_criteria'][1]
    assert context['initial_robot_memory']==['box_0: floor entryway, is_clean','scissors_0: known absent']
    assert context['viewer_reference']['object_summary']=={'object_count':2,'objects_with_expected_state_change':1,'label_counts':{'target':2}}
    timing=context['viewer_reference']['ground_truth_replay']
    assert timing['sim_steps']==387 and timing['action_count']==9
    assert timing['is_this_baseline_run'] is False and timing['includes_model_calls'] is False
    assert context['viewer_reference']['source_files_sha256']


def test_archived_spec_wins_over_changed_live_viewer(tmp_path):
    p=packet();directory=tmp_path/'specs/T2';directory.mkdir(parents=True)
    path=directory/'T2-ACC-ABS.md';path.write_text(p['reference_truth_not_robot_knowledge']['spec.md'].replace('table_16','table_99'))
    context=episode_context(p,tmp_path)
    assert 'table_16' in context['success_criteria'][0]
    assert context['scenario_source']['matches_current_viewer_summary'] is False
    assert any('differ' in x for x in context['warnings'])
    before=context['viewer_reference']['source_files_sha256']
    path.write_text(path.read_text().replace('table_99','table_98'))
    assert episode_context(p,tmp_path)['viewer_reference']['source_files_sha256']!=before


def test_missing_sources_remain_unavailable_and_fallback_is_explicit(tmp_path):
    p=packet();p['reference_truth_not_robot_knowledge']={}
    context=episode_context(p,tmp_path)
    assert context['success_criteria'] is None
    assert context['initial_robot_memory'] is None
    assert context['viewer_reference']['object_summary'] is None
    assert context['viewer_reference']['ground_truth_replay'] is None
    directory=tmp_path/'specs/T2';directory.mkdir(parents=True)
    (directory/'T2-ACC-ABS.md').write_text(packet()['reference_truth_not_robot_knowledge']['spec.md'])
    context=episode_context(p,tmp_path)
    assert context['success_criteria'][0]=='is_on_top(box_0, table_16)'
    assert context['scenario_source']['kind']=='current_scenario_viewer_spec_fallback'
    assert context['warnings']
