from types import SimpleNamespace
from habitat_llm.planner.vlm_tamp_pddl_planner import VlmTampPddlPlanner
from habitat_llm.utils.episode_cost import combined_time_breakdown
from habitat_llm.vlm_tamp.prompts_vlm_tamp import build_english_subgoal_prompt


def test_prompt_describes_all_information_boundaries_and_memory():
    text = build_english_subgoal_prompt('task', {}, '[remembered] box on table')
    for expected in ['Opening furniture', 'Explore(room)', 'execution stops without another VLM request', 'Unknown is not absent', '"decision":"complete"']:
        assert expected in text


def test_budget_includes_physical_explore_without_double_counting_approximation():
    result = combined_time_breakdown(llm_planning_time_s=10, action_sim_steps={'Explore':240, 'Open':120}, explore_approx_sim_time_s=999, physical_explore=True)
    assert result['used_s'] == 13
    assert result['explore_approx_s'] == 2


def test_finished_plan_stops_without_observation_or_vlm_completion_check():
    p=VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    p.is_done=False;p._agents=[];p.trace='task';p._episode_banner_printed=True
    p.branches=[['opened-drawer(drawer)']];p.branch_idx=0;p.subgoal_idx=1
    p.current_plan=[];p.current_action_idx=0;p.enable_partial_obs_explore=False
    p._subgoals_completed=1;p._vprint_header=lambda *a:None
    def forbidden(*args, **kwargs):
        raise AssertionError('A finished plan must not request observations or VLM continuation')
    p._observe_and_continue=p._generate_subgoals=p._observe_failure_state=forbidden
    records=[];p._log_event=records.append;p._trace_append=lambda *a:None
    p._planner_info=lambda *a,**k:k
    actions,info,done=p.get_next_action('original goal',{}, {0:object()})
    assert done and p.is_done and actions=={}
    assert info['is_done']=={0:True} and 'replanned' not in info
    assert p._stop_reason=='plan_completed'
    assert records[-1]['event']=='planner_decision'
    assert records[-1]['evaluator_success']=='not consulted'


def test_observation_continuation_clears_old_plan_and_records_memory_delta():
    p=VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    objects=[SimpleNamespace(name='scissors', properties={})]
    wg=SimpleNamespace(get_all_objects=lambda:objects,get_neighbors=lambda o:{})
    def forbidden(*args):
        raise AssertionError("Planner must not request an extra perception update")
    p.env_interface=SimpleNamespace(update_world_graphs=forbidden)
    p._last_memory_snapshot={}
    p._observe_failure_state=lambda obs,**kw:obs
    p._reprompt_round=0;p.branch_idx=0;p.subgoal_idx=0;p._already_succeeded_subgoals=['opened-drawer(drawer)']
    records=[];p._log_event=records.append;calls=[];p._generate_subgoals=lambda *a,**kw:calls.append((a,kw))
    p._observe_and_continue('task',{},wg,'opened_furniture','drawer')
    assert records[0]['new_objects']==['scissors']
    assert calls[0][1]['observation_reason']=='opened_furniture'
    assert p.current_plan==[] and p.last_high_level_actions=={}


def test_global_cycle_limit_stops_before_another_vlm_call():
    p=VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    p.max_vlm_cycles=2;p._vlm_cycles=2
    calls=[];p._finish_planning=lambda *a:calls.append(a)
    p._generate_subgoals('task',None,{})
    assert calls==[('budget_exhausted','VLM cycle limit reached')]


def test_explicit_complete_is_recorded_as_model_judgment_without_predicate_call():
    p=VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    p.use_images=False;p._reprompt_round=0;p._agents=[];p._already_succeeded_subgoals=[]
    p._extract_visible_entity_names=lambda *a:set()
    p._build_objects_by_type=lambda *a,**k:{}
    p._build_scene_description=lambda *a,**k:'remembered facts'
    p._get_annotated_vlm_image_urls=lambda *a,**k:[]
    p._vprint=p._vprint_header=p._trace_append=lambda *a:None
    p._append_vlm_prompt=lambda **kw:None
    calls=[]
    def ask(*a,**k):
        calls.append(a)
        return '{"decision":"complete","reason":"All requested placements observed"}'
    p.vlm=SimpleNamespace(new_session=lambda:None,ask=ask,get_last_response_metadata=lambda:{})
    p.vlm_max_tokens=100;p.vlm_temperature=0.2;p.vlm_reasoning_effort='high'
    records=[];p._log_event=records.append
    p._generate_subgoals('task',SimpleNamespace(get_all_objects=lambda:[]),{})
    assert p.is_done and p._stop_reason=='complete' and len(calls)==1
    assert records[-1]['evaluator_success']=='not consulted'
    assert p._vlm_calls==1


def test_scene_description_matches_partnr_and_does_not_filter_memory():
    from habitat_llm.llm.instruct.utils import get_world_descr
    from habitat_llm.world_model.world_graph import WorldGraph
    from habitat_llm.world_model import Object, Furniture, Room, House
    graph = WorldGraph()
    room = Room('kitchen', {'type': 'kitchen'})
    table = Furniture('table', {'type': 'table'})
    cup = Object('cup', {'type': 'cup', 'states': {'is_clean': True}})
    house = House('house', {'type': 'root'})
    for node in (house, room, table, cup):
        graph.add_node(node)
    graph.add_edge(room, house, 'inside', 'contains')
    graph.add_edge(table, room, 'inside', 'contains')
    graph.add_edge(cup, table, 'on', 'under')
    p = VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    expected = get_world_descr(graph, agent_uid=0, include_room_name=True,
                              add_state_info=True, centralized=False)
    assert p._build_scene_description(graph, 0, set()) == expected
    assert p._build_scene_description(graph, 0, {'cup'}) == expected
    assert 'cup: table in kitchen' in expected
    assert 'clean: True' in expected


def test_internal_open_replans_without_claiming_parent_subgoal_complete():
    p=VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    p.is_done=False;p._agents=[];p.trace='task';p._episode_banner_printed=True
    p.branches=[['picked(scissors)']];p.branch_idx=0;p.subgoal_idx=0
    p.current_plan=[('open', ['drawer']), ('pick', ['scissors'])]
    p.current_action_idx=0;p.enable_partial_obs_explore=False
    p.last_high_level_actions={0: ('Open', 'drawer', None)}
    p.history=[];p._already_succeeded_subgoals=[];p._subgoal_exec_log_lines=[]
    p.pddl_baseline_html=False
    p._vprint=p._trace_append=lambda *a:None
    p.process_high_level_actions=lambda *a: ({}, {0:'Successful execution!'})
    p._response_failed=lambda *a:False
    p._planner_info=lambda *a,**kw:kw
    statuses=[];p._emit_subgoal_execution=lambda *a:statuses.append(a)
    calls=[];p._observe_and_continue=lambda *a:calls.append(a)
    p.get_next_action('pick scissors', {}, {0:object()})
    assert statuses==[('observation_boundary','picked(scissors)')]
    assert not p._already_succeeded_subgoals
    assert calls[0][-2:]==('opened_furniture','drawer')
    assert p._search_counts=={'inspect:drawer':1}


def test_implicit_explore_does_not_claim_manipulation_goal_succeeded():
    p=VlmTampPddlPlanner.__new__(VlmTampPddlPlanner)
    p.is_done=False;p._agents=[];p.trace='task';p._episode_banner_printed=True
    p.branches=[['picked(scissors)']];p.branch_idx=0;p.subgoal_idx=0
    p.current_plan=[('__special_explore__', ['kitchen'])]
    p.current_action_idx=0;p.enable_partial_obs_explore=False
    p.last_high_level_actions={0: ('Explore', 'kitchen', None)}
    p.history=[];p._already_succeeded_subgoals=[];p.pddl_baseline_html=False
    p._pending_replan_after_explore=True;p._reprompt_round=0
    p._vprint=p._vprint_header=p._trace_append=lambda *a:None
    p._log_event=p._log_plan_tree=p._append_subgoal_exec_log_to_trace=lambda *a:None
    p._capture_explore_image=lambda *a,**kw:None
    p.process_high_level_actions=lambda *a: ({}, {0:'Successful execution!'})
    p._response_failed=lambda *a:False
    p._planner_info=lambda *a,**kw:kw
    statuses=[];p._emit_subgoal_execution=lambda *a:statuses.append(a)
    calls=[];p._generate_subgoals=lambda *a,**kw:calls.append(kw)
    p.get_next_action('pick scissors', {}, {0:object()})
    assert statuses==[('observation_boundary','picked(scissors)')]
    assert not p._already_succeeded_subgoals
    assert calls[0]['after_explore']
    assert 'picked(scissors)' not in calls[0]['history_str']
