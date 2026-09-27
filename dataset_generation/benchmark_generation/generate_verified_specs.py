"""Instantiate audited, fixed specs through the existing generation scripting API.

No language-model interpretation or placement fallback: rejected placements are
retried with the same constraints, and failures remain visible in the manifest.
Existing instruction-generation stages are intentionally not rerun for fixed specs.
"""
import argparse
import copy
import functools
import gzip
import hashlib
import json
import random
import re
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import magnum as mn
import habitat_sim
from PIL import Image
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer

from dataset_generation.benchmark_generation.generate_episodes import (
    default_gen_config, default_metadata_dict, initialize_generator,
    generate_episode, save_ep_dataset,
)
from dataset_generation.benchmark_generation.evaluation_generation.attach_auto_dependencies import infer_and_attach_dependencies
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.agent.env.evaluation.evaluation_functions import (
    EvaluationProposition, EvaluationPropositionDependency, TemporalConstraint,
    TerminalSatisfactionConstraint,
)
from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)



def scene_info_for_spec(root, spec, runtime_furniture):
    """Resolve explicit spec aliases to checked runtime furniture instances."""
    scene = spec['scene']
    info = json.loads((Path('data/datasets/custom/scene_info') / (scene + '.json')).read_text())
    alias_path = root / 'generation/furniture_aliases.json'
    aliases = json.loads(alias_path.read_text()).get(scene, {}) if alias_path.exists() else {}
    for alias, target in aliases.items():
        canonical, handle = target['runtime_name'], target['handle']
        if runtime_furniture.get(canonical) != handle:
            raise ValueError(f'Furniture alias {alias}: runtime handle changed')
        if alias in runtime_furniture and runtime_furniture[alias] != handle:
            raise ValueError(f'Furniture alias {alias}: conflicts with runtime furniture')
        if info['receptacle_to_handle'].get(canonical) != handle:
            raise ValueError(f'Furniture alias {alias}: cached handle changed')
        if info['recep_to_description'].get(canonical) != target['description']:
            raise ValueError(f'Furniture alias {alias}: description changed')
        if canonical not in info['furniture'].get(target['room'], []):
            raise ValueError(f'Furniture alias {alias}: room changed')
        info['receptacle_to_handle'][alias] = handle
        info['recep_to_description'][alias] = target['description']
    if aliases:
        info['spec_furniture_aliases'] = aliases
    return info


def summarize_run(root):
    """Check serialized provenance and object states; assemble verifier input."""
    audit = json.loads((root/'generation/audit.json').read_text())
    index = json.loads((root/'generation/episodes.json').read_text())
    dataset, physical, counts = [], {}, defaultdict(int)
    for vid, item in index['variants'].items():
        counts[item['status']] += 1
        if item['status'] != 'generated':
            continue
        spec = audit['specs'][vid]
        source = root/'specs'/vid.split('-')[0]/(vid+'.md')
        dest = root/item['dataset']
        with gzip.open(dest,'rt') as f:
            packed = json.load(f)
        ep = packed['episodes'][0]
        assert len(packed['episodes']) == 1
        assert ep['episode_id'] == vid
        assert ep['instruction'] == spec['instruction']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['sha256']
        assert hashlib.sha256((dest.parent/'spec.md').read_bytes()).hexdigest() == spec['sha256']
        assert ep['info']['variant_spec']['sha256'] == spec['sha256']
        assert ep['info']['variant_spec']['initial_robot_memory'] == spec['memory']
        assert len(ep['rigid_objs']) == len(spec['entities'])
        handles = ep['info']['variant_spec']['entity_handles']
        for (entity,obj), (asset,transform) in zip(spec['entities'].items(),ep['rigid_objs']):
            assert asset.removesuffix('.object_config.json') == obj['asset'], (vid,entity,asset)
            assert np.isfinite(np.array(transform)).all()
            for state,value in spec['initial'][entity]['states'].items():
                assert ep['object_states'][state][handles[entity]] is value, (vid,entity,state)
        key = (vid.split('-')[0],vid.split('-')[2])
        actual = {k:ep[k] for k in ['rigid_objs','ao_states','object_states','start_position','start_rotation','evaluation_propositions','evaluation_constraints']}
        if key in physical:
            assert actual == physical[key], 'Memory variants differ physically: '+vid
        physical[key] = actual
        dataset.append(ep)
    combined = CollaborationDatasetV0()
    combined.from_json(json.dumps({'episodes':dataset}))
    temp = root/'generation/review_dataset.tmp.json.gz'
    save_ep_dataset(combined.episodes,str(temp))
    temp.replace(root/'generation/review_dataset.json.gz')
    result = dict(counts=dict(counts), serialized_episodes_checked=len(dataset),
        checks=['unchanged source/copied specs','instruction','exact assets and count','initial states','finite transforms','variant memory text','identical physical memory variants'],
        benchmark_ready=False, full_rollouts_performed=False)
    write_json(root/'generation/serialized_validation.json',result)
    print(json.dumps(result),flush=True)


def initial_config(spec, gen):
    rows = []
    mi = gen.metadata_interface
    for entity, obj in spec['entities'].items():
        state = spec['initial'][entity]
        category = mi.get_object_category(obj['asset'])
        if not category:
            raise ValueError('Unknown pinned asset: ' + obj['asset'])
        # The existing sampler UNIONs classes and instances. Excluding every
        # other asset in the class is required to express an exact whitelist.
        templates = mi.get_template_handles_of_class(gen.sim.metadata_mediator, category)
        hashes = {sutils.object_shortname_from_handle(h) for h in templates}
        if obj['asset'] not in hashes:
            raise ValueError('Pinned asset has no loaded template: ' + obj['asset'])
        rows.append(dict(name=entity, number=1, object_classes=[category],
            object_instances=[obj['asset']], excluded_object_instances=sorted(hashes - {obj['asset']}),
            location=state['relation'], furniture_names=['floor' if state['relation']=='floor' else state['furniture']],
            allowed_regions=[state['room']], object_states=state['states'],
            **({'next_to': state['next_to']} if state['next_to'] else {})))
        missing = [other for other in state['next_to'] if other not in {r['name'] for r in rows[:-1]}]
        if missing:
            raise ValueError(f'{entity} must be listed after its next_to partners: {missing}')
    return dict(scene_id=spec['scene'], episode_id=spec['variant'],
        instruction=spec['instruction'], initial_state=rows)


ORDER_LINE = re.compile(r'- order: (is_\w+\([^)]+\)) before (is_\w+\([^)]+\))')


def compile_success(spec, handles, furniture):
    """Bind explicit success predicates by entity identity, never class queues.

    `- order: A before B` makes criterion A a transient step that must hold before B
    is evaluated. Steps are placed after all final-state propositions and returned as
    (step, later) index pairs, so dependency inference only sees the final state.
    """
    lines = spec['success'].splitlines()
    order = [m.groups() for m in map(ORDER_LINE.fullmatch, lines) if m]
    steps = {before for before, _ in order}
    props, unscored, index = [], [], {}
    # Stable sort: final-state criteria keep their order, transient steps go last.
    for line in sorted(lines, key=lambda text: text[2:] in steps):
        if ORDER_LINE.fullmatch(line):
            continue
        m = re.fullmatch(r'- (is_\w+)\(([^)]+)\)', line)
        if not m:
            unscored.append(line)
            continue
        name, raw = m.groups()
        args = [a.strip() for a in raw.split(',')]
        if name in ['is_on_top', 'is_inside']:
            # Pickupable containers are episode entities, not scene furniture.
            destination = handles[args[1]] if args[1] in handles else furniture[args[1]]
            kw = dict(object_handles=[handles[args[0]]], receptacle_handles=[destination], number=1)
        elif name == 'is_next_to':
            kw = dict(entity_handles_a=[handles[args[0]]], entity_handles_b=[handles[args[1]]], number=1, l2_threshold=0.5)
        elif name in ['is_clean','is_dirty','is_filled','is_empty','is_powered_on','is_powered_off','is_on_floor']:
            if len(args) != 1:
                raise ValueError('Unexpected predicate arguments: ' + line)
            kw = dict(object_handles=[handles[args[0]]], number=1)
        else:
            raise ValueError('Unsupported explicit success criterion: ' + line)
        index[line[2:]] = len(props)
        props.append(EvaluationProposition(name, kw))
    if not props:
        raise ValueError('No executable success criteria')
    missing = [c for pair in order for c in pair if c not in index]
    if missing:
        raise ValueError('Ordering references unlisted criteria: ' + ', '.join(missing))
    return props, unscored, [(index[before], index[after]) for before, after in order]


STATE_PROPOSITIONS = {'is_clean', 'is_dirty', 'is_filled', 'is_empty', 'is_powered_on', 'is_powered_off'}


def attach_ordered_steps(ep, props, order):
    """Append transient steps after dependency inference; steps are not required in the
    final state. A later placement is evaluated only once its step has held, so it can be
    redone. Object states persist once set, so a later state must first become true after
    its step (TemporalConstraint on first-satisfaction times)."""
    if len(ep.evaluation_constraints) != 2:
        raise ValueError('Unexpected evaluation constraints: ' + str(ep.evaluation_constraints))
    terminal = next(c for c in ep.evaluation_constraints if isinstance(c, TerminalSatisfactionConstraint))
    n = len(props)
    timed = [(before, after) for before, after in order if props[after].function_name in STATE_PROPOSITIONS]
    gated = [(before, after) for before, after in order if props[after].function_name not in STATE_PROPOSITIONS]
    ep.evaluation_propositions = list(props)
    ep.evaluation_constraints = [TemporalConstraint(dag_edges=timed, n_propositions=n),
        TerminalSatisfactionConstraint(proposition_indices=list(terminal.proposition_indices), n_propositions=n)]
    ep.evaluation_proposition_dependencies = list(ep.evaluation_proposition_dependencies) + [
        EvaluationPropositionDependency(proposition_indices=[after], depends_on=[before], relation_type='after_satisfied')
        for before, after in gated]


def validate_initial(spec, ep, gen):
    objects = gen.ep_sampled_objects
    if len(objects) != len(spec['entities']):
        raise ValueError('Spawned object count differs from spec')
    handles, checks = {}, []
    mi = gen.metadata_interface
    for (entity, expected), obj in zip(spec['entities'].items(), objects):
        actual = sutils.object_shortname_from_handle(obj.handle)
        if actual != expected['asset']:
            raise ValueError(f'{entity}: wrong asset {actual}, expected {expected["asset"]}')
        handles[entity] = obj.handle
        state = spec['initial'][entity]
        for key, value in state['states'].items():
            if ep.object_states.get(key, {}).get(obj.handle) is not value:
                raise ValueError(f'{entity}: incorrect initial {key}')
        rec = gen.object_to_containing_receptacle[obj.handle]
        if state['relation'] == 'floor':
            if rec is not None:
                raise ValueError(entity + ': expected floor')
        else:
            expected_parent = mi.recobj_semname_to_handle[state['furniture']]
            if rec is None or rec.parent_object_handle != expected_parent:
                raise ValueError(entity + ': sampled wrong furniture')
            if (state['relation'] == 'within') != (rec.unique_name in gen._within_rec_names):
                raise ValueError(f"{entity}: '{state['relation']}' placement saved on receptacle {rec.unique_name}, which is {'' if rec.unique_name in gen._within_rec_names else 'not '}an interior")
            predicate = 'is_inside' if state['relation']=='within' else 'is_on_top'
            if not getattr(SimBasedPredicates, predicate)(gen.sim, [obj.handle], [expected_parent]).is_satisfied:
                raise ValueError(f'{entity}: saved placement fails {predicate}({state["furniture"]})')
        region = gen.sim.semantic_scene.regions[mi.region_semname_to_id[state['room']]]
        if not region.contains(obj.translation):
            raise ValueError(entity + ': sampled position is outside specified room')
        checks.append(entity + ': exact asset, initial states, furniture, region checked')
    for entity, state in spec['initial'].items():
        for other in state['next_to']:
            if not SimBasedPredicates.is_next_to(gen.sim, [handles[entity]], [handles[other]]).is_satisfied:
                raise ValueError(f'{entity}: initial next-to {other} failed')
        for parent in state['closed']:
            handle = mi.recobj_semname_to_handle[parent]
            ao = gen.sim.get_articulated_object_manager().get_object_by_handle(handle)
            if ao is not None and any(abs(float(v)) > 1e-4 for v in ao.joint_positions):
                raise ValueError(parent + ': specified closed but joints are not zero')
    return handles, checks


def start_position(spec, gen):
    match = re.search(r'robot starts in ([^\.]+)\.', spec['notes'])
    if not match:
        raise ValueError('Spec has no robot start room')
    room = match[1]
    region = gen.sim.semantic_scene.regions[gen.metadata_interface.region_semname_to_id[room]]
    for _ in range(20000):
        point = gen.sim.pathfinder.get_random_navigable_point()
        if region.contains(point):
            return list(map(float, point))
    raise ValueError('No navigable robot start in ' + room)


def render_views(gen, spec, handles, directory):
    """Render the actual initial transforms, with scene context around each object."""
    directory.mkdir(parents=True, exist_ok=True)
    dbv = DebugVisualizer(gen.sim, resolution=(768, 768))
    result = []
    try:
        for entity, handle in handles.items():
            obj = sutils.get_obj_from_handle(gen.sim, handle)
            state = spec['initial'][entity]
            p = obj.translation
            views = []
            # Context views include nearby furniture; no movement or opening of
            # objects for the image. Contained objects may be occluded by design.
            candidates = []
            distances = [(1.0,0.65,'Angle ')]
            if state['relation']=='within':
                distances += [(0.25,0.12,'Interior ')]
            for distance, elevation, label in distances:
                for k in range(8):
                    angle = k*np.pi/4
                    offset = mn.Vector3(float(distance*np.cos(angle)), elevation, float(distance*np.sin(angle)))
                    ray = habitat_sim.geo.Ray(p+offset, -offset.normalized())
                    hits = gen.sim.cast_ray(ray).hits
                    visible = bool(hits and hits[0].object_id == obj.object_id)
                    candidates.append((visible, label+str(k+1), offset))
            candidates.sort(key=lambda c: c[0], reverse=True)
            for visible, suffix, offset in candidates[:3]:
                obs = dbv.get_observation(look_at=p, look_from=p+offset)
                name = entity + '-' + suffix.lower().replace(' ','-') + '.webp'
                Image.fromarray(obs.obs_data[:,:,:3]).save(directory/name, quality=88)
                views.append(dict(label=suffix, image=name, target_visible=visible))
            result.append(dict(entity=entity, asset=spec['entities'][entity]['asset'], handle=handle,
                initial=state, expected=spec['final'][entity], position=list(map(float,p)), views=views,
                receptacle=gen.object_to_containing_receptacle[handle].unique_name if gen.object_to_containing_receptacle[handle] else 'floor'))
    finally:
        dbv.remove_dbv_agent()
    return result


def rerender_saved(root, selected=None):
    """Reload serialized transforms to render views; never resample an episode."""
    audit = json.loads((root/'generation/audit.json').read_text())
    index = json.loads((root/'generation/episodes.json').read_text())
    gen = initialize_generator(default_gen_config,default_metadata_dict)
    completed = set()
    try:
        for vid, item in index['variants'].items():
            if item['status']!='generated' or (selected and vid not in selected):
                continue
            if item['image_root'] in completed:
                continue
            completed.add(item['image_root'])
            with gzip.open(root/item['dataset'],'rt') as f:
                ep = json.load(f)['episodes'][0]
            gen.initialize_fresh_scene(audit['specs'][vid]['scene'])
            for handle, joints in ep['ao_states'].items():
                ao = gen.sim.get_articulated_object_manager().get_object_by_handle(handle)
                positions = ao.joint_positions.copy()
                for link,value in joints.items():
                    positions[ao.get_link_joint_pos_offset(int(link))] = value
                ao.joint_positions = positions
            manager = gen.sim.get_rigid_object_manager()
            templates = gen.sim.get_object_template_manager()
            handles = ep['info']['variant_spec']['entity_handles']
            for (entity,handle),(asset,transform) in zip(handles.items(),ep['rigid_objs']):
                choices = [h for h in templates.get_file_template_handles(asset) if Path(h).name==asset]
                if not choices:
                    raise ValueError('Saved template unavailable: '+asset)
                obj = manager.add_object_by_template_handle(choices[-1])
                assert obj.handle==handle,(obj.handle,handle)
                obj.transformation = mn.Matrix4(np.array(transform,dtype=np.float32))
                rec = ep['name_to_receptacle'][handle]
                gen.object_to_containing_receptacle[handle] = None if rec=='floor' else SimpleNamespace(unique_name=rec)
            objects = render_views(gen,audit['specs'][vid],handles,root/item['image_root'])
            for other in index['variants'].values():
                if other.get('image_root')==item['image_root']:
                    p=root/other['review'];data=json.loads(p.read_text());data['objects']=objects;write_json(p,data)
            print('RENDERED '+vid,flush=True)
    finally:
        gen.sim.close()


def recompile_evaluation(root, selected=None):
    """Rebuild the evaluation of generated episodes from their current specs, keeping
    sampled placements and views unchanged. The final-state propositions and inferred
    dependencies must match the saved episode, so only ordered steps can be added."""
    audit = json.loads((root/'generation/audit.json').read_text())
    index_path = root/'generation/episodes.json'
    index = json.loads(index_path.read_text())
    as_dict = lambda dep: dep if isinstance(dep, dict) else vars(dep)
    for vid, item in index['variants'].items():
        if item['status'] != 'generated' or (selected and vid not in selected):
            continue
        spec = audit['specs'][vid]
        src = root/'specs'/vid.split('-')[0]/(vid+'.md')
        if hashlib.sha256(src.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Spec changed since audit: '+vid)
        dest = root/item['dataset']
        with gzip.open(dest, 'rt') as f:
            packed = json.load(f)
        dataset = CollaborationDatasetV0()
        dataset.from_json(json.dumps(packed))
        ep = dataset.episodes[0]
        old_props = [(p.function_name, p.args) for p in ep.evaluation_propositions]
        old_deps = [as_dict(d) for d in ep.evaluation_proposition_dependencies]
        handles = ep.info['variant_spec']['entity_handles']
        furniture = json.loads((dest.parent/'scene_info.json').read_text())['receptacle_to_handle']
        props, unscored, order = compile_success(spec, handles, furniture)
        n_final = len(props) - len({before for before, _ in order})
        ep.evaluation_propositions = props[:n_final]
        ep.evaluation_proposition_dependencies = []
        ep.evaluation_constraints = [TemporalConstraint(dag_edges=[], n_propositions=n_final), TerminalSatisfactionConstraint(proposition_indices=list(range(n_final)), n_propositions=n_final)]
        infer_and_attach_dependencies(dataset, override_existing=False)
        ep = dataset.episodes[0]
        new_deps = [as_dict(d) for d in ep.evaluation_proposition_dependencies]
        if old_props[:n_final] != [(p.function_name, p.args) for p in ep.evaluation_propositions] or old_deps[:len(new_deps)] != new_deps:
            raise ValueError('Final-state evaluation differs from the saved episode: '+vid)
        if order:
            attach_ordered_steps(ep, props, order)
        ep.info['variant_spec'].update(sha256=spec['sha256'], unscored_success_text=unscored)
        save_ep_dataset([ep], str(dest))
        shutil.copy2(src, dest.parent/'spec.md')
        review_path = root/item['review']
        review = json.loads(review_path.read_text())
        review.update(spec_sha256=spec['sha256'], unscored_success_text=unscored)
        review.pop('runtime_verified', None)
        write_json(review_path, review)
        item.update(spec_sha256=spec['sha256'], unscored_success_text=unscored)
        item.pop('runtime_verified', None)
        write_json(index_path, index)
        print(f'RECOMPILED {vid}: {len(props)} propositions, {len(order)} ordered steps', flush=True)


PHYSICAL_SPEC_KEYS = ['scene', 'instruction', 'entities', 'initial', 'final', 'success', 'notes']


def reuse_sample(root, audit, index, sample, group):
    """Give memory variants the generated sample of their task/axis unchanged, so all
    memory versions share one physical episode (placements, views and evaluation)."""
    specs, source = audit['specs'], index['variants'][sample]
    if source['spec_sha256'] != specs[sample]['sha256']:
        raise ValueError('Sample spec changed since generation: ' + sample)
    with gzip.open(root/source['dataset'], 'rt') as f:
        packed = json.load(f)
    review = json.loads((root/source['review']).read_text())
    for vid in group:
        current = specs[vid]
        if any(current[k] != specs[sample][k] for k in PHYSICAL_SPEC_KEYS):
            raise ValueError(f'{vid} differs physically from generated sample {sample}')
        src = root/'specs'/vid.split('-')[0]/(vid+'.md')
        if hashlib.sha256(src.read_bytes()).hexdigest() != current['sha256']:
            raise ValueError('Spec changed since audit: '+vid)
        dest = root/'episodes'/('task_'+vid[1])/vid.lower()
        if (dest/'dataset.json.gz').exists():
            raise ValueError('Refusing to overwrite an existing dataset: '+str(dest))
        ep = copy.deepcopy(packed['episodes'][0])
        ep['episode_id'] = ep['info']['episode_id'] = ep['info']['extra_info']['episode_id'] = vid
        ep['info']['variant_spec'].update(variant_id=vid, sha256=current['sha256'],
            initial_robot_memory=current['memory'], reused_sample=sample)
        dest.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest/'dataset.json.gz', 'wt') as f:
            json.dump(dict(packed, episodes=[ep]), f)
        shutil.copy2(src, dest/'spec.md')
        shutil.copy2((root/source['dataset']).parent/'scene_info.json', dest/'scene_info.json')
        data = {k: v for k, v in review.items() if k != 'runtime_verified'}
        data.update(variant=vid, dataset=str((dest/'dataset.json.gz').relative_to(root)), episode_id=vid,
            spec_sha256=current['sha256'], initial_robot_memory=current['memory'],
            issues=index['variants'][vid]['issues'], reused_sample=sample)
        write_json(dest/'review.json', data)
        index['variants'][vid] = {k: v for k, v in data.items() if k != 'objects'}
        index['variants'][vid]['review'] = str((dest/'review.json').relative_to(root))
    print('REUSED '+sample+' for '+', '.join(group), flush=True)


def run(root, only=None, attempts=8):
    audit = json.loads((root/'generation/audit.json').read_text())
    specs = audit['specs']
    index_path = root/'generation/episodes.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'variants': {}}
    blocked = {v for i in audit['issues'] if i['blocking'] for v in i['variants']}
    for vid in specs:
        if vid not in index['variants']:
            index['variants'][vid] = dict(status='blocked' if vid in blocked else 'pending', issues=[i['id'] for i in audit['issues'] if vid in i['variants']])
    write_json(index_path, index)
    gen = None
    groups = defaultdict(list)
    for vid in specs:
        if vid not in blocked and (not only or vid in only):
            groups[(vid.split('-')[0],vid.split('-')[2])].append(vid)
    try:
        for group in groups.values():
            if all(index['variants'][v]['status']=='generated' for v in group):
                continue
            key = (group[0].split('-')[0], group[0].split('-')[2])
            sample = next((v for v, item in index['variants'].items() if item['status']=='generated'
                and (v.split('-')[0], v.split('-')[2])==key), None)
            if sample:
                missing = [v for v in group if index['variants'][v]['status']!='generated']
                try:
                    reuse_sample(root, audit, index, sample, missing)
                except Exception as exc:
                    for vid in missing:
                        index['variants'][vid].update(status='failed', error=str(exc))
                    print('REJECTED reuse '+sample+': '+str(exc), flush=True)
                write_json(index_path, index)
                continue
            spec = specs[group[0]]
            print('GENERATING ' + ', '.join(group), flush=True)
            if gen is None:
                gen = initialize_generator(dict(default_gen_config, enable_check_obj_stability=True), default_metadata_dict)
                # Bound work on geometrically impossible placements; the stock
                # sampler still chooses every transform and applies all constraints.
                gen.sample_objects = functools.partial(gen.sample_objects, max_tries=40)
            failures = []
            for attempt in range(attempts):
                try:
                    seed = int(hashlib.sha256(('-'.join(group)+str(attempt)).encode()).hexdigest()[:8],16)
                    random.seed(seed)
                    np.random.seed(seed)
                    gen.initialize_fresh_scene(spec['scene'])
                    gen.sim.seed(seed)
                    config = initial_config(spec, gen)
                    write_json(root/'generation/inputs'/(group[0]+'.json'), {'initial_state_dicts':[config], 'seed':seed})
                    ep, details = generate_episode(gen, config)
                    if ep is None:
                        raise ValueError(json.dumps(details))
                    handles, checks = validate_initial(spec, ep, gen)
                    ep.start_position = start_position(spec, gen)
                    scene_info = scene_info_for_spec(root, spec, gen.metadata_interface.recobj_semname_to_handle)
                    furniture = dict(gen.metadata_interface.recobj_semname_to_handle)
                    furniture.update({alias: target['handle'] for alias, target in scene_info.get('spec_furniture_aliases', {}).items()})
                    props, unscored, order = compile_success(spec, handles, furniture)
                    n_final = len(props) - len({before for before, _ in order})
                    ep.evaluation_propositions = props[:n_final]
                    ep.evaluation_constraints = [TemporalConstraint(dag_edges=[], n_propositions=n_final), TerminalSatisfactionConstraint(proposition_indices=list(range(n_final)), n_propositions=n_final)]
                    dataset = CollaborationDatasetV0(episodes=[ep])
                    infer_and_attach_dependencies(dataset, override_existing=False)
                    ep = dataset.episodes[0]
                    if order:
                        attach_ordered_steps(ep, props, order)
                    views_dir = root/'gui/episode_views'/group[0]
                    objects = render_views(gen, spec, handles, views_dir)
                    for vid in group:
                        current = specs[vid]
                        src = root/'specs'/vid.split('-')[0]/(vid+'.md')
                        if hashlib.sha256(src.read_bytes()).hexdigest() != current['sha256']:
                            raise ValueError('Spec changed since audit: '+vid)
                        dest = root/'episodes'/('task_'+vid[1])/vid.lower()
                        if (dest/'dataset.json.gz').exists():
                            raise ValueError('Refusing to overwrite an existing dataset: '+str(dest))
                        dest.mkdir(parents=True, exist_ok=True)
                        saved = copy.deepcopy(ep)
                        saved.episode_id = vid
                        saved.info['episode_id'] = vid
                        saved.info['extra_info']['episode_id'] = vid
                        saved.info['variant_spec'] = dict(variant_id=vid, sha256=current['sha256'], entity_handles=handles,
                            initial_robot_memory=current['memory'], memory_runtime_applied=False,
                            unscored_success_text=unscored, seed=seed, initial_validation=checks,
                            source_workbook_sha256=audit['workbook_sha256'])
                        save_ep_dataset([saved], str(dest/'dataset.json.gz'))
                        shutil.copy2(src,dest/'spec.md')
                        write_json(dest/'scene_info.json', scene_info)
                        data = dict(variant=vid, status='generated', dataset=str((dest/'dataset.json.gz').relative_to(root)),
                            episode_id=vid, scene=current['scene'], objects=objects,
                            image_root=str(views_dir.relative_to(root)), checks=checks, unscored_success_text=unscored,
                            spec_sha256=current['sha256'], initial_robot_memory=current['memory'], memory_runtime_applied=False,
                            issues=index['variants'][vid]['issues'], attempts=attempt+1, benchmark_ready=False)
                        write_json(dest/'review.json',data)
                        index['variants'][vid] = {k:v for k,v in data.items() if k!='objects'}
                        index['variants'][vid]['review'] = str((dest/'review.json').relative_to(root))
                        write_json(index_path,index)
                    print('SAVED '+', '.join(group),flush=True)
                    break
                except Exception as exc:
                    failures.append(dict(attempt=attempt+1,error=str(exc),traceback=traceback.format_exc()))
                    print('REJECTED '+group[0]+f' attempt {attempt+1}: '+str(exc)[:800],flush=True)
            else:
                for vid in group:
                    index['variants'][vid].update(status='failed', error=failures[-1]['error'])
                write_json(index_path,index)
            write_json(root/'generation/logs'/(group[0]+'.json'), failures)
    finally:
        if gen is not None:
            gen.sim.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('baseline_evaluation_v3'))
    parser.add_argument('--variant',action='append')
    parser.add_argument('--attempts',type=int,default=8)
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    parser.add_argument('--recompile-evaluation',action='store_true')
    args=parser.parse_args()
    if args.render_only:
        rerender_saved(args.root,args.variant)
    elif args.recompile_evaluation:
        recompile_evaluation(args.root,args.variant)
    elif not args.summarize_only:
        run(args.root,args.variant,args.attempts)
    summarize_run(args.root)
