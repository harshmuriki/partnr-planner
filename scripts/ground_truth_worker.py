#!/usr/bin/env python3
"""Habitat subprocess for the scenario viewer (JSON commands on stdin)."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/skill_runner_gui'))
from session import SkillRunnerSession, _json_safe
import magnum as mn
import imageio
import omegaconf


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(_json_safe(data)))
    os.replace(temporary, path)


class GroundTruthSession(SkillRunnerSession):
    # Ground truth only: oracle_nav.yaml caps Navigate at 600 skill steps, which cuts off long
    # cross-apartment trips (T6 laundry room -> bathroom_2). Match run_skill's 2400 cap here
    # instead of editing the shared config the benchmarked planners use. A Navigate that
    # finished under 600 steps behaves identically with the higher cap.
    NAVIGATE_MAX_SKILL_STEPS = 2400

    def _compose_config(self, *args, **kwargs):
        config = super()._compose_config(*args, **kwargs)
        def raise_nav_cap(node):
            if isinstance(node, omegaconf.DictConfig):
                if node.get('name') == 'Navigate' and 'max_skill_steps' in node:
                    node.max_skill_steps = max(node.max_skill_steps, self.NAVIGATE_MAX_SKILL_STEPS)
                for key in node.keys():
                    raise_nav_cap(node._get_node(key))
            elif isinstance(node, omegaconf.ListConfig):
                for item in node:
                    raise_nav_cap(item)
        with omegaconf.open_dict(config):
            raise_nav_cap(config.evaluation)
        return config

    def _do_run_skill(self, **kwargs):
        # Count actual low-level step calls even if execute_skill later fails.
        # Its exception path otherwise reports zero steps for a failed attempt.
        action_started = time.perf_counter()
        self._action_timing = {'environment_step_wall_seconds': 0.0,
                               'frame_callback_wall_seconds': 0.0,
                               'clip_close_wall_seconds': 0.0}
        steps = 0
        env = self.env_interface
        original_step = env.step
        def counted_step(*args, **kw):
            nonlocal steps
            steps += 1
            step_started = time.perf_counter()
            try:
                result = original_step(*args, **kw)
            finally:
                self._action_timing['environment_step_wall_seconds'] += time.perf_counter() - step_started
            # Publish evaluator state during long skills, without stepping again.
            if time.monotonic() - getattr(self, '_last_evaluation_publish', 0) > 0.15:
                self._last_evaluation_publish = time.monotonic()
                write_json(self.ipc_dir / 'evaluation.json', {
                    'load_id': self.load_id, 'evaluation': self.evaluation()})
            return result
        clip = self.results_dir / 'clips' / f'{self.command_index:04d}.mp4'
        clip.parent.mkdir(parents=True, exist_ok=True)
        self._recording_error = None
        self._recording_frames = 0
        self._recording_writer = None
        try:
            self._recording_writer = imageio.get_writer(str(clip), fps=30, codec='libx264',
                quality=7, pixelformat='yuv420p', macro_block_size=2)
        except Exception as error:
            self._recording_error = str(error)
        env.step = counted_step
        try:
            entry = super()._do_run_skill(**kwargs)
        finally:
            env.step = original_step
            if self._recording_writer is not None:
                close_started = time.perf_counter()
                try:
                    self._recording_writer.close()
                except Exception as error:
                    self._recording_error = str(error)
                self._action_timing['clip_close_wall_seconds'] += time.perf_counter() - close_started
                self._recording_writer = None
        entry['timing'] = {**self._action_timing, 'action_wall_seconds': time.perf_counter() - action_started}
        self._action_timing = None
        entry['skill_steps'] = steps
        entry['steps_source'] = 'environment_step_calls'
        entry['recorded_frames'] = self._recording_frames
        entry['recording_error'] = self._recording_error
        entry['ground_truth_video'] = str(clip.relative_to(self.results_dir)) if clip.exists() and self._recording_frames else None
        return entry

    def _apply_runtime_config(self):
        # RearrangeTask samples agent starts even when the dataset has a saved pose.
        # Restore robot 0 before the first frame or counted action, without stepping.
        env = self.env_interface
        episode = env.sim.ep_info
        robot = env.sim.get_agent_data(0).articulated_agent
        robot.base_pos = mn.Vector3(episode.start_position)
        x, y, z, w = episode.start_rotation
        robot.base_rot = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
        robot.update()
        graph = env.perception.get_recent_graph()
        for world_graph in env.world_graph.values():
            world_graph.update(graph, False, 'gt')
        env.full_world_graph.update(graph, False, 'gt')
        self.active_world_graph = env.world_graph[self.config.robot_agent_uid]

    def set_frame_rgb(self, frame):
        started = time.perf_counter()
        try:
            self._record_frame_rgb(frame)
        finally:
            timing = getattr(self, '_action_timing', None)
            if timing is not None:
                timing['frame_callback_wall_seconds'] += time.perf_counter() - started

    def _record_frame_rgb(self, frame):
        super().set_frame_rgb(frame)
        writer = getattr(self, '_recording_writer', None)
        if writer is not None and not self._recording_error:
            try:
                writer.append_data(frame)
                self._recording_frames += 1
            except Exception as error:
                self._recording_error = str(error)
        if time.monotonic() - getattr(self, '_last_publish', 0) > 0.15:
            self._last_publish = time.monotonic()
            path = self.ipc_dir / 'frame.jpg'
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(self.get_latest_jpeg())
            os.replace(tmp, path)

    def evaluation(self):
        env = self.env_interface.env.habitat_env
        metrics = env.get_metrics()
        tracker = env.task.measurements.measures['task_evaluation_log'].get_metric()
        # The log exposes the same proposition/constraint history used for success.
        return {
            'success': bool(metrics.get('task_state_success', False)),
            'percent_complete': float(metrics.get('task_percent_complete', 0)),
            'satisfied_at': _json_safe(tracker['proposition_satisfied_at']),
            'constraint_satisfaction': _json_safe(tracker['constraint_satisfaction']),
            'current': [bool(p.is_satisfied) for p in tracker['state_sequence'][-1]] if tracker['state_sequence'] else [],
        }

    def _do_snapshot(self):
        result = super()._do_snapshot()
        if not self.loaded:
            return result
        robot = self.env_interface.sim.get_agent_data(0).articulated_agent
        result['robot_pose'] = {'position': list(robot.base_pos), 'yaw': float(robot.base_rot)}
        result['evaluation'] = self.evaluation()
        handles = self.env_interface.sim.ep_info.info.get('variant_spec', {}).get('entity_handles', {})
        graph = self.env_interface.perception.gt_graph
        by_handle = {node.sim_handle: node.name for node in graph.get_all_objects()}
        result['aliases'] = {name: by_handle[handle] for name, handle in handles.items() if handle in by_handle}
        scene_info = json.loads((Path(self.episode_info['data_path']).parent / 'scene_info.json').read_text())
        by_handle = {node.sim_handle: node.name for node in graph.get_all_furnitures()}
        result['aliases'].update({name: by_handle[handle] for name, handle in scene_info['receptacle_to_handle'].items() if handle in by_handle})
        room_names = self.env_interface.perception.region_id_to_name
        for name, region in scene_info['room_to_id'].items():
            if region in room_names:
                result['aliases'][name] = room_names[region]
                result['aliases']['floor_' + name.replace('/', '_')] = 'floor_' + room_names[region]
        return result


def main():
    ipc = Path(sys.argv[1])
    session = None
    for line in sys.stdin:
        command = json.loads(line)
        try:
            if command['op'] == 'load':
                results = Path(command['results_dir'])
                # Every load starts from a clean scratch folder, so a recording holds only its own clips.
                for stale in ('clips', 'videos'):
                    shutil.rmtree(results / stale, ignore_errors=True)
                (results / 'initial.jpg').unlink(missing_ok=True)
                results.mkdir(parents=True, exist_ok=True)
                if session is None:
                    session = GroundTruthSession(results_dir=results)
                    session.ipc_dir = ipc
                else:
                    session.results_dir = results
                    (results / 'videos').mkdir(parents=True, exist_ok=True)
                session.load_id = command.get('load_id')
                session.load_episode(data_path=command['dataset'], episode_id=command['episode_id'])
                if session.get_latest_jpeg():
                    (results / 'initial.jpg').write_bytes(session.get_latest_jpeg())
                result = session.call_on_worker('snapshot')
            elif command['op'] == 'skill':
                command_started = time.perf_counter()
                snapshot = session.call_on_worker('snapshot')
                aliases = snapshot.get('aliases', {})
                def resolve(part):
                    name = part.strip()
                    return name[len('runtime:'):] if name.startswith('runtime:') else aliases.get(name, name)
                target = ','.join(resolve(part) for part in command['target'].split(','))
                action = session.run_skill(skill=command['skill'], agent_index=0, target=target)
                result = session.call_on_worker('snapshot')
                action.setdefault('timing', {})['worker_command_wall_seconds'] = time.perf_counter() - command_started
                result['action_result'] = action
            else:
                raise ValueError('Unknown worker command')
            write_json(ipc / 'response.json', {'id': command['id'], 'result': result})
        except Exception as error:
            traceback.print_exc()
            write_json(ipc / 'response.json', {'id': command['id'], 'error': str(error)})


if __name__ == '__main__':
    main()
