"""Incremental episode logs and recovery after wall-clock termination.

Standalone stdlib module so the launcher can recover logs without Habitat/GPU.
"""
import copy
import functools
import json
from pathlib import Path
import signal
import threading
import time


class EpisodeTerminated(BaseException):
    """Stop unwinding cleanly on SIGTERM; not caught by planner retry handlers."""


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
    temporary.replace(path)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, default=json_value) + '\n')


def read_events(path):
    """A killed write can leave an incomplete final line; keep all prior events."""
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class EpisodeProgress:
    def __init__(self, output_dir, name, instruction):
        self.output_dir = Path(output_dir)
        self.name = Path(name).name
        self.journal = self.output_dir / 'planner-log' / f'planner-log-{self.name}.jsonl'
        self.path = self.journal.with_suffix('.json')
        self.actions_path = self.output_dir / 'actions' / f'actions-{self.name}.json'
        self.actions_journal = self.actions_path.with_suffix('.jsonl')
        self.document = {'task': instruction, 'subgoal_instructions': [instruction],
                         'status': 'running', 'complete': False, 'steps': [], 'actions': []}
        self.instruction = instruction
        self.active = {}
        self.sim_steps = 0
        self.last_checkpoint = 0.0
        self.last_text = {}
        self.last_signature = None
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.actions_path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh run, including deliberate reuse of the same episode output name.
        self.journal.write_text('')
        self.actions_journal.write_text('')
        self._append(self.journal, {'event': 'start', 'task': instruction})
        self.flush()

    def _append(self, path, record):
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({**record, 'wall_time': time.time()}, default=json_value) + '\n')
            stream.flush()

    def continue_instruction(self, instruction):
        self.instruction = instruction
        self.document.update(status='running', complete=False)
        if instruction not in self.document['subgoal_instructions']:
            self.document['subgoal_instructions'].append(instruction)
        self._append(self.journal, {'event': 'instruction', 'instruction': instruction})
        self.flush()

    def start_action(self, agent_id, action, action_input):
        key = str(agent_id)
        current = self.active.get(key)
        if current is not None:
            if current['action'] == action and current['action_input'] == action_input:
                return
            current.update(status='superseded', finished_sim_step=self.sim_steps)
            self._append(self.actions_journal, {'event': 'action', 'action': current})
        entry = {'action_id': len(self.document['actions']), 'agent_id': agent_id,
                 'action': action, 'action_input': action_input, 'instruction': self.instruction,
                 'status': 'in_progress', 'response': None, 'started_sim_step': self.sim_steps,
                 'started_at': time.time()}
        self.document['actions'].append(entry)
        self.active[key] = entry
        self._append(self.actions_journal, {'event': 'action', 'action': entry})
        self.flush()

    def finish_action(self, agent_id, response):
        if not response:
            return
        entry = self.active.pop(str(agent_id), None)
        if entry is not None:
            # A returned failure is a finished attempt, not task success.
            entry.update(status='finished', response=response, finished_sim_step=self.sim_steps,
                         finished_at=time.time())
            self._append(self.actions_journal, {'event': 'action', 'action': entry})
            self.flush()

    def simulator_step(self, info):
        self.sim_steps += 1
        checkpoint = {'sim_step_count': self.sim_steps, 'wall_time': time.time(),
                      'stats': {k: info[k] for k in ('task_percent_complete', 'task_state_success', 'task_explanation') if k in info},
                      'active_action_ids': [a['action_id'] for a in self.active.values()]}
        self.document['latest_simulator_checkpoint'] = checkpoint
        atomic_json(self.output_dir / 'planner-log' / f'checkpoint-{self.name}.json', checkpoint)
        # Never rewrite a long planner log on every physics step.

    def record(self, info):
        step = {k: copy.deepcopy(v) for k, v in info.items()
                if k not in ('prompts', 'traces', 'print', 'print_no_tags')}
        step.update(instruction=self.instruction, log_index=len(self.document['steps']))
        self.document['steps'].append(step)
        self._append(self.journal, {'event': 'step', 'step': step})
        for kind, prefix in (('prompts', 'prompt'), ('traces', 'trace')):
            for uid, text in info.get(kind, {}).items():
                key = (kind, uid)
                if self.last_text.get(key) != text:
                    atomic_write(self.output_dir / kind / str(uid) / f'{prefix}-{self.name}-{uid}.txt', text)
                    self.last_text[key] = text
        if 'actions_per_agent' in info:
            plan_text = str(info['actions_per_agent'])
            if self.last_text.get('plan') != plan_text:
                atomic_write(self.output_dir / 'plan' / f'{self.name}.txt', plan_text)
                self.last_text['plan'] = plan_text
        signature = json.dumps({k: info.get(k) for k in ('high_level_actions', 'responses', 'replanned', 'is_done')}, default=json_value, sort_keys=True)
        boundary = signature != self.last_signature
        self.last_signature = signature
        if boundary or time.monotonic() - self.last_checkpoint >= 5:
            self.flush()
        return boundary

    def flush(self):
        self.document['updated_at'] = time.time()
        atomic_json(self.path, self.document)
        atomic_json(self.actions_path, {'task': self.document['task'], 'status': self.document['status'],
                                       'complete': self.document['complete'], 'actions': self.document['actions'],
                                       'sim_step_count': self.sim_steps})
        self.last_checkpoint = time.monotonic()

    def finish(self, status):
        self.document.update(status=status, complete=status == 'completed')
        for entry in self.active.values():
            entry.update(status='incomplete', last_sim_step=self.sim_steps)
            self._append(self.actions_journal, {'event': 'action', 'action': entry})
        self.active.clear()
        self._append(self.journal, {'event': 'end', 'status': status})
        self.flush()


def start_episode_progress(runner, preserve=False):
    progress = getattr(runner, '_episode_progress', None) if preserve else None
    if progress is None:
        progress = EpisodeProgress(runner.output_dir, runner.episode_filename, runner.current_instruction)
        runner._episode_progress = progress
    else:
        progress.continue_instruction(runner.current_instruction)
    runner.env_interface._episode_progress = progress


def checkpoint_episode(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        status = 'interrupted'
        previous = None
        installed = threading.current_thread() is threading.main_thread()
        def terminate(signum, frame):
            raise EpisodeTerminated(f'Episode received signal {signum}')
        if installed:
            previous = signal.signal(signal.SIGTERM, terminate)
        try:
            result = method(self, *args, **kwargs)
            status = 'completed'
            if result.get('combined_time_limit_hit'):
                status = 'time_limit'
            return result
        except EpisodeTerminated:
            status = 'terminated'
            raise
        finally:
            if installed:
                signal.signal(signal.SIGTERM, previous)
            progress = getattr(self.env_interface, '_episode_progress', None)
            if progress is not None:
                try:
                    progress.finish(status)
                finally:
                    self.env_interface._episode_progress = None
            if status != 'completed' and hasattr(self, '_save_object_state_report'):
                try:
                    self._save_object_state_report(status=status)
                except Exception as error:
                    # Initial snapshot remains durable if simulator teardown prevents
                    # a fresh cutoff snapshot. Do not replace it with invented states.
                    import warnings
                    warnings.warn(f'Cutoff object snapshot unavailable: {type(error).__name__}')
    return wrapped


def recover_episode(journal, status='timeout'):
    """Materialize standard logs from the flushed journals after the worker exits."""
    journal = Path(journal)
    name = journal.stem.removeprefix('planner-log-')
    output_dir = journal.parent.parent
    document = {'task': '', 'steps': [], 'actions': [], 'subgoal_instructions': [],
                'status': status, 'complete': False}
    for record in read_events(journal):
        if record['event'] == 'start':
            document['task'] = record['task']
            document['subgoal_instructions'].append(record['task'])
        elif record['event'] == 'instruction':
            if record['instruction'] not in document['subgoal_instructions']:
                document['subgoal_instructions'].append(record['instruction'])
        elif record['event'] == 'step':
            document['steps'].append(record['step'])
    actions_path = output_dir / 'actions' / f'actions-{name}.json'
    actions_journal = actions_path.with_suffix('.jsonl')
    actions = {}
    if actions_journal.exists():
        for record in read_events(actions_journal):
            entry = record['action']
            actions[entry['action_id']] = entry
    checkpoint = output_dir / 'planner-log' / f'checkpoint-{name}.json'
    if checkpoint.exists():
        document['latest_simulator_checkpoint'] = json.loads(checkpoint.read_text())
    for entry in actions.values():
        if entry['status'] == 'in_progress':
            entry.update(status='incomplete', last_sim_step=document.get('latest_simulator_checkpoint', {}).get('sim_step_count'))
    document['actions'] = [actions[key] for key in sorted(actions)]
    document['recovered_at'] = time.time()
    atomic_json(journal.with_suffix('.json'), document)
    atomic_json(actions_path, {'task': document['task'], 'status': status, 'complete': False,
                              'actions': document['actions']})
    return document


def recover_run(output_dir, status='timeout'):
    output_dir = Path(output_dir)
    recovered = []
    for journal in output_dir.glob('*/planner-log/planner-log-*.jsonl'):
        recover_episode(journal, status)
        recovered.append(str(journal.with_suffix('.json')))
    for path in output_dir.glob('*/llm_costs/*.json'):
        report = json.loads(path.read_text())
        report['status'] = status
        atomic_json(path, report)
    return recovered
