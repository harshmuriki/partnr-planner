"""Run the timeout probe for three baseline configs in an allocated GPU job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.recover_episode_logs import recover_run
from scripts.run_parallel import result_metrics


def main():
    root = Path(sys.argv[1]).resolve()
    results = []
    for baseline, sig in [('single_agent_zero_shot_react_summary', signal.SIGTERM),
                          ('single_agent_vlm_tamp_pddl', signal.SIGKILL),
                          ('single_agent_tru_pomdp', signal.SIGTERM)]:
        output = root / baseline
        output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', '-m', 'scripts.probe_timeout_logging',
                   f'--config-name=baselines/{baseline}',
                   'habitat.dataset.data_path=baseline_evaluation_v3/episodes/task_3/t3-acc-base/dataset.json',
                   '+habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/',
                   f'paths.results_dir={output}', f'hydra.run.dir={output}/hydra',
                   'hydra.job.chdir=False', 'evaluation.save_video=False',
                   'evaluation.log_data=False', 'evaluation.generate_prediviz=False', 'device=cpu',
                   'evaluation.agents.agent_0.config.tools.motor_skills.wait.skill_config.sim_freq=1']
        with (output / 'console.log').open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 180
                while not (output / 'timeout_probe_ready.json').exists():
                    if child.poll() is not None:
                        raise AssertionError(f'{baseline} exited early: {output}/console.log')
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'{baseline} failed to reach blocked skill')
                    time.sleep(.2)
                # Baseline is killed inside a skill, before it can return planner_info.
                os.killpg(child.pid, sig)
                child.wait(timeout=15)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
        recovered = recover_run(output, 'timeout')
        assert len(recovered) == 1
        data = json.loads(Path(recovered[0]).read_text())
        assert len(data['actions']) == 2
        assert data['actions'][0]['status'] == 'finished'
        assert data['actions'][0]['response']
        assert data['actions'][1]['status'] == 'incomplete'
        assert data['actions'][1]['response'] is None
        assert data['steps'] and not data['complete']
        assert (output / 'dataset/traces/0/trace-timeout_probe-0.txt').is_file()
        assert (output / 'dataset/detailed_traces/detailed_trace-timeout_probe.pkl').is_file()
        metrics = result_metrics(output)
        assert metrics['metrics_are_partial']
        costs = json.loads((output / 'dataset/llm_costs/timeout_probe.json').read_text())
        assert costs['summary']['api_attempts'] == 0
        assert costs['status'] == 'timeout'
        result = {'baseline': baseline, 'signal': sig.name, 'passed': True,
                  'planner_steps': len(data['steps']), 'actions_saved': len(data['actions']),
                  'last_sim_step': data['latest_simulator_checkpoint']['sim_step_count'],
                  'llm_calls': 0, 'report': recovered[0]}
        results.append(result)
        print(json.dumps(result), flush=True)
    (root / 'validation.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
