"""Recover flushed episode logs after termination, without importing Habitat.

python scripts/recover_episode_logs.py RUN_OUTPUT --status timeout
"""
import argparse
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    'episode_progress_recovery',
    Path(__file__).resolve().parents[1] / 'habitat_llm/utils/episode_progress.py',
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
recover_run = _MODULE.recover_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--status', default='timeout')
    args = parser.parse_args()
    for path in recover_run(args.output, args.status):
        print(path)
