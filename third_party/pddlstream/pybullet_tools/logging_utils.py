"""Stub for pybullet_tools.logging_utils — pddlstream uses this only for printing."""
import json
import os


def myprint(*args, **kwargs):
    print(*args, **kwargs)


def dump_json(data, path, **kwargs):
    """Write data as JSON; used by pddlstream statistics."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass
