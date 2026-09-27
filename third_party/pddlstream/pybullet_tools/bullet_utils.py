"""Stub for pybullet_tools.bullet_utils — pddlstream uses this only for printing."""


def print_action_plan(plan, *args, **kwargs):
    if plan:
        for i, action in enumerate(plan):
            print(f"  {i + 1}. {action}")
