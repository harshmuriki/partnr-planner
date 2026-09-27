
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_THIRD_PARTY = os.path.join(_ROOT, "third_party", "pddlstream")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
