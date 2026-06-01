import os
import sys

# Make phase_3 (under test) and phases 1–2 (reused) importable.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))                   # phase_3/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "phase_2"))  # phase_2/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "phase_1"))  # phase_1/
