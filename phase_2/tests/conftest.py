import os
import sys

# Make both phase_2 (module under test) and phase_1 (golden reference) importable.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))            # phase_2/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "phase_1"))  # phase_1/
