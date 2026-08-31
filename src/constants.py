"""Torch-free shared constants.

MODEL_NAMES lives here (not in src/models.py) so the plotting helpers in
src/evaluation.py stay importable on machines without torch installed —
models.py imports torch at module level and re-exports this list after
asserting it matches MODEL_REGISTRY.
"""

MODEL_NAMES = ['integral', 'baseline', 'truncation', 'meanpool']
