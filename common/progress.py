"""
Shared progress-reporting helper for the three entry scripts (datasets_generator.py,
PINN_training.py, kernels.py), so each one doesn't invent its own print format.
"""

import time
from contextlib import contextmanager


@contextmanager
def stage(label):
    """
    Brackets a pipeline stage with a start/elapsed-time pair of prints instead of progress
    per chunk/epoch/row -- exactly one line before and one after, regardless of how long the
    stage takes, so this never turns into per-iteration log noise.
    """
    print(f"{label}...")
    t0 = time.perf_counter()
    yield
    print(f"{label}: done in {time.perf_counter() - t0:.1f}s")
