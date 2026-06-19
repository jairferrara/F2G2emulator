"""
Fixtures shared across unit/ tests. CONFIG["N_steps"] isn't mocked here because
computeAAndB is jitted and its CONFIG reads get fixed on the first trace (see the notes
in unit/test_ode.py); that mocking is done inline, function by function, where it's safe.
"""

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler


class FakeKerasModel:
    """
    Stub of a Keras model: only implements .predict() with the same signature used in
    common/inference.py, recording the received arguments so they can be checked.
    """

    def __init__(self, n_outputs=4, fn=None):
        self.n_outputs   = n_outputs
        self.fn          = fn
        self.calls       = []

    def predict(self, x_data, batch_size=None, verbose=None):
        self.calls.append({
            "x_data": np.asarray(x_data).copy(),
            "batch_size": batch_size,
            "verbose": verbose,
        })
        if self.fn is not None:
            return self.fn(np.asarray(x_data))
        return np.zeros((len(x_data), self.n_outputs))


@pytest.fixture
def fake_model():
    return FakeKerasModel()


@pytest.fixture
def synthetic_scalers():
    """
    Input (6 cols) and output (4 cols) StandardScalers, fit on small synthetic data with
    real variation in every column (needed so the fit doesn't degenerate).
    """
    rng = np.random.default_rng(0)
    i_set = rng.uniform(low=0.0, high=1.0, size=(20, 6))
    o_set = rng.uniform(low=-1.0, high=1.0, size=(20, 4))

    scaler_i = StandardScaler().fit(i_set)
    scaler_o = StandardScaler().fit(o_set)

    return scaler_i, scaler_o
