"""
Fixtures compartidas entre los tests de unit/. No se mockea CONFIG["N_steps"] aquí porque
AandBfunctions está jiteada y la lectura de CONFIG queda fijada en la primera traza (ver
notas en unit/test_ode.py); ese mock se hace inline, función por función, donde es seguro.
"""

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler


class FakeKerasModel:
    """
    Stub de un modelo Keras: solo implementa .predict() con la misma firma que se usa en
    emulator/inference.py, registrando los argumentos recibidos para poder verificarlos.
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
    StandardScaler de input (6 cols) y de output (4 cols), fiteados sobre datos sintéticos
    pequeños pero con variación real en cada columna (necesario para que el fit no degenere).
    """
    rng = np.random.default_rng(0)
    i_set = rng.uniform(low=0.0, high=1.0, size=(20, 6))
    o_set = rng.uniform(low=-1.0, high=1.0, size=(20, 4))

    scaler_i = StandardScaler().fit(i_set)
    scaler_o = StandardScaler().fit(o_set)

    return scaler_i, scaler_o
