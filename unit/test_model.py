"""
Smoke tests de arquitectura para PINN/model.py. Deliberadamente NO se testea train_model ni
save_model (entrenamiento real); solo construcción/compilación de objetos, que es barato.
"""

import tensorflow.keras as K

from emulator.inference import N_INPUTS
from PINN.model import create_model, callbacks


def _small_params():
    return {
        "neurons":      [8, 4],
        "activations":  ["tanh", "linear"],
        "lr":            1e-3,
        "loss":         "mse",
        "early_patience":   5,
        "RLRonP_factor":    0.5,
        "RLRonP_patience":  3,
    }


def test_create_model_architecture():
    params = _small_params()
    model = create_model(params)

    dense_layers = [layer for layer in model.layers if isinstance(layer, K.layers.Dense)]
    assert len(dense_layers) == 2

    assert dense_layers[0].units == 8
    assert dense_layers[0].activation.__name__ == "tanh"
    assert dense_layers[1].units == 4
    assert dense_layers[1].activation.__name__ == "linear"

    assert model.input_shape == (None, N_INPUTS)
    assert model.output_shape == (None, 4)
    assert isinstance(model.optimizer, K.optimizers.Adam)


def test_callbacks_are_configured_from_params():
    params = _small_params()
    early_stopping, reduce_lr = callbacks(params)

    assert isinstance(early_stopping, K.callbacks.EarlyStopping)
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == params["early_patience"]
    assert early_stopping.restore_best_weights is True

    assert isinstance(reduce_lr, K.callbacks.ReduceLROnPlateau)
    assert reduce_lr.monitor == "val_loss"
    assert reduce_lr.factor == params["RLRonP_factor"]
    assert reduce_lr.patience == params["RLRonP_patience"]
    assert reduce_lr.min_lr == 1e-6
