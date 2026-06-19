"""
Architecture smoke tests for common/model.py. Deliberately does NOT test trainModel (real
training); object construction/compilation and save/load metadata are cheap enough to test
directly.
"""

import json

import numpy as np
import tensorflow.keras as K

from common.inference import N_INPUTS
from common.model import createModel, buildCallbacks, saveModel, loadModel


def _smallParams():
    return {
        "neurons":      [8, 4],
        "activations":  ["tanh", "linear"],
        "lr":            1e-3,
        "loss":         "mse",
        "early_patience":   5,
        "RLRonP_factor":    0.5,
        "RLRonP_patience":  3,
    }


def test_createModel_architecture():
    params = _smallParams()
    model = createModel(params)

    dense_layers = [layer for layer in model.layers if isinstance(layer, K.layers.Dense)]
    assert len(dense_layers) == 2

    assert dense_layers[0].units == 8
    assert dense_layers[0].activation.__name__ == "tanh"
    assert dense_layers[1].units == 4
    assert dense_layers[1].activation.__name__ == "linear"

    assert model.input_shape == (None, N_INPUTS)
    assert model.output_shape == (None, 4)
    assert isinstance(model.optimizer, K.optimizers.Adam)


def test_createModel_supports_alternate_optimizer():
    params = {**_smallParams(), "optimizer": "sgd"}
    model = createModel(params)

    assert isinstance(model.optimizer, K.optimizers.SGD)


def test_createModel_dropout_inserted_only_after_hidden_layers():
    params = {**_smallParams(), "dropout": 0.2}
    model = createModel(params)

    dropout_layers = [layer for layer in model.layers if isinstance(layer, K.layers.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].rate == 0.2

    assert isinstance(model.layers[-1], K.layers.Dense)


def test_createModel_batch_norm_inserted_only_after_hidden_layers():
    params = {**_smallParams(), "batch_norm": True}
    model = createModel(params)

    bn_layers = [layer for layer in model.layers if isinstance(layer, K.layers.BatchNormalization)]
    assert len(bn_layers) == 1

    assert isinstance(model.layers[-1], K.layers.Dense)


def test_createModel_metrics_passed_through():
    params = {**_smallParams(), "metrics": ["mae"]}
    model = createModel(params)

    assert model.get_compile_config()["metrics"] == ["mae"]


def test_buildCallbacks_are_configured_from_params():
    params = _smallParams()
    early_stopping, reduce_lr = buildCallbacks(params)

    assert isinstance(early_stopping, K.callbacks.EarlyStopping)
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == params["early_patience"]
    assert early_stopping.restore_best_weights is True

    assert isinstance(reduce_lr, K.callbacks.ReduceLROnPlateau)
    assert reduce_lr.monitor == "val_loss"
    assert reduce_lr.factor == params["RLRonP_factor"]
    assert reduce_lr.patience == params["RLRonP_patience"]
    assert reduce_lr.min_lr == 1e-6


def test_saveModel_writes_params_json(tmp_path, synthetic_scalers):
    scaler_i, scaler_o = synthetic_scalers
    params = {**_smallParams(), "path_model": str(tmp_path)}
    model = createModel(params)

    saveModel(params, model, scaler_i, scaler_o)

    with open(tmp_path / "params.json") as f:
        saved_params = json.load(f)
    assert saved_params == params


def test_loadModel_roundtrips_model_scalers_and_params(tmp_path, synthetic_scalers):
    scaler_i, scaler_o = synthetic_scalers
    params = {**_smallParams(), "path_model": str(tmp_path)}
    model = createModel(params)
    saveModel(params, model, scaler_i, scaler_o)

    loaded_model, loaded_scaler_i, loaded_scaler_o, loaded_params = loadModel(str(tmp_path))

    assert loaded_params == params

    for original_w, loaded_w in zip(model.get_weights(), loaded_model.get_weights()):
        np.testing.assert_array_equal(original_w, loaded_w)

    sample = np.zeros((1, scaler_i.n_features_in_))
    np.testing.assert_allclose(
        loaded_scaler_i.transform(sample), scaler_i.transform(sample)
    )
    sample_o = np.zeros((1, scaler_o.n_features_in_))
    np.testing.assert_allclose(
        loaded_scaler_o.inverse_transform(sample_o), scaler_o.inverse_transform(sample_o)
    )
