"""
Tests for common/inference.py. No real TensorFlow needed: the functions take
`model`/`scaler_i`/`scaler_o` as explicit (duck-typed) arguments, so unit/conftest.py's
stubs/fixtures (`fake_model`, `synthetic_scalers`) are used instead of a trained Keras model.
"""

import numpy as np

from common import inference


def test_loadSplit_roundtrip(tmp_path):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.savez(tmp_path / "train.npz", data=data)

    loaded = inference.loadSplit(str(tmp_path), "train")

    np.testing.assert_array_equal(loaded, data)


def test_scaleSplit_splits_at_N_INPUTS(synthetic_scalers):
    scaler_i, scaler_o = synthetic_scalers
    rng = np.random.default_rng(1)
    i_set = rng.uniform(0.0, 1.0, size=(5, 6))
    o_set = rng.uniform(-1.0, 1.0, size=(5, 4))
    data_set = np.hstack([i_set, o_set])

    scaled_i, scaled_o = inference.scaleSplit(data_set, scaler_i, scaler_o)

    np.testing.assert_allclose(scaled_i, scaler_i.transform(i_set))
    np.testing.assert_allclose(scaled_o, scaler_o.transform(o_set))


def test_unscale_is_inverse_of_scaler_transform(synthetic_scalers):
    _scaler_i, scaler_o = synthetic_scalers
    rng = np.random.default_rng(2)
    o_set = rng.uniform(-1.0, 1.0, size=(7, 4))

    scaled = scaler_o.transform(o_set)
    unscaled = inference.unscale(scaled, scaler_o)

    np.testing.assert_allclose(unscaled, o_set, rtol=1e-10)


def test_makePrediction_calls_model_predict_with_expected_args(fake_model):
    x_data = np.zeros((3, 6))

    result = inference.makePrediction(fake_model, x_data, batch_size=16)

    assert len(fake_model.calls) == 1
    call = fake_model.calls[0]
    assert call["batch_size"] == 16
    assert call["verbose"] == 0
    np.testing.assert_array_equal(call["x_data"], x_data)
    assert result.shape == (3, fake_model.n_outputs)


def test_relativeErrorPct_zero_when_equal():
    y_true = np.array([[10.0, 20.0, 5.0, 2.0]])
    y_pred = y_true.copy()

    error = inference.relativeErrorPct(y_true, y_pred)

    np.testing.assert_allclose(error, np.zeros_like(y_true))


def test_relativeErrorPct_known_value():
    y_true = np.array([[10.0, 20.0, 5.0, 2.0]])
    y_pred = 0.9 * y_true

    error = inference.relativeErrorPct(y_true, y_pred)

    np.testing.assert_allclose(error, np.full_like(y_true, 10.0))


def test_percentileReport_prints_expected_percentiles(capsys):
    rng = np.random.default_rng(3)
    error = rng.normal(loc=0.0, scale=2.0, size=(200, 4))
    names = ("A", "Ap", "B", "Bp")

    inference.percentileReport(error, names=names, q=99)

    captured = capsys.readouterr().out
    for r, name in enumerate(names):
        expected_perc = np.percentile(np.abs(error.T[r]), 99)
        assert f"{name:>4}: {expected_perc:.6f}%" in captured


def test_emulate_tiles_args_and_unscales_correctly(synthetic_scalers, fake_model):
    scaler_i, scaler_o = synthetic_scalers
    z_arr = np.array([0.1, 0.2, 0.3])
    args  = np.array([0.05, 0.05, 0.0, 0.3, -5.7])

    """The stub model just returns the first 4 columns of the scaled input, so the
    expected result can be reconstructed independently."""
    fake_model.fn = lambda x: x[:, :4]

    result = inference.emulate(z_arr, args, fake_model, scaler_i, scaler_o, batch_size=8)

    n = len(z_arr)
    expected_x = np.hstack([z_arr.reshape(n, 1), np.tile(args, n).reshape(n, len(args))])
    expected_scaled_x = scaler_i.transform(expected_x)
    expected = scaler_o.inverse_transform(expected_scaled_x[:, :4])

    np.testing.assert_allclose(result, expected, rtol=1e-10)
