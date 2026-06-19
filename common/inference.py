"""
Inference utilities shared by PINN_training.py and kernels.py: load splits, scale/unscale
with already-fitted StandardScalers, predict with the Keras model, and compute the relative
percentage error. Every function takes model/scaler_i/scaler_o explicitly instead of reading
them from caller-script globals.
"""

import os

import numpy as np

"""
Column order (z, k1, k2, x12, Om0, log10fR0); everything past N_INPUTS is an output
(A, A', B, B'). This order is an invariant shared across the whole pipeline.
"""
N_INPUTS = 6


def loadSplit(path_datasets, name):
    return np.load(os.path.join(path_datasets, f"{name}.npz"))["data"]


def scaleSplit(data_set, scaler_i, scaler_o):
    input_set, output_set = data_set[:, :N_INPUTS], data_set[:, N_INPUTS:]
    return [scaler_i.transform(input_set), scaler_o.transform(output_set)]


def unscale(data, scaler_o):
    return scaler_o.inverse_transform(data)


def makePrediction(model, x_data, batch_size):
    return model.predict(
        x_data,
        batch_size=batch_size,
        verbose=0,
    )


def predictAndUnscale(model, scaled_x, scaler_o, batch_size):
    scaled_y_pred = makePrediction(model, scaled_x, batch_size)
    return unscale(scaled_y_pred, scaler_o)


def relativeErrorPct(y_true, y_pred):
    return (1 - y_pred / y_true) * 100


def percentileReport(error, names=("A", "Ap", "B", "Bp"), q=99):
    separator = "=" * 20
    print(f"Percentil {q} for unscaled data in relative percentual error")
    print(separator)
    for r, name in enumerate(names):
        perc = np.percentile(np.abs(error.T[r]), q)
        print(f"{name:>4}: {perc:.6f}%")
    print(separator)


def emulate(z_arr, args, model, scaler_i, scaler_o, batch_size):
    z_len = len(z_arr)
    n_args = len(args)

    args   = np.tile(args, z_len).reshape(z_len, n_args)
    x_data = np.hstack([z_arr.reshape(z_len, 1), args])

    scaled_x = scaler_i.transform(x_data)
    return predictAndUnscale(model, scaled_x, scaler_o, batch_size)
