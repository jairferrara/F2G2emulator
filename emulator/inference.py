"""
Utilidades de inferencia compartidas por PINN_training.py y kernels.py: cargar splits,
escalar/desescalar con los StandardScaler ya entrenados, predecir con el modelo Keras y
calcular el error relativo porcentual. Todas las funciones reciben model/scaler_i/scaler_o
explícitamente en vez de leerlos de variables globales del script que las llama.
"""

import os

import numpy as np

N_INPUTS = 6    # (z, k1, k2, x12, Om0, log10fR0) -> el resto son outputs (A, A', B, B')


def load_split(path_datasets, name):
    return np.load(os.path.join(path_datasets, f"{name}.npz"))["data"]


def scale_split(data_set, scaler_i, scaler_o):
    """
    Divide los N_INPUTS inputs y el resto de outputs de cada data set.
    """
    i_set, o_set = data_set[:, :N_INPUTS], data_set[:, N_INPUTS:]
    return [scaler_i.transform(i_set), scaler_o.transform(o_set)]


def unscale(data, scaler_o):
    """
    Aplica el escalamiento inverso para obtener los valores en los rangos originales.
    """
    return scaler_o.inverse_transform(data)


def make_prediction(model, x_data, batch_size):
    return model.predict(
        x_data,
        batch_size=batch_size,
        verbose=0,
    )


def relative_error_pct(y_true, y_pred):
    return (1 - y_pred / y_true) * 100


def percentile_report(error, names=("A", "Ap", "B", "Bp"), q=99):
    print(f"Percentil {q} for unscaled data in relative percentual error")
    print(20 * "=")
    for r, name in enumerate(names):
        perc = np.percentile(np.abs(error.T[r]), q)
        print(f"{name:>4}: {perc:.6f}%")
    print(20 * "=")


def emulate(z_arr, args, model, scaler_i, scaler_o, batch_size):
    z_len = len(z_arr)
    n_args = len(args)

    args   = np.tile(args, z_len).reshape(z_len, n_args)
    x_data = np.hstack([z_arr.reshape(z_len, 1), args])

    scaled_x = scaler_i.transform(x_data)
    scaled_y_predic = make_prediction(model, scaled_x, batch_size)

    return unscale(scaled_y_predic, scaler_o)
