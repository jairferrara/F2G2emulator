import os
os.environ["PYTHONHASHSEED"] = "42"
# os.environ["TF_DETERMINISTIC_OPS"] = "42"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "42"

import random

from sklearn.preprocessing import StandardScaler

import numpy as np

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import matplotlib.pyplot as plt

from emulator.inference import (
    N_INPUTS, load_split, scale_split, unscale, make_prediction,
    relative_error_pct, percentile_report,
)
from PINN.model import train_model, save_model

params = {
            "neurons" : [64, 512, 512, 4],    # The last one correspond to the output layer
        "activations" : ["tanh", "tanh", "tanh", "linear"],    # The last one correspond to the output layer
             "epochs" : 100,
                 "lr" : 5e-3,
               "loss" : "mse",
         "batch_size" : 256,
     "early_patience" : 10,
      "RLRonP_factor" : 0.6,
    "RLRonP_patience" : 7,
      "path_datasets" : "./src/datasets/",
         "path_model" : "./src/model/"
}

# ### Pre-processing data

def loadData():
    """
    Lo que se resuelve en el script numérico es con fR0, no con log10fR0.
    """
    path_datasets = params["path_datasets"]

    train      = load_split(path_datasets, "train")
    validation = load_split(path_datasets, "validation")
    test       = load_split(path_datasets, "test")

    print("Samples size:\n")
    print(f"Train: {len(train)} || val & test: {len(test)}.")

    return train, validation, test

def createScalers(train):
    """
    Se crean los scalers únicamente con train para impedir que el modelo sepa de
    antemano información de validation o test.
    """
    i_set, o_set = train[:, :N_INPUTS], train[:, N_INPUTS:]

    scaler_i = StandardScaler().fit(i_set)
    scaler_o = StandardScaler().fit(o_set)

    return scaler_i, scaler_o

# ### Plotting

def plotLossFunction(history):
    succ_epochs = len(history.history['loss'])
    plt.figure(figsize=(10,7))
    x_axis = np.linspace(1, succ_epochs, succ_epochs)
    plt.semilogy(x_axis, history.history['loss'], 'r--', label='Train')
    plt.semilogy(x_axis, history.history['val_loss'], 'b-', label='Validation')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

def plotRelError(rel_error):
    rows, cols = 2, 2
    names = ["A", "Ap", "B", "Bp"]

    fig, axes = plt.subplots(rows, cols, figsize=(14,10))

    for c in range(cols):
        for r in range(rows):
            axes[r, c].hist(
                rel_error.T[r * 2 + c],
                bins=40,
                #range=(-1, 1),
                log=True,
                orientation="horizontal",
            )
            axes[r, c].set_xlabel("Frecuency", fontsize=18)
            axes[r, c].set_ylabel(r"$\Delta y$%", fontsize=18)
            axes[r, c].tick_params(
                axis="both",
                labelsize=14,
            )
            axes[r, c].set_title(f"Error relativo porcentual de {names[r * 2 + c]}")

    plt.show()

# ### Evaluation

train, validation, test = loadData()
scaler_i, scaler_o = createScalers(train)

scaled_train = scale_split(train,      scaler_i, scaler_o)
scaled_val   = scale_split(validation, scaler_i, scaler_o)

history, model = train_model(params, scaled_train, scaled_val)
plotLossFunction(history)
save_model(params, model, scaler_i, scaler_o)

scaled_x, scaled_y = scale_split(test, scaler_i, scaler_o)
scaled_y_predic     = make_prediction(model, scaled_x, params["batch_size"])

unscaled_y        = unscale(scaled_y, scaler_o)
unscaled_y_predic = unscale(scaled_y_predic, scaler_o)

unscaled_rel_error = relative_error_pct(unscaled_y, unscaled_y_predic)
percentile_report(unscaled_rel_error)

plotRelError(unscaled_rel_error)
