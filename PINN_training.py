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

from common.progress import stage
from common.inference import (
    N_INPUTS, loadSplit, scaleSplit, unscale, predictAndUnscale,
    relativeErrorPct, percentileReport,
)
from common.model import trainModel, saveModel

"""
Last entry of "neurons"/"activations" is the output layer.
"""
params = {
            "neurons" : [64, 512, 512, 4],
        "activations" : ["tanh", "tanh", "tanh", "linear"],
             "epochs" : 100,
                 "lr" : 5e-3,
               "loss" : "mse",
         "batch_size" : 1024,
     "early_patience" : 10,
      "RLRonP_factor" : 0.6,
    "RLRonP_patience" : 7,
      "path_datasets" : "./src/datasets/",
         "path_model" : "./src/model/"
}

def loadData():
    path_datasets = params["path_datasets"]

    train      = loadSplit(path_datasets, "train")
    validation = loadSplit(path_datasets, "validation")
    test       = loadSplit(path_datasets, "test")

    print("Samples size:\n")
    print(f"Train: {len(train)} || val & test: {len(test)}.")

    return train, validation, test

def createScalers(train):
    """
    Scalers are fit only on train to prevent leaking validation/test information into
    the model beforehand.
    """
    input_set, output_set = train[:, :N_INPUTS], train[:, N_INPUTS:]

    scaler_i = StandardScaler().fit(input_set)
    scaler_o = StandardScaler().fit(output_set)

    return scaler_i, scaler_o

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

with stage("Loading datasets"):
    train, validation, test = loadData()
scaler_i, scaler_o = createScalers(train)

scaled_train = scaleSplit(train,      scaler_i, scaler_o)
scaled_val   = scaleSplit(validation, scaler_i, scaler_o)

with stage("Training"):
    history, model = trainModel(params, scaled_train, scaled_val)
plotLossFunction(history)

saveModel(params, model, scaler_i, scaler_o)
print(f"Model saved to {params['path_model']}")

with stage("Evaluating on test split"):
    scaled_x, scaled_y = scaleSplit(test, scaler_i, scaler_o)
    unscaled_y         = unscale(scaled_y, scaler_o)
    unscaled_y_predic   = predictAndUnscale(model, scaled_x, scaler_o, params["batch_size"])

unscaled_rel_error = relativeErrorPct(unscaled_y, unscaled_y_predic)
percentileReport(unscaled_rel_error)

plotRelError(unscaled_rel_error)
