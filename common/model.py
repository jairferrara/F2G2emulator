"""
Builds, trains, and saves the emulator's Keras model. Specific to PINN_training.py:
kernels.py only loads the already-trained model, never builds one.
"""

import os
import json

import joblib

import tensorflow.keras as K

from common.inference import N_INPUTS


"""
Keyed by params["optimizer"] (default "adam", matching today's only option) so future
architecture experiments can swap optimizers without touching createModel.
"""
_OPTIMIZERS = {
    "adam"    : K.optimizers.Adam,
    "sgd"     : K.optimizers.SGD,
    "rmsprop" : K.optimizers.RMSprop,
}


def createModel(params):
    """
    Architecture is entirely dict-driven via params["neurons"]/params["activations"];
    the last entry of each is always the output layer.
    """
    neurons        = params["neurons"]
    activations    = params["activations"]
    lr             = params["lr"]
    loss           = params["loss"]
    dropout        = params.get("dropout")
    batch_norm     = params.get("batch_norm", False)
    optimizer_name = params.get("optimizer", "adam")

    model = K.Sequential()
    model.add(K.Input(shape=(N_INPUTS,)))

    n_layers = len(neurons)
    for i, (n, func) in enumerate(zip(neurons, activations)):
        model.add(K.layers.Dense(n, activation=func))

        is_output_layer = (i == n_layers - 1)
        if not is_output_layer:
            if batch_norm:
                model.add(K.layers.BatchNormalization())
            if dropout:
                rate = dropout[i] if isinstance(dropout, (list, tuple)) else dropout
                model.add(K.layers.Dropout(rate))

    model.compile(
        optimizer=_OPTIMIZERS[optimizer_name](learning_rate=lr),
        loss=loss,
        metrics=params.get("metrics"),
    )

    model.summary()
    return model


def buildCallbacks(params):
    early_stopping = K.callbacks.EarlyStopping(
        monitor               = 'val_loss',
        mode                  = "min",
        patience              = params["early_patience"],
        restore_best_weights  = True,
        verbose               = 1,
    )

    reduce_lr = K.callbacks.ReduceLROnPlateau(
        monitor   = 'val_loss',
        factor    = params["RLRonP_factor"],
        patience  = params["RLRonP_patience"],
        min_lr    = 1e-6,
        verbose   = 1
    )

    return [early_stopping, reduce_lr]


def trainModel(params, scaled_train, scaled_val):
    """
    shuffle=True doesn't break Sobol coverage: it only reorders rows within a set, and
    each set already covers the full sampling space.
    """
    epochs     = params["epochs"]
    batch_size = params["batch_size"]

    x_train, y_train = scaled_train[0], scaled_train[1]
    x_val, y_val     = scaled_val[0], scaled_val[1]
    model = createModel(params)

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_val, y_val),
        verbose=2,
        callbacks=buildCallbacks(params),
    )

    return history, model


def saveModel(params, model, scaler_i, scaler_o):
    path_model = params["path_model"]

    joblib.dump(scaler_i, os.path.join(path_model, "scaler_i.pkl"))
    joblib.dump(scaler_o, os.path.join(path_model, "scaler_o.pkl"))

    model.save(os.path.join(path_model, "model.keras"))

    """
    Saved alongside the model so it's self-describing instead of a black box.
    """
    with open(os.path.join(path_model, "params.json"), "w") as f:
        json.dump(params, f, indent=2)


def loadModel(path_model):
    """
    Companion to saveModel: returns (model, scaler_i, scaler_o, params) in one call.
    """
    scaler_i = joblib.load(os.path.join(path_model, "scaler_i.pkl"))
    scaler_o = joblib.load(os.path.join(path_model, "scaler_o.pkl"))
    model    = K.models.load_model(os.path.join(path_model, "model.keras"))

    with open(os.path.join(path_model, "params.json")) as f:
        params = json.load(f)

    return model, scaler_i, scaler_o, params
