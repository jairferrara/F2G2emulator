"""
Construcción, entrenamiento y guardado del modelo Keras del emulador. Específico de
PINN_training.py: kernels.py solo carga el modelo ya entrenado, no lo construye.
"""

import os

import joblib

import tensorflow.keras as K

from emulator.inference import N_INPUTS


def create_model(params):
    """
    La arquitectura de la red está descrita por params["neurons"]/params["activations"].

    Es mutable para realizar muchas pruebas, pero el input y output siempre son los mismos.
    """
    neurons     = params["neurons"]
    activations = params["activations"]
    lr          = params["lr"]
    loss        = params["loss"]

    model = K.Sequential()
    model.add(K.Input(shape=(N_INPUTS,)))
    for n, func in zip(neurons, activations):
        model.add(K.layers.Dense(n, activation=func))

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr),
        loss=loss
    )

    model.summary()
    return model


def callbacks(params):
    ES = K.callbacks.EarlyStopping
    RLRonP = K.callbacks.ReduceLROnPlateau

    early_patience  = params["early_patience"]
    RLRonP_factor   = params["RLRonP_factor"]
    RLRonP_patience = params["RLRonP_patience"]

    call_1 = ES(
        monitor               = 'val_loss',
        mode                  = "min",
        patience              = early_patience,
        restore_best_weights  = True,
        verbose               = 1,
    )

    call_2 = RLRonP(
        monitor   = 'val_loss',
        factor    = RLRonP_factor,
        patience  = RLRonP_patience,
        min_lr    = 1e-6,
        verbose   = 1
    )

    return [call_1, call_2]


def train_model(params, scaled_train, scaled_val):
    """
    shuffle=True no afecta a Sobol ya que solo está cambiando el orden de cada uno de los sets,
    pero cada set ya está cubriendo todo el espacio.
    """
    epochs     = params["epochs"]
    batch_size = params["batch_size"]

    x_train, y_train = scaled_train[0], scaled_train[1]
    x_val, y_val     = scaled_val[0], scaled_val[1]
    model = create_model(params)

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_val, y_val),
        verbose=2,
        callbacks=callbacks(params),
    )

    return history, model


def save_model(params, model, scaler_i, scaler_o):
    path_model = params["path_model"]

    joblib.dump(scaler_i, os.path.join(path_model, "scaler_i.pkl"))
    joblib.dump(scaler_o, os.path.join(path_model, "scaler_o.pkl"))

    model.save(os.path.join(path_model, "model.keras"))
