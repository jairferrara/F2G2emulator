#!/usr/bin/env python
# coding: utf-8

# ### Configuration

# In[1]:


import time

import sklearn

import numpy as np

import jax
import jax.numpy as jnp

import tensorflow as tf
import tensorflow.keras as K

import matplotlib.pyplot as plt


# In[2]:


params = {
    "neurons": [128, 128, 128, 4],
    "activations": ["tanh", "tanh", "tanh", "linear"],
    "epochs": 100,
    "lr": 5e-4,
    "loss": "mse",
    "batch_size": 128,
    "early_patience": 10,
    "RLRonP_factor": 0.65,
    "RLRonP_patience": 7,
}


# ### Pre-processing data

# In[3]:


def loadData():
    return np.loadtxt("./AandB_output.txt", skiprows=1)


# In[4]:


def splitData(data):
    """
    Tres datasets con una dist. de los datos 60/20/20 de la cantidad original.

    Sobol no es compatible el shuffleo de los datos, por eso se separan de esta manera.
    Se generó un solo archivo con todos los datos para no importar 3 distintos.
    """
    block = int(data.shape[0] * 0.2)
    train      = data[           : -2 * block]
    validation = data[-2 * block : -block    ]
    test       = data[-block     :           ]
    return train, validation, test


# In[5]:


def _createScaler():
    """
    Se crea un Scaler global para toda la instancia y se impiden problemas con la aleatoriedad.
    """
    global SSi, SSo
    SSi = sklearn.preprocessing.StandardScaler()
    SSo = sklearn.preprocessing.StandardScaler()
    return 0


# In[6]:


def scaleData(data_set):
    """
    Divide los 6 imputs y 4 outputs ded cada data set.

    Se crea 2 SS y 2 scalers para input y output. Los scalers solo se crean con train
    para impedir que el modelo sepa de antemano información de validation o test.
    """
    i_set, o_set = data_set[:,:6], data_set[:,6:]    # (z, k1, k2, x12, om, logf), (A, Ap, B, Bp)

    if "SSi" and "SSo" not in globals():
        _createScaler()

        global scaler_i, scaler_o
        scaler_i = SSi.fit(i_set)
        scaler_o = SSo.fit(o_set)

    return [scaler_i.transform(i_set), scaler_o.transform(o_set)]


# ### Model & training

# In[7]:


def _createModel():
    """
    La arquitectura de la red está descrita por el dict arqui. 

    Es mutables para realizar muchas pruebas, pero el input y output siempre son los mismo.

    Claude recomienda una aquitectura densa de: 6->128->128->4
    """
    neurons     = params["neurons"]
    activations = params["activations"]
    lr          = params["lr"]
    loss        = params["loss"]

    model = K.Sequential()
    model.add(K.Input(shape=(6,)))
    for n, func in zip(neurons, activations):
        model.add(K.layers.Dense(n, activation=func))

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=lr),
        loss=loss
    )

    model.summary()
    return model


# In[8]:


def _callbacks():
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


# In[9]:


def trainModel(scaled_train, scaled_val):
    """
    shuffle=True no afecta a Sobol ya que solo está cambiando el orden de cada uno de los sets,
    pero cada set ya está cubriendo todo el espacio.
    """
    epochs     = params["epochs"]
    batch_size = params["batch_size"]

    x_train, y_train = scaled_train[0], scaled_train[1]
    x_val, y_val     = scaled_val[0], scaled_val[1]
    model = _createModel()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(x_val, y_val),
        verbose=2,
        callbacks=_callbacks(),
    )

    return history, model


# ### Post-processing data

# In[10]:


def unScaleData(data):
    """
    Aplica el escalamiento inverso para obtener los valores en los rangos originales.
    """        
    return scaler_o.inverse_transform(data)


# ### Data validation

# In[11]:


def _makePrediction(model, x_data):
    batch_size = params["batch_size"]

    prediction = model.predict(
        x_data,
        batch_size=batch_size,
        verbose=1,
    )

    return prediction


# In[12]:


def relError(model, scaled_data):
    scaled_x, scaled_y = scaled_data[0], scaled_data[1]
    scaled_y_predic    = _makePrediction(model, scaled_x)

    unscaled_y        = unScaleData(scaled_y)
    unscaled_y_predic = unScaleData(scaled_y_predic)

    scaled_rel_error   = (scaled_y - scaled_y_predic) / scaled_y * 100
    unscaled_rel_error = (unscaled_y - unscaled_y_predic) / unscaled_y * 100

    return [scaled_rel_error, unscaled_rel_error]


# In[13]:


def calcPercentil(rel_error):
    rows = 4
    r_names = ["A", "Ap", "B", "Bp"]

    print(f"Percentil 99 for unscaled data in relative error")
    print(20 * "=")
    for r in range(rows):
        perc = np.percentile(rel_error[1].T[r], 99)
        print(f"{r_names[r]}: {perc}")
    print(20 * "=")


# ### Plotting

# In[14]:


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


# In[24]:


def plotRelError(rel_error):
    rows, cols = 2, 2
    names = ["A", "Aprime", "B", "Bprime"]

    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))

    for c in range(cols):
        for r in range(rows):
            axes[r, c].hist(
                rel_error[1].T[r * 2 + c],
                bins=40,
                #range=(-2, 2),
                log=True,
                orientation="horizontal",
            )
            axes[r, c].set_xlabel("Frecuency", fontsize=18)
            axes[r, c].set_ylabel(r"$\Delta y / y$%", fontsize=18)
            axes[r, c].tick_params(
                axis="both",
                labelsize=14,
            )
            axes[r, c].set_title(f"Error relativo porcentual de {names[r * 2 + c]}")

    plt.show()


# ### Evaluation

# In[16]:


data = loadData()
train, validation, test = splitData(data)


# In[17]:


get_ipython().run_cell_magic('time', '', 'scaled_train = scaleData(train)\nscaled_val = scaleData(validation)\n\nhistory, model = trainModel(scaled_train, scaled_val)\nplotLossFunction(history)\n')


# In[25]:


get_ipython().run_cell_magic('time', '', 'scaled_test = scaleData(test)\n\nrel_error = relError(model, scaled_test)\nplotRelError(rel_error)\n')


# In[26]:


calcPercentil(rel_error)


# In[ ]:




