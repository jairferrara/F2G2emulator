#!/usr/bin/env python
# coding: utf-8

# ### 0. Configuration & Setup

# In[1]:


import os
os.environ["JAX_PLATFORMS"] = "cuda"
# os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time

import math

import jax
import numpy as np
import jax.numpy as jnp
print(jax.devices())
print(jax.default_backend())

from jax import jit, vmap, lax

from scipy.stats.qmc import Sobol, scale

jax.config.update("jax_enable_x64", True)

"""
Rangos posibles de los parámetros:
[0.001, 0.001, -1.0, 0.1, -7.0]
[0.6,   0.6,    1.0, 0.4, -3.0]
"""
CONFIG = {
    "N_train"         : 60000,
    "N_z"             : 64,
    "z_bounds"        : [0, 3],
    "slower"          : [0.001, 0.001, -1.0, 0.1, -7.0],    # [k1, k2, x12, Om0, log10fR0]
    "supper"          : [0.6,   0.6,    1.0, 0.4, -3.0],
    "invH0"           : 2997.92458,    # H_0^-1 in Mpc/h units
    "etaini"          : -4.0,
    "N_steps"         : 2000,    # steps in the rk4 integrator
    "batch_solver"    : 2**14,    # Depends on GPU's memory aviable
    "path_model"      : "./../src/datasets/",
}


# ### 1. Background cosmology functions

# In[2]:


def Omega_m(eta, Om0):
    return 1.0 / (1.0 + (1.0 - Om0) / Om0 * jnp.exp(3.0 * eta))


# In[3]:


def H_func(eta, Om0):
    return jnp.sqrt(Om0 * jnp.exp(-3.0 * eta) + (1.0 - Om0))


# In[4]:


def f1(eta, Om0):
    return 2.0 - 1.5 * Omega_m(eta, Om0)


# In[5]:


def f2(eta, Om0):
    return 1.5 * Omega_m(eta, Om0)


# ### 2. Model-dependent functions

# In[6]:


def mass(eta, Om0, fR0, invH0):
    num = (Om0 * jnp.exp(-3.0 * eta) + 4.0 * (1.0 - Om0)) ** 1.5
    den = Om0 + 4.0 * (1.0 - Om0)

    return (1.0 / invH0) * jnp.sqrt(1.0 / (2.0 * jnp.abs(fR0))) * num / den


# In[7]:


def mu(eta, k, Om0, fR0, invH0):
    m2 = mass(eta, Om0, fR0, invH0) ** 2

    return 1.0 + (1.0 / 3.0) * k**2 / (k**2 + jnp.exp(2.0 * eta) * m2)


# In[8]:


def M2(eta, Om0, fR0, invH0):
    num = (Om0 * jnp.exp(-3.0 * eta) + 4.0 * (1.0 - Om0)) ** 5
    den = (Om0 + 4.0 * (1.0 - Om0)) ** 4

    return (9.0 / 4.0) / invH0**2 * (1.0 / jnp.abs(fR0)) ** 2 * num / den


# ### 3. Source functions

# In[9]:


def sourceA(kf, k1, k2, eta, Om0, fR0, invH0):
    f2v   = f2(eta, Om0)
    mu_kf = mu(eta, kf, Om0, fR0, invH0)
    mu_k1 = mu(eta, k1, Om0, fR0, invH0)
    mu_k2 = mu(eta, k2, Om0, fR0, invH0)

    # sourcea
    sa = f2v * mu_kf

    # sourceFL
    c1 = (kf**2 - k1**2 - k2**2) / (2.0 * k1**2)
    c2 = (kf**2 - k1**2 - k2**2) / (2.0 * k2**2)
    sFL = f2v * ((c1 + c2) * mu_kf - c1 * mu_k2 - c2 * mu_k1)

    # sourcedI
    m2 = mass(eta, Om0, fR0, invH0) ** 2
    omH_over_aH0 = (Omega_m(eta, Om0) * H_func(eta, Om0)) / (jnp.exp(eta) * invH0)
    sdI = (1.0 / 6.0) * omH_over_aH0**2 * (kf**2 * M2(eta, Om0, fR0, invH0)) / (
        (kf**2 * jnp.exp(-2.0 * eta) + m2) *
        (k1**2 * jnp.exp(-2.0 * eta) + m2) *
        (k2**2 * jnp.exp(-2.0 * eta) + m2)
    )

    return sa + sFL - sdI


# In[10]:


def sourceB(kf, k1, k2, eta, Om0, fR0, invH0):
    f2v   = f2(eta, Om0)
    mu_kf = mu(eta, kf, Om0, fR0, invH0)
    mu_k1 = mu(eta, k1, Om0, fR0, invH0)
    mu_k2 = mu(eta, k2, Om0, fR0, invH0)
    return f2v * (mu_k1 + mu_k2 - mu_kf)


# ### 4. ODE -- State vector shape: y = [af, afp, bf, bfp, d1, d1p, d2, d2p]

# In[11]:


def rhs(eta, y, args):
    kf, k1, k2, Om0, fR0, invH0 = args
    af, afp, bf, bfp, d1, d1p, d2, d2p = y

    f1v   = f1(eta, Om0)
    f2v   = f2(eta, Om0)
    mu_kf = mu(eta, kf, Om0, fR0, invH0)
    mu_k1 = mu(eta, k1, Om0, fR0, invH0)
    mu_k2 = mu(eta, k2, Om0, fR0, invH0)

    sA = sourceA(kf, k1, k2, eta, Om0, fR0, invH0)
    sB = sourceB(kf, k1, k2, eta, Om0, fR0, invH0)

    d_af  = afp
    d_afp = -f1v * afp + f2v * mu_kf * af + sA * d1 * d2
    d_bf  = bfp
    d_bfp = -f1v * bfp + f2v * mu_kf * bf + sB * d1 * d2
    d_d1  = d1p
    d_d1p = -f1v * d1p + f2v * mu_k1 * d1
    d_d2  = d2p
    d_d2p = -f1v * d2p + f2v * mu_k2 * d2

    return jnp.array([d_af, d_afp, d_bf, d_bfp, d_d1, d_d1p, d_d2, d_d2p])


# ### 5. Runge-Kutta 4th Order integrator

# In[12]:


def rk4_step(y, eta, dt, args):
    k1 = rhs(eta,            y,                 args)
    k2 = rhs(eta + 0.5 * dt, y + 0.5 * dt * k1, args)
    k3 = rhs(eta + 0.5 * dt, y + 0.5 * dt * k2, args)
    k4 = rhs(eta + dt,       y + dt * k3,       args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# In[13]:


def solve_ode(y0, eta_array, args):
    dt = eta_array[1] - eta_array[0]

    def step_fn(y, eta):
        y_next = rk4_step(y, eta, dt, args)
        return y_next, None    # discard history

    y_final, _ = lax.scan(step_fn, y0, eta_array[:-1])    # raplece the vanilla python for
    return y_final


# ### 6. Initial conditions

# In[14]:


def get_initial_conditions(etaini):
    Dplusi   = jnp.exp(etaini)
    dDplusi  = jnp.exp(etaini)
    D2plusi  = (3.0 / 7.0) * jnp.exp(2.0 * etaini)
    dD2plusi = (6.0 / 7.0) * jnp.exp(2.0 * etaini)

    return jnp.array([D2plusi, dD2plusi, D2plusi, dD2plusi,
                       Dplusi,  dDplusi,  Dplusi,  dDplusi])


# ### 7. Main function

# In[15]:


@jit
def AandBfunctions(dataset_arr):
    invH0   = CONFIG["invH0"]
    etaini  = CONFIG["etaini"]
    N_steps = CONFIG["N_steps"]
    z, k1, k2, x12, Om0, log10fR0 = dataset_arr

    fR0   = 10.0 ** log10fR0
    etaev = -jnp.log(1.0 + z)
    kf    = jnp.sqrt(k1**2 + k2**2 + 2.0 * k1 * k2 * x12)

    eta_array = jnp.linspace(etaini, etaev, N_steps)
    y0        = get_initial_conditions(etaini)
    args      = (kf, k1, k2, Om0, fR0, invH0)

    y_final = solve_ode(y0, eta_array, args)

    af, afp, bf, bfp, d1, d1p, d2, d2p = y_final    # @etaev
    return postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p)


# ### 7b. Postprocessing

# In[16]:


def postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p):
    norm      = (3.0 / 7.0) * d1 * d2
    d_norm    = (3.0 / 7.0) * (d1p * d2 + d1 * d2p)

    Aval      = af / norm
    Aprimeval = afp / norm - af * d_norm / norm**2
    Bval      = bf / norm
    Bprimeval = bfp / norm - bf * d_norm / norm**2

    return jnp.array([Aval, Aprimeval, Bval, Bprimeval])


# ### 8. Solver with vmap

# In[17]:


def AandB_solver(dataset_in):
    """
    vmap exige que el arreglo sea uno de jax
    """
    batch_solver = CONFIG["batch_solver"]

    solver = vmap(AandBfunctions, in_axes=0)

    solutions = []
    for idx in range(0, dataset_in.shape[0], batch_solver):
        batch = jnp.array(dataset_in[idx:idx + batch_solver])
        solutions.append(solver(batch))
    solutions = np.concatenate(solutions, axis=0)

    return jnp.array(solutions)


# ### 9. Output to file

# In[18]:


def write_results(path, train_in, train_out):
    data = np.hstack([train_in, train_out])

    np.savetxt(
        path,
        data,
        header="z k1 k2 x12 Om0 log10fR0 A A' B B'",
        fmt=["%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e"]
    )


# ### 10. Sampling

# In[19]:


def generate_samples():
    """
    Genera 3 dataset de cosmologías: train, validation y test.

    N_train es el valor mínimo del dataset train. Sobol exige una potencia de 2 para asegurar que 
    funcione bien, así que se calcula la potencia de 2 más cercana. 

    El tamaño de validation y test se busca que sea ~20% de train. Con solo potencias de 2 lo mejor 
    que se puede conseguir es 1/4, por lo que el tamaño de estos datasets debe ser 1/4 del de train.

    Para cada cosmología, se extiende el muestreo con N_z número de redshift. La cosmología no cambia,
    solo se replica N_z veces.
    """
    N_train  = CONFIG["N_train"]
    N_z      = CONFIG["N_z"]
    z_bounds = CONFIG["z_bounds"]
    lower    = CONFIG["slower"]
    upper    = CONFIG["supper"]

    N_z        = 2 ** math.ceil(math.log2(N_z))
    N_train    = 2 ** math.ceil(math.log2(N_train))
    N_val_test = int(N_train / 4)
    #N_val_test = N_train

    sampler_z         = Sobol(d=1, scramble=True, seed=67)
    sampler_train_val = Sobol(d=len(lower), scramble=True, seed=42)
    sampler_test      = Sobol(d=len(lower), scramble=True, seed=420)

    unscaled_z          = sampler_z.random(N_z)
    unscaled_train      = sampler_train_val.random(N_train)
    unscaled_validation = sampler_train_val.random(N_val_test)
    unscaled_test       = sampler_test.random(N_val_test)

    z          = scale(unscaled_z,          z_bounds[0], z_bounds[1]).flatten()
    train      = scale(unscaled_train,      lower,       upper)
    validation = scale(unscaled_validation, lower,       upper)
    test       = scale(unscaled_test,       lower,       upper)

    datasets = [train, validation, test]
    for idx, ds in enumerate(datasets):
        z_arr         = np.tile(z, len(ds))[:, None]
        ds            = np.repeat(ds, N_z, axis=0)
        datasets[idx] = np.hstack([z_arr, ds])

    return datasets[0], datasets[1], datasets[2]


# ### 11. Main function

# In[20]:


get_ipython().run_cell_magic('time', '', '\nN_train    = CONFIG["N_train"]\npath_model = CONFIG["path_model"]\n\n### Genera los samples del input\ntrain_in, validation_in, test_in = generate_samples()\nprint(f"Sampled for train (~{N_train}), validation and test.")\n\n### Resuelve el EDP para los output\ntrain_out      = AandB_solver(train_in)\nvalidation_out = AandB_solver(validation_in)\ntest_out       = AandB_solver(test_in)\nprint("EDP solver completed.")\n\n### Escribe el resultado en un .txt\nwrite_results(path_model + "train.txt",      train_in,      train_out)\nwrite_results(path_model + "validation.txt", validation_in, validation_out)\nwrite_results(path_model + "test.txt",       test_in,       test_out)\nprint("Written.\\n")\n')

