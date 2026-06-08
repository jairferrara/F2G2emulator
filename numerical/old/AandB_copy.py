#!/usr/bin/env python
# coding: utf-8

# ### 0. Configuration & Setup

# In[16]:


#import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time

import jax
import numpy as np
import jax.numpy as jnp

from jax import jit, vmap, lax

from scipy.stats.qmc import Sobol, scale

jax.config.update("jax_enable_x64", True)

CONFIG = {
    "N_samples"  : 500,
    "slower"     : [0.0, 0.001, 0.001, -1.0, 0.1, -7.0],    # [z, k1, k2, x12, om, log10fR0]
    "supper"     : [3.0, 0.6,   0.6,    1.0, 0.5, -3.0],
    "invH0"      : 2997.92458,    # H_0^-1 in Mpc/h units
    "etaini"     : -4.0,
    "N_steps"    : 2000,    # steps in the rk4 integrator
    "output_file": "AandB_output.txt",
}


# ### 1. Background cosmology functions

# In[2]:


def OmM(eta, om):
    return 1.0 / (1.0 + (1.0 - om) / om * jnp.exp(3.0 * eta))

def H_func(eta, om):
    return jnp.sqrt(om * jnp.exp(-3.0 * eta) + (1.0 - om))

def f1(eta, om):
    return 2.0 - 1.5 * OmM(eta, om)

def f2(eta, om):
    return 1.5 * OmM(eta, om)


# ### 2. Model-dependent functions

# In[3]:


def mass(eta, om, fR0, invH0):
    num = (om * jnp.exp(-3.0 * eta) + 4.0 * (1.0 - om)) ** 1.5
    den = om + 4.0 * (1.0 - om)
    return (1.0 / invH0) * jnp.sqrt(1.0 / (2.0 * jnp.abs(fR0))) * num / den

def mu_func(eta, k, om, fR0, invH0):
    m2 = mass(eta, om, fR0, invH0) ** 2
    return 1.0 + (1.0 / 3.0) * k**2 / (k**2 + jnp.exp(2.0 * eta) * m2)

def M2_func(eta, om, fR0, invH0):
    num = (om * jnp.exp(-3.0 * eta) + 4.0 * (1.0 - om)) ** 5
    den = (om + 4.0 * (1.0 - om)) ** 4
    return (9.0 / 4.0) / invH0**2 * (1.0 / jnp.abs(fR0)) ** 2 * num / den


# ### 3. Source functions

# In[4]:


def sourceA(kf, k1, k2, eta, om, fR0, invH0):
    f2v   = f2(eta, om)
    mu_kf = mu_func(eta, kf, om, fR0, invH0)
    mu_k1 = mu_func(eta, k1, om, fR0, invH0)
    mu_k2 = mu_func(eta, k2, om, fR0, invH0)

    # sourcea
    sa = f2v * mu_kf

    # sourceFL
    c1 = (kf**2 - k1**2 - k2**2) / (2.0 * k1**2)
    c2 = (kf**2 - k1**2 - k2**2) / (2.0 * k2**2)
    sFL = f2v * ((c1 + c2) * mu_kf - c1 * mu_k2 - c2 * mu_k1)

    # sourcedI
    m2 = mass(eta, om, fR0, invH0) ** 2
    omH_over_aH0 = (OmM(eta, om) * H_func(eta, om)) / (jnp.exp(eta) * invH0)
    sdI = (1.0 / 6.0) * omH_over_aH0**2 * (kf**2 * M2_func(eta, om, fR0, invH0)) / (
        (kf**2 * jnp.exp(-2.0 * eta) + m2) *
        (k1**2 * jnp.exp(-2.0 * eta) + m2) *
        (k2**2 * jnp.exp(-2.0 * eta) + m2)
    )

    return sa + sFL - sdI

def sourceB(kf, k1, k2, eta, om, fR0, invH0):
    f2v   = f2(eta, om)
    mu_kf = mu_func(eta, kf, om, fR0, invH0)
    mu_k1 = mu_func(eta, k1, om, fR0, invH0)
    mu_k2 = mu_func(eta, k2, om, fR0, invH0)
    return f2v * (mu_k1 + mu_k2 - mu_kf)


# ### 4. ODE -- State vector shape: y = [af, afp, bf, bfp, d1, d1p, d2, d2p]

# In[5]:


def rhs(eta, y, args):
    kf, k1, k2, om, fR0, invH0 = args
    af, afp, bf, bfp, d1, d1p, d2, d2p = y

    f1v   = f1(eta, om)
    f2v   = f2(eta, om)
    mu_kf = mu_func(eta, kf, om, fR0, invH0)
    mu_k1 = mu_func(eta, k1, om, fR0, invH0)
    mu_k2 = mu_func(eta, k2, om, fR0, invH0)

    sA = sourceA(kf, k1, k2, eta, om, fR0, invH0)
    sB = sourceB(kf, k1, k2, eta, om, fR0, invH0)

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

# In[6]:


def rk4_step(y, eta, dt, args):
    k1 = rhs(eta,            y,                 args)
    k2 = rhs(eta + 0.5 * dt, y + 0.5 * dt * k1, args)
    k3 = rhs(eta + 0.5 * dt, y + 0.5 * dt * k2, args)
    k4 = rhs(eta + dt,       y + dt * k3,       args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def solve_ode(y0, eta_array, args):
    dt = eta_array[1] - eta_array[0]
    def step_fn(y, eta):
        y_next = rk4_step(y, eta, dt, args)
        return y_next, None    # discard history
    y_final, _ = lax.scan(step_fn, y0, eta_array[:-1])    # raplece the vanilla python for
    return y_final


# ### 6. Initial conditions

# In[7]:


def get_initial_conditions(etaini):
    Dplusi   = jnp.exp(etaini)
    dDplusi  = jnp.exp(etaini)
    D2plusi  = (3.0 / 7.0) * jnp.exp(2.0 * etaini)
    dD2plusi = (6.0 / 7.0) * jnp.exp(2.0 * etaini)
    # [af, afp, bf, bfp, d1, d1p, d2, d2p]
    return jnp.array([D2plusi, dD2plusi, D2plusi, dD2plusi,
                       Dplusi,  dDplusi,  Dplusi,  dDplusi])


# ### 7. Main function -- AandBfunctions

# In[8]:


@jit
def AandBfunctions(z, k1, k2, x12, om, log10fR0):
    invH0   = CONFIG["invH0"]
    etaini  = CONFIG["etaini"]
    N_steps = CONFIG["N_steps"]

    fR0   = 10.0 ** log10fR0
    etaev = -jnp.log(1.0 + z)
    kf    = jnp.sqrt(k1**2 + k2**2 + 2.0 * k1 * k2 * x12)

    eta_array = jnp.linspace(etaini, etaev, N_steps)
    y0        = get_initial_conditions(etaini)
    args      = (kf, k1, k2, om, fR0, invH0)

    y_final = solve_ode(y0, eta_array, args)

    af, afp, bf, bfp, d1, d1p, d2, d2p = y_final    # @etaev
    return postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p)


# ### 7b. Postprocessing

# In[9]:


def postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p):
    norm      = (3.0 / 7.0) * d1 * d2
    d_norm    = (3.0 / 7.0) * (d1p * d2 + d1 * d2p)

    Aval      = af / norm
    Aprimeval = afp / norm - af * d_norm / norm**2
    Bval      = bf / norm
    Bprimeval = bfp / norm - bf * d_norm / norm**2

    return jnp.array([Aval, Aprimeval, Bval, Bprimeval])


# ### 8. Batch evaluation with vmap

# In[10]:


def batch_AandB(z_arr, k1_arr, k2_arr, x12_arr, om_arr, log10fR0_arr):
    batched_fn = vmap(AandBfunctions, in_axes=(0, 0, 0, 0, 0, 0))
    return batched_fn(z_arr, k1_arr, k2_arr, x12_arr, om_arr, log10fR0_arr)


# ### 9. Output to file

# In[11]:


def write_results(filename, params, results):
    """
    params  : dict with input parameter arrays
    results : (N, 4) array of [A, A', B, B']
    Creates or overwrites the file.
    """
    results_np = np.array(results)

    data = np.column_stack([params["z"], params["k1"], params["k2"], params["x12"],
                            params["om"], params["log10fR0"], results_np])

    np.savetxt(
        filename,
        data,
        header="z k1 k2 x12 om log10fR0 A A' B B'",
        fmt=["%.6e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e", "%.12e"],
    )
    print(f"Results written to: {filename}")


# ### 10. Sampling

# In[12]:


def generate_samples(n_points):
    sampler = Sobol(d=6, scramble=True)
    N = int(2 ** jnp.ceil(jnp.log2(n_points)))
    unscaled_samples = sampler.random(N)

    lower = CONFIG["slower"]
    upper = CONFIG["supper"]

    samples = scale(unscaled_samples, lower, upper)
    return jnp.array(samples).T


# ### 11. Main function

# In[13]:


def main(CONFIG):
    N = int(2 ** jnp.ceil(jnp.log2(CONFIG["N_samples"])))
    print(f"Sampling size: {N}\n")

    t0 = time.time()
    z_arr, k1_arr, k2_arr, x12_arr, om_arr, log10fR0_arr = generate_samples(CONFIG["N_samples"])
    t1 = time.time()
    print(f"\nSampling time: {t1 - t0:.6f} s\n")

    t0 = time.time()
    batch_results = batch_AandB(z_arr, k1_arr, k2_arr, x12_arr, om_arr, log10fR0_arr)
    batch_results.block_until_ready()
    t1 = time.time()
    print(f"Batch compilation + run: {t1 - t0:.4f} s\n")

    t0 = time.time()
    params = {
            "z"        : np.array(z_arr),
            "k1"       : np.array(k1_arr),
            "k2"       : np.array(k2_arr),
            "x12"      : np.array(x12_arr),
            "om"       : np.array(om_arr),
            "log10fR0" : np.array(log10fR0_arr),
        }
    write_results(CONFIG["output_file"], params, batch_results)
    t1 = time.time()
    print(f"Writing: {t1 - t0:.4f} s")


# ## 12. Main routine

# In[21]:


if __name__ == "__main__":
    main(CONFIG)


# In[ ]:




