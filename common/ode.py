"""
Numerical core of the F2G2 emulator: background cosmology, f(R) model functions, source
terms, RK4 solver for (A, A', B, B'), Sobol sampling of cosmologies, and the analytic
F2(z)/G2(z) combination (calKernels), also reused by kernels.py.
"""

import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import random
from functools import partial
from math import ceil, log2

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax, config, devices, default_backend
from scipy.stats.qmc import Sobol, scale


def resolveJaxPlatform(env=os.environ, default="cpu"):
    """
    Lets JAX_PLATFORMS be set externally (e.g. JAX_PLATFORMS=cuda) for a GPU run without
    editing this file; defaults to "cpu" to match today's no-GPU development machine.
    """
    return env.get("JAX_PLATFORMS", default)

def resolveX64Enabled(env=os.environ, default=True):
    """
    F2G2_JAX_X64=0 is an unvalidated speed experiment trading RK4 numerical stability
    for float32 speed; defaults to True (today's validated behavior).
    """
    return env.get("F2G2_JAX_X64", str(default)) not in ("0", "false", "False")

os.environ["JAX_PLATFORMS"] = resolveJaxPlatform()

"""
jax_enable_x64 must be set before any JAX computation runs, e.g. devices().
"""
config.update("jax_enable_x64", resolveX64Enabled())

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def reportJaxBackend():
    """
    Explicit call (not a module-import side effect) so this only prints when an entry
    script actually starts, not on every `import common.ode` during pytest collection.
    """
    print(f"JAX backend: {default_backend()} (devices: {devices()})")

"""
"slower"/"supper" bound the (k1, k2, x12, Om0, log10fR0) sampling domain valid for the
emulator. "invH0" is in Mpc/h. "N_steps" is the RK4 integrator resolution and
"batch_solver" the vmap chunk size, limited by GPU memory.
"""
CONFIG = {
    "N_train"         : 4000,
    "N_z"             : 64,
    "z_bounds"        : [0, 3],
    "slower"          : [0.001, 0.001, -1.0, 0.1, -7.0],
    "supper"          : [0.6,   0.6,    1.0, 0.4, -3.0],
    "invH0"           : 2997.92458,
    "etaini"          : -4.0,
    "N_steps"         : 2000,
    "batch_solver"    : 2**14,
    "path_datasets"   : "./src/datasets/",
    "seed_z"          : 67,
    "seed_train_val"  : 42,
    "seed_test"       : 420,
}


def omegaM(eta, Om0):
    return 1.0 / (1.0 + (1.0 - Om0) / Om0 * jnp.exp(3.0 * eta))

def hubble(eta, Om0):
    return jnp.sqrt(Om0 * jnp.exp(-3.0 * eta) + (1.0 - Om0))

def f1(eta, Om0):
    return 2.0 - 1.5 * omegaM(eta, Om0)

def f2(eta, Om0):
    return 1.5 * omegaM(eta, Om0)

def scalaronMass(eta, Om0, fR0, invH0):
    num = (Om0 * jnp.exp(-3.0 * eta) + 4.0 * (1.0 - Om0)) ** 1.5
    den = Om0 + 4.0 * (1.0 - Om0)

    return (1.0 / invH0) * jnp.sqrt(1.0 / (2.0 * jnp.abs(fR0))) * num / den

def mu(eta, k, Om0, fR0, invH0):
    mass2 = scalaronMass(eta, Om0, fR0, invH0) ** 2

    return 1.0 + (1.0 / 3.0) * k**2 / (k**2 + jnp.exp(2.0 * eta) * mass2)

def m2(eta, Om0, fR0, invH0):
    num = (Om0 * jnp.exp(-3.0 * eta) + 4.0 * (1.0 - Om0)) ** 5
    den = (Om0 + 4.0 * (1.0 - Om0)) ** 4

    return (9.0 / 4.0) / invH0**2 * (1.0 / jnp.abs(fR0)) ** 2 * num / den

def sourceA(kf, k1, k2, eta, Om0, fR0, invH0, f2v, mu_kf, mu_k1, mu_k2):
    sa = f2v * mu_kf

    c1 = (kf**2 - k1**2 - k2**2) / (2.0 * k1**2)
    c2 = (kf**2 - k1**2 - k2**2) / (2.0 * k2**2)
    sFL = f2v * ((c1 + c2) * mu_kf - c1 * mu_k2 - c2 * mu_k1)

    mass2 = scalaronMass(eta, Om0, fR0, invH0) ** 2
    omH_over_aH0 = (omegaM(eta, Om0) * hubble(eta, Om0)) / (jnp.exp(eta) * invH0)
    sdI = (1.0 / 6.0) * omH_over_aH0**2 * (kf**2 * m2(eta, Om0, fR0, invH0)) / (
        (kf**2 * jnp.exp(-2.0 * eta) + mass2) *
        (k1**2 * jnp.exp(-2.0 * eta) + mass2) *
        (k2**2 * jnp.exp(-2.0 * eta) + mass2)
    )

    return sa + sFL - sdI

def sourceB(f2v, mu_kf, mu_k1, mu_k2):
    return f2v * (mu_k1 + mu_k2 - mu_kf)

"""
State vector layout for the ODE: y = [af, afp, bf, bfp, d1, d1p, d2, d2p].
"""
def rhs(eta, y, kf, k1, k2, Om0, fR0, invH0):
    af, afp, bf, bfp, d1, d1p, d2, d2p = y

    f1v   = f1(eta, Om0)
    f2v   = f2(eta, Om0)
    mu_kf = mu(eta, kf, Om0, fR0, invH0)
    mu_k1 = mu(eta, k1, Om0, fR0, invH0)
    mu_k2 = mu(eta, k2, Om0, fR0, invH0)

    sA = sourceA(kf, k1, k2, eta, Om0, fR0, invH0, f2v, mu_kf, mu_k1, mu_k2)
    sB = sourceB(f2v, mu_kf, mu_k1, mu_k2)

    daf  = afp
    dafp = -f1v * afp + f2v * mu_kf * af + sA * d1 * d2
    dbf  = bfp
    dbfp = -f1v * bfp + f2v * mu_kf * bf + sB * d1 * d2
    dd1  = d1p
    dd1p = -f1v * d1p + f2v * mu_k1 * d1
    dd2  = d2p
    dd2p = -f1v * d2p + f2v * mu_k2 * d2

    return jnp.array([daf, dafp, dbf, dbfp, dd1, dd1p, dd2, dd2p])

def rk4Step(rhs_fn, y, eta, dt):
    """
    Generic RK4 stage shared by solveOde and computeF0; rhs_fn takes only (eta, y).
    """
    k1 = rhs_fn(eta,            y)
    k2 = rhs_fn(eta + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs_fn(eta + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs_fn(eta + dt,       y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def solveOde(rhs_fn, y0, eta_array):
    """
    Uses lax.scan instead of a Python for-loop so the integrator stays traceable/jittable.
    """
    dt = eta_array[1] - eta_array[0]

    def step(y, eta):
        return rk4Step(rhs_fn, y, eta, dt), None

    y_final, _ = lax.scan(step, y0, eta_array[:-1])
    return y_final

def getInitialConditions(etaini):
    d1_seed  = jnp.exp(etaini)
    d1p_seed = jnp.exp(etaini)
    af_seed  = (3.0 / 7.0) * jnp.exp(2.0 * etaini)
    afp_seed = (6.0 / 7.0) * jnp.exp(2.0 * etaini)

    return jnp.array([af_seed, afp_seed, af_seed, afp_seed,
                       d1_seed, d1p_seed, d1_seed, d1p_seed])

"""
@jit traces this function once per input shape/dtype and bakes in CONFIG["N_steps"]/
CONFIG["invH0"] as Python constants at that point; changing CONFIG afterwards has no effect
on an already-compiled trace.
"""
@jit
def computeAAndB(dataset_arr):
    invH0   = CONFIG["invH0"]
    etaini  = CONFIG["etaini"]
    N_steps = CONFIG["N_steps"]
    z, k1, k2, x12, Om0, log10fR0 = dataset_arr

    fR0   = 10.0 ** log10fR0
    etaev = -jnp.log(1.0 + z)
    kf    = jnp.sqrt(k1**2 + k2**2 + 2.0 * k1 * k2 * x12)

    eta_array = jnp.linspace(etaini, etaev, N_steps)
    y0        = getInitialConditions(etaini)
    rhs_fn    = lambda eta, y: rhs(eta, y, kf, k1, k2, Om0, fR0, invH0)

    y_final = solveOde(rhs_fn, y0, eta_array)

    af, afp, bf, bfp, d1, d1p, d2, d2p = y_final
    return postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p)

def postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p):
    norm   = (3.0 / 7.0) * d1 * d2
    d_norm = (3.0 / 7.0) * (d1p * d2 + d1 * d2p)

    a_val       = af / norm
    a_prime_val = afp / norm - af * d_norm / norm**2
    b_val       = bf / norm
    b_prime_val = bfp / norm - bf * d_norm / norm**2

    return jnp.array([a_val, a_prime_val, b_val, b_prime_val])

"""
batch_solver is a static arg because lax.map's batch_size must be a Python int.
"""
@partial(jit, static_argnames=("batch_solver",))
def _solveAAndBBatchTraced(dataset_in, batch_solver):
    return lax.map(computeAAndB, dataset_in, batch_size=batch_solver)

def resolveBatchSolver(n_rows, default_batch_solver=None):
    """
    lax.map treats batch_size=0 as "vmap everything in one shot", so datasets at or below
    batch_solver skip chunking entirely.
    """
    if default_batch_solver is None:
        default_batch_solver = CONFIG["batch_solver"]
    return 0 if n_rows <= default_batch_solver else default_batch_solver

def solveAAndBBatch(dataset_in):
    """
    lax.map vmaps internally in chunks of batch_solver rows to stay within GPU memory.
    """
    batch_solver = resolveBatchSolver(dataset_in.shape[0])

    return np.asarray(_solveAAndBBatchTraced(jnp.asarray(dataset_in), batch_solver))

COLUMNS = ["z", "k1", "k2", "x12", "Om0", "log10fR0", "A", "A'", "B", "B'"]

def writeResults(path, train_in, train_out):
    """
    Saved as float32: the RK4 integrator runs in jax x64 for numerical stability, but
    that's independent of output precision, which was already truncated to ~7
    significant figures by the old "%.6e" text format.
    """
    data = np.hstack([train_in, train_out]).astype(np.float32)

    np.savez_compressed(path, data=data, columns=COLUMNS)

def generateSamples():
    """
    Sobol requires power-of-2 sample sizes, so N_train/N_z are rounded up to the nearest
    one. validation/test are sized at N_train/4, the closest power-of-2 fraction to the
    ~20% target. Each cosmology is replicated across N_z redshift values from a single
    z sample, shared across train/validation/test.
    """
    N_train  = CONFIG["N_train"]
    N_z      = CONFIG["N_z"]
    z_bounds = CONFIG["z_bounds"]
    lower    = CONFIG["slower"]
    upper    = CONFIG["supper"]

    N_z        = 2 ** ceil(log2(N_z))
    N_train    = 2 ** ceil(log2(N_train))
    N_val_test = int(N_train / 4)

    sampler_z         = Sobol(d=1, scramble=True, seed=CONFIG["seed_z"])
    sampler_train_val = Sobol(d=len(lower), scramble=True, seed=CONFIG["seed_train_val"])
    sampler_test      = Sobol(d=len(lower), scramble=True, seed=CONFIG["seed_test"])

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

def growthRateRhs(eta, f0, Om0):
    return f2(eta, Om0) - f0**2 - f1(eta, Om0) * f0

@partial(jit, static_argnames=("N_steps",))
def computeF0(eta_ev, Om0, etaini, N_steps):
    eta_array = jnp.linspace(etaini, eta_ev, N_steps)
    rhs_fn = lambda eta, f0: growthRateRhs(eta, f0, Om0)
    return solveOde(rhs_fn, 1.0, eta_array)

"""
Lives here (not in kernels.py) so the growth-rate ODE and the F2/G2 combination stay next
to the rest of the numerical core that kernels.py only imports and reuses.
"""
def calKernels(z_arr, AB_functions, k1, k2, x12, Om0, invH0, etaini, N_steps):
    a_val, a_prime, b_val, b_prime = AB_functions.T
    eta = -np.log(1 + z_arr)

    k1_k2 = k1 * k2
    k_dot = k1 * k2 * x12
    f1_val = f1(eta, Om0)
    f2_val = f2(eta, Om0)
    h_eta  = invH0 * hubble(eta, Om0)

    f0 = vmap(lambda e: computeF0(e, Om0, etaini, N_steps))(jnp.array(eta))

    F2 = (
        0.5
        + (3.0 / 14.0) * a_val
        + (0.5 - (3.0 / 14.0) * b_val) * k_dot**2 / k1_k2**2
        + k_dot / (2 * k1_k2) * (k1 / k2 + k2 / k1)
    )

    G2 = (
        (3.0 * a_val * (f1_val + f2_val) + 3.0 * a_prime / h_eta) / (14.0 * f0)
        + ((f1_val + f2_val) / (2 * f0)
           - (3 * b_val * (f1_val + f2_val) + 3 * b_prime / h_eta) / (14.0 * f0))
          * k_dot**2 / k1_k2**2
        + k_dot / (2 * k1_k2) * (
              (f2_val * k2) / (f0 * k1)
            + (f1_val * k1) / (f0 * k2)
          )
    )

    return F2, G2
