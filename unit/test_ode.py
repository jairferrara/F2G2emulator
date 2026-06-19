"""
Tests for common/ode.py.

JIT note: `computeAAndB` is decorated with @jit and reads CONFIG["N_steps"]/CONFIG["invH0"]
as Python globals (not traced arguments). JAX caches the compiled trace by input
shape/dtype; if CONFIG changed *after* the first trace with that shape, the change would
have no effect. So CONFIG["N_steps"] is NEVER mocked here -- the computeAAndB/solveAAndBBatch
tests use the real CONFIG. `solveOde`/`rk4Step` aren't jitted, so the integrator is tested
by calling them directly with a test-local N_steps. `computeF0` takes N_steps as a static
argument (`static_argnames`), so it doesn't have this problem.
"""

import math

import numpy as np
import jax.numpy as jnp
import pytest

from common import ode


"""1. Background cosmology"""

@pytest.mark.parametrize("Om0", [0.1, 0.25, 0.3, 0.4])
def test_omegaM_today_equals_Om0(Om0):
    assert float(ode.omegaM(0.0, Om0)) == pytest.approx(Om0)


def test_omegaM_matter_domination_limit():
    """As eta -> -inf the universe is matter-dominated, so omegaM -> 1."""
    assert float(ode.omegaM(-50.0, 0.3)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("Om0", [0.1, 0.25, 0.3, 0.4])
def test_hubble_today_equals_one(Om0):
    assert float(ode.hubble(0.0, Om0)) == pytest.approx(1.0)


@pytest.mark.parametrize("eta", [-10.0, -1.0, 0.0, 2.0])
@pytest.mark.parametrize("Om0", [0.1, 0.25, 0.4])
def test_f1_plus_f2_identity(eta, Om0):
    """f1 = 2 - 1.5*Omega_m and f2 = 1.5*Omega_m, so f1 + f2 == 2 always."""
    total = float(ode.f1(eta, Om0)) + float(ode.f2(eta, Om0))
    assert total == pytest.approx(2.0)


"""2. Model-dependent functions"""

def test_mu_screening_limit_k_to_zero():
    mu_val = float(ode.mu(0.0, 1e-12, 0.3, 1e-5, ode.CONFIG["invH0"]))
    assert mu_val == pytest.approx(1.0, abs=1e-6)


def test_mu_no_screening_limit_k_to_infinity():
    mu_val = float(ode.mu(0.0, 1e12, 0.3, 1e-5, ode.CONFIG["invH0"]))
    assert mu_val == pytest.approx(4.0 / 3.0, abs=1e-6)


def test_scalaronMass_and_m2_are_positive():
    eta, Om0, fR0, invH0 = 0.0, 0.3, 1e-5, ode.CONFIG["invH0"]
    assert float(ode.scalaronMass(eta, Om0, fR0, invH0)) > 0.0
    assert float(ode.m2(eta, Om0, fR0, invH0)) > 0.0


"""3. RK4 integrator (via solveOde directly, bypassing computeAAndB's jit)"""

def _growingModeD1Error(N_steps, etaini=-4.0, etaev=-1.0):
    """
    For Om0=1 and k1, k2 ~0 (GR regime), d1's growing mode has the closed form
    d1(eta) = exp(eta); returns the absolute error between solveOde and that value.
    """
    Om0    = 1.0
    fR0    = 1e-7
    k1     = k2 = 1e-3
    invH0  = ode.CONFIG["invH0"]
    kf     = k1

    eta_array = jnp.linspace(etaini, etaev, N_steps)
    y0     = ode.getInitialConditions(etaini)
    rhs_fn = lambda eta, y: ode.rhs(eta, y, kf, k1, k2, Om0, fR0, invH0)

    y_final = ode.solveOde(rhs_fn, y0, eta_array)
    d1_final = float(y_final[4])

    return abs(d1_final - math.exp(etaev))


def test_rk4_matches_analytic_growing_mode():
    error = _growingModeD1Error(N_steps=500)
    assert error < 1e-4


def test_rk4_convergence_with_more_steps():
    coarse_error = _growingModeD1Error(N_steps=50)
    fine_error   = _growingModeD1Error(N_steps=2000)
    assert fine_error < coarse_error


"""4. Initial conditions"""

def test_getInitialConditions_matches_closed_form():
    etaini = -4.0
    y0 = np.asarray(ode.getInitialConditions(etaini))

    d1_seed  = math.exp(etaini)
    d1p_seed = math.exp(etaini)
    af_seed  = (3.0 / 7.0) * math.exp(2.0 * etaini)
    afp_seed = (6.0 / 7.0) * math.exp(2.0 * etaini)

    expected = np.array([af_seed, afp_seed, af_seed, afp_seed,
                          d1_seed, d1p_seed, d1_seed, d1p_seed])
    np.testing.assert_allclose(y0, expected, rtol=1e-12)


"""5. Postprocess (pure arithmetic)"""

def test_postprocess_arithmetic():
    """d1 = d2 = 1, d1p = d2p = 0 => norm = 3/7, d_norm = 0, simplifying the formulas."""
    af, afp, bf, bfp = 1.0, 2.0, 3.0, 4.0
    d1, d1p, d2, d2p = 1.0, 0.0, 1.0, 0.0

    result = np.asarray(ode.postprocess(af, afp, bf, bfp, d1, d1p, d2, d2p))

    expected = np.array([
        af * 7.0 / 3.0,
        afp * 7.0 / 3.0,
        bf * 7.0 / 3.0,
        bfp * 7.0 / 3.0,
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


"""5b. Environment toggles (resolveJaxPlatform / resolveX64Enabled)"""

def test_resolveJaxPlatform_defaults_to_cpu():
    assert ode.resolveJaxPlatform({}) == "cpu"


def test_resolveJaxPlatform_respects_override():
    assert ode.resolveJaxPlatform({"JAX_PLATFORMS": "cuda"}) == "cuda"


def test_resolveX64Enabled_defaults_to_true():
    assert ode.resolveX64Enabled({}) is True


@pytest.mark.parametrize("value", ["0", "false", "False"])
def test_resolveX64Enabled_can_be_disabled(value):
    assert ode.resolveX64Enabled({"F2G2_JAX_X64": value}) is False


def test_resolveX64Enabled_other_values_stay_enabled():
    assert ode.resolveX64Enabled({"F2G2_JAX_X64": "1"}) is True


"""6. computeAAndB / solveAAndBBatch (with the real CONFIG, see header note)"""

def _sampleRows(n):
    """Rows within CONFIG["slower"]/CONFIG["supper"] bounds, varying z."""
    z_vals = np.linspace(0.1, 1.5, n)
    rows = np.array([
        [z, 0.05, 0.05, 0.0, 0.3, -5.7] for z in z_vals
    ])
    return rows


def test_computeAAndB_output_shape_and_finiteness():
    row = jnp.array(_sampleRows(1)[0])
    out = np.asarray(ode.computeAAndB(row))

    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_solveAAndBBatch_chunking_consistency(monkeypatch):
    dataset = _sampleRows(6)

    result_single_chunk = ode.solveAAndBBatch(dataset)

    monkeypatch.setitem(ode.CONFIG, "batch_solver", 2)
    result_multi_chunk = ode.solveAAndBBatch(dataset)

    assert result_single_chunk.shape == (6, 4)
    assert result_multi_chunk.shape == (6, 4)
    np.testing.assert_allclose(result_multi_chunk, result_single_chunk, rtol=1e-10)


def test_solveAAndBBatch_handles_non_divisible_remainder_chunk(monkeypatch):
    """
    batch_solver=3 with 7 rows leaves a 1-row remainder chunk; checks it still matches
    computeAAndB applied row-by-row.
    """
    dataset = _sampleRows(7)

    monkeypatch.setitem(ode.CONFIG, "batch_solver", 3)
    batched = ode.solveAAndBBatch(dataset)

    expected = np.stack([
        np.asarray(ode.computeAAndB(jnp.array(row))) for row in dataset
    ])

    assert batched.shape == (7, 4)
    np.testing.assert_allclose(batched, expected, rtol=1e-10)


"""6b. resolveBatchSolver"""

def test_resolveBatchSolver_uses_no_chunking_for_small_datasets():
    assert ode.resolveBatchSolver(100, default_batch_solver=16384) == 0


def test_resolveBatchSolver_keeps_default_for_large_datasets():
    assert ode.resolveBatchSolver(20000, default_batch_solver=16384) == 16384


def test_resolveBatchSolver_boundary_equal_to_default_uses_no_chunking():
    assert ode.resolveBatchSolver(16384, default_batch_solver=16384) == 0


def test_resolveBatchSolver_reads_CONFIG_when_default_unspecified(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "batch_solver", 500)

    assert ode.resolveBatchSolver(100) == 0
    assert ode.resolveBatchSolver(1000) == 500


def test_solveAAndBBatch_single_vmap_path_matches_row_by_row():
    """
    With the real (unmocked) CONFIG["batch_solver"]=16384, a small dataset takes
    resolveBatchSolver's batch_size=0 ("just vmap everything") path instead of chunking.
    """
    dataset = _sampleRows(6)

    assert ode.resolveBatchSolver(len(dataset)) == 0

    result = ode.solveAAndBBatch(dataset)
    expected = np.stack([
        np.asarray(ode.computeAAndB(jnp.array(row))) for row in dataset
    ])

    np.testing.assert_allclose(result, expected, rtol=1e-10)


"""7. writeResults"""

def test_writeResults_roundtrip(tmp_path):
    train_in  = np.array([[0.1, 0.05, 0.05, 0.0, 0.3, -5.7],
                           [0.2, 0.06, 0.04, 0.1, 0.35, -4.5]])
    train_out = np.array([[1.0, 0.1, 0.9, -0.1],
                           [1.1, 0.2, 0.8, -0.2]])

    out_path = tmp_path / "out.npz"
    ode.writeResults(str(out_path), train_in, train_out)

    loaded = np.load(out_path)

    assert loaded["columns"].tolist() == ode.COLUMNS
    assert loaded["data"].dtype == np.float32
    assert loaded["data"].shape == (2, 10)
    np.testing.assert_allclose(
        loaded["data"],
        np.hstack([train_in, train_out]).astype(np.float32),
    )


"""8. generateSamples"""

def test_generateSamples_shapes_and_bounds(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "N_train", 8)
    monkeypatch.setitem(ode.CONFIG, "N_z", 4)

    train, validation, test = ode.generateSamples()

    N_z         = 4
    N_train     = 8
    N_val_test  = N_train // 4

    assert train.shape      == (N_train * N_z, 6)
    assert validation.shape == (N_val_test * N_z, 6)
    assert test.shape       == (N_val_test * N_z, 6)

    z_bounds = ode.CONFIG["z_bounds"]
    lower    = np.array(ode.CONFIG["slower"])
    upper    = np.array(ode.CONFIG["supper"])

    for ds in (train, validation, test):
        assert np.all(ds[:, 0] >= z_bounds[0]) and np.all(ds[:, 0] <= z_bounds[1])
        assert np.all(ds[:, 1:] >= lower) and np.all(ds[:, 1:] <= upper)


def test_generateSamples_train_validation_do_not_overlap(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "N_train", 8)
    monkeypatch.setitem(ode.CONFIG, "N_z", 4)

    train, validation, _test = ode.generateSamples()
    N_z = 4

    train_cosmologies      = train[::N_z][:, 1:]
    validation_cosmologies = validation[::N_z][:, 1:]

    for row in validation_cosmologies:
        assert not np.any(np.all(np.isclose(train_cosmologies, row), axis=1))


def test_generateSamples_is_reproducible(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "N_train", 8)
    monkeypatch.setitem(ode.CONFIG, "N_z", 4)

    train_a, validation_a, test_a = ode.generateSamples()
    train_b, validation_b, test_b = ode.generateSamples()

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    np.testing.assert_array_equal(test_a, test_b)


"""8b. reportJaxBackend (explicit call, not an import-time side effect)"""

def test_reportJaxBackend_prints_backend_info(capsys):
    ode.reportJaxBackend()

    captured = capsys.readouterr().out
    assert "JAX backend:" in captured


"""9. computeF0 (fixed point under matter domination)"""

@pytest.mark.parametrize("eta_ev", [-3.0, -1.5, 0.0])
def test_computeF0_fixed_point_at_Om0_one(eta_ev):
    """growthRateRhs(eta, f0=1, Om0=1) = f2 - f0^2 - f1*f0 = 1.5 - 1 - 0.5 = 0, so f0=1 is a fixed point."""
    f0 = float(ode.computeF0(eta_ev, 1.0, etaini=-4.0, N_steps=50))
    assert f0 == pytest.approx(1.0, abs=1e-10)


"""10. calKernels (smoke test)"""

def test_calKernels_output_shapes_and_finiteness():
    z_arr = np.linspace(0.0, 1.0, 5)
    AB_functions = np.tile(np.array([0.9, 0.1, 0.95, 0.05]), (5, 1))

    k1, k2, x12, Om0 = 0.05, 0.05, 0.0, 0.3
    invH0  = ode.CONFIG["invH0"]
    etaini = ode.CONFIG["etaini"]

    F2, G2 = ode.calKernels(z_arr, AB_functions, k1, k2, x12, Om0, invH0, etaini, N_steps=50)

    F2 = np.asarray(F2)
    G2 = np.asarray(G2)

    assert F2.shape == (5,)
    assert G2.shape == (5,)
    assert np.all(np.isfinite(F2))
    assert np.all(np.isfinite(G2))
