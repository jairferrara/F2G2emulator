"""
Tests para numerical/ode.py.

Nota sobre JIT: `AandBfunctions` está decorada con @jit y lee CONFIG["N_steps"]/CONFIG["invH0"]
como globales de Python (no como argumentos trazados). JAX cachea la compilación por firma de
shape/dtype de entrada; si CONFIG cambiara *después* de la primera traza con esa shape, el
cambio no se reflejaría. Por eso aquí NUNCA se mockea CONFIG["N_steps"] -- los tests de
AandBfunctions/AandB_solver usan el CONFIG real. `solve_ode`/`rk4_step` no están jiteadas, así
que para probar el integrador se las llama directamente con un N_steps propio del test.
`compute_f0` recibe N_steps como argumento estático (`static_argnames`), por lo que tampoco
tiene este problema.
"""

import math

import numpy as np
import jax.numpy as jnp
import pytest

from numerical import ode


# ---------------------------------------------------------------------------
# 1. Cosmología de fondo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Om0", [0.1, 0.25, 0.3, 0.4])
def test_Omega_m_today_equals_Om0(Om0):
    assert float(ode.Omega_m(0.0, Om0)) == pytest.approx(Om0)


def test_Omega_m_matter_domination_limit():
    # eta muy negativo -> dominación de materia -> Omega_m -> 1
    assert float(ode.Omega_m(-50.0, 0.3)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("Om0", [0.1, 0.25, 0.3, 0.4])
def test_H_func_today_equals_one(Om0):
    assert float(ode.H_func(0.0, Om0)) == pytest.approx(1.0)


@pytest.mark.parametrize("eta", [-10.0, -1.0, 0.0, 2.0])
@pytest.mark.parametrize("Om0", [0.1, 0.25, 0.4])
def test_f1_plus_f2_identity(eta, Om0):
    # f1 = 2 - 1.5*Omega_m, f2 = 1.5*Omega_m  =>  f1 + f2 == 2 siempre
    total = float(ode.f1(eta, Om0)) + float(ode.f2(eta, Om0))
    assert total == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 2. Funciones dependientes del modelo
# ---------------------------------------------------------------------------

def test_mu_screening_limit_k_to_zero():
    mu_val = float(ode.mu(0.0, 1e-12, 0.3, 1e-5, ode.CONFIG["invH0"]))
    assert mu_val == pytest.approx(1.0, abs=1e-6)


def test_mu_no_screening_limit_k_to_infinity():
    mu_val = float(ode.mu(0.0, 1e12, 0.3, 1e-5, ode.CONFIG["invH0"]))
    assert mu_val == pytest.approx(4.0 / 3.0, abs=1e-6)


def test_mass_and_M2_are_positive():
    eta, Om0, fR0, invH0 = 0.0, 0.3, 1e-5, ode.CONFIG["invH0"]
    assert float(ode.mass(eta, Om0, fR0, invH0)) > 0.0
    assert float(ode.M2(eta, Om0, fR0, invH0)) > 0.0


# ---------------------------------------------------------------------------
# 3. Integrador RK4 (vía solve_ode directo, sin pasar por el jit de AandBfunctions)
# ---------------------------------------------------------------------------

def _growing_mode_d1_error(N_steps, etaini=-4.0, etaev=-1.0):
    """
    Para Om0=1 (dominación de materia exacta) y k1, k2 muy pequeños frente a la escala de masa
    (mu_k1, mu_k2 ~ 1, régimen GR), la ecuación para d1 se reduce a
    d1'' + 0.5 d1' - 1.5 d1 = 0, cuyo modo creciente compatible con las condiciones iniciales
    (Dplusi = dDplusi = exp(etaini)) es exactamente d1(eta) = exp(eta).

    Devuelve el error absoluto entre el d1 final de solve_ode y exp(etaev).
    """
    Om0    = 1.0
    fR0    = 1e-7      # extremo "screened" de CONFIG["slower"][-1] (log10fR0 = -7)
    k1     = k2 = 1e-3  # mucho menor que la escala de masa -> mu_k1, mu_k2 ~ 1
    invH0  = ode.CONFIG["invH0"]
    kf     = k1         # irrelevante: af/bf están desacoplados de d1/d2

    eta_array = jnp.linspace(etaini, etaev, N_steps)
    y0        = ode.get_initial_conditions(etaini)
    args      = (kf, k1, k2, Om0, fR0, invH0)

    y_final = ode.solve_ode(y0, eta_array, args)
    d1_final = float(y_final[4])

    return abs(d1_final - math.exp(etaev))


def test_rk4_matches_analytic_growing_mode():
    error = _growing_mode_d1_error(N_steps=500)
    assert error < 1e-4


def test_rk4_convergence_with_more_steps():
    coarse_error = _growing_mode_d1_error(N_steps=50)
    fine_error   = _growing_mode_d1_error(N_steps=2000)
    assert fine_error < coarse_error


# ---------------------------------------------------------------------------
# 4. Condiciones iniciales
# ---------------------------------------------------------------------------

def test_get_initial_conditions_matches_closed_form():
    etaini = -4.0
    y0 = np.asarray(ode.get_initial_conditions(etaini))

    Dplusi   = math.exp(etaini)
    dDplusi  = math.exp(etaini)
    D2plusi  = (3.0 / 7.0) * math.exp(2.0 * etaini)
    dD2plusi = (6.0 / 7.0) * math.exp(2.0 * etaini)

    expected = np.array([D2plusi, dD2plusi, D2plusi, dD2plusi,
                          Dplusi,  dDplusi,  Dplusi,  dDplusi])
    np.testing.assert_allclose(y0, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Postprocess (aritmética pura)
# ---------------------------------------------------------------------------

def test_postprocess_arithmetic():
    # d1 = d2 = 1, d1p = d2p = 0 => norm = 3/7, d_norm = 0 -> fórmulas se simplifican
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


# ---------------------------------------------------------------------------
# 6. AandBfunctions / AandB_solver (con CONFIG real, ver nota de cabecera)
# ---------------------------------------------------------------------------

def _sample_rows(n):
    # Filas dentro de los bounds de CONFIG["slower"]/CONFIG["supper"], variando z.
    z_vals = np.linspace(0.1, 1.5, n)
    rows = np.array([
        [z, 0.05, 0.05, 0.0, 0.3, -5.7] for z in z_vals
    ])
    return rows


def test_AandBfunctions_output_shape_and_finiteness():
    row = jnp.array(_sample_rows(1)[0])
    out = np.asarray(ode.AandBfunctions(row))

    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_AandB_solver_chunking_consistency(monkeypatch):
    dataset = _sample_rows(6)

    result_single_chunk = ode.AandB_solver(dataset)

    monkeypatch.setitem(ode.CONFIG, "batch_solver", 2)
    result_multi_chunk = ode.AandB_solver(dataset)

    assert result_single_chunk.shape == (6, 4)
    assert result_multi_chunk.shape == (6, 4)
    np.testing.assert_allclose(result_multi_chunk, result_single_chunk, rtol=1e-10)


# ---------------------------------------------------------------------------
# 7. write_results
# ---------------------------------------------------------------------------

def test_write_results_roundtrip(tmp_path):
    train_in  = np.array([[0.1, 0.05, 0.05, 0.0, 0.3, -5.7],
                           [0.2, 0.06, 0.04, 0.1, 0.35, -4.5]])
    train_out = np.array([[1.0, 0.1, 0.9, -0.1],
                           [1.1, 0.2, 0.8, -0.2]])

    out_path = tmp_path / "out.npz"
    ode.write_results(str(out_path), train_in, train_out)

    loaded = np.load(out_path)

    assert loaded["columns"].tolist() == ode.COLUMNS
    assert loaded["data"].dtype == np.float32
    assert loaded["data"].shape == (2, 10)
    np.testing.assert_allclose(
        loaded["data"],
        np.hstack([train_in, train_out]).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# 8. generate_samples
# ---------------------------------------------------------------------------

def test_generate_samples_shapes_and_bounds(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "N_train", 8)
    monkeypatch.setitem(ode.CONFIG, "N_z", 4)

    train, validation, test = ode.generate_samples()

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


def test_generate_samples_train_validation_do_not_overlap(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "N_train", 8)
    monkeypatch.setitem(ode.CONFIG, "N_z", 4)

    train, validation, _test = ode.generate_samples()
    N_z = 4

    train_cosmologies      = train[::N_z][:, 1:]
    validation_cosmologies = validation[::N_z][:, 1:]

    for row in validation_cosmologies:
        assert not np.any(np.all(np.isclose(train_cosmologies, row), axis=1))


def test_generate_samples_is_reproducible(monkeypatch):
    monkeypatch.setitem(ode.CONFIG, "N_train", 8)
    monkeypatch.setitem(ode.CONFIG, "N_z", 4)

    train_a, validation_a, test_a = ode.generate_samples()
    train_b, validation_b, test_b = ode.generate_samples()

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    np.testing.assert_array_equal(test_a, test_b)


# ---------------------------------------------------------------------------
# 9. compute_f0 (punto fijo en dominación de materia)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eta_ev", [-3.0, -1.5, 0.0])
def test_compute_f0_fixed_point_at_Om0_one(eta_ev):
    # rhs_f0(eta, f0=1, Om0=1) = f2 - f0^2 - f1*f0 = 1.5 - 1 - 0.5 = 0 -> f0=1 es punto fijo
    f0 = float(ode.compute_f0(eta_ev, 1.0, etaini=-4.0, N_steps=50))
    assert f0 == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 10. calKernels (smoke test)
# ---------------------------------------------------------------------------

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
