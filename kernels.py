import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from common.progress import stage
from common.ode import calKernels, reportJaxBackend
from common.inference import (
    loadSplit, scaleSplit, unscale, predictAndUnscale,
    relativeErrorPct, percentileReport, emulate,
)
from common.model import loadModel

reportJaxBackend()

"""
Valid emulator domain: k1, k2 in [0.001, 0.6]; x12 in [-1, 1]; Om0 in [0.1, 0.4];
log10fR0 in [-7, -4] (matches common.ode.CONFIG["slower"]/["supper"]).
"""
params = {
               "k1" : 0.1,
               "k2" : 0.1,
              "x12" : 0.2,
              "Om0" : 0.3,
         "log10fR0" : -5.7,
            "z_max" : 1.7,
            "invH0" : 2997.92458,
       "batch_size" : 64,
           "etaini" : -4,
          "N_steps" : 4,
    "path_datasets" : "./src/datasets/",
       "path_model" : "./src/model/",
       "path_plots" : "./src/plots/"
}

_RC = {
    "font.family":          "serif",
    "mathtext.fontset":     "cm",
    "axes.titlesize":       13,
    "axes.labelsize":       12,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      9.5,
    "legend.framealpha":    0.90,
    "legend.edgecolor":     "0.75",
    "legend.handlelength":  2.0,
    "axes.grid":            True,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.50,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.constrained_layout.use": True,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
}

"""
Color roles reused across the plotting functions below: _C_TRUE/_C_PRED/_C_FILL for the
numeric-vs-emulator comparison, _C_F2/_C_G2 for the two kernels, _C_HIST/_C_P99 for the
error histograms.
"""
_C_TRUE = "#1B4F9B"
_C_PRED = "#C0392B"
_C_FILL = "#EAB7B2"
_C_F2   = "#1B4F9B"
_C_G2   = "#D35400"
_C_HIST = "#2980B9"
_C_P99  = "#C0392B"

_NAMES  = [r"\mathcal{A}", r"\mathcal{A}^{\prime}", r"\mathcal{B}", r"\mathcal{B}^{\prime}"]
_DELTA_NAMES = [r"$\Delta\mathcal{A}\;[\%]$", r"$\Delta\mathcal{A}^{\prime}\;[\%]$",
                r"$\Delta\mathcal{B}\;[\%]$", r"$\Delta\mathcal{B}^{\prime}\;[\%]$"]

def importModel():
    model, scaler_i, scaler_o, training_params = loadModel(params["path_model"])
    print(f"Loaded model trained with neurons={training_params['neurons']}, "
          f"activations={training_params['activations']}.")

    return model, scaler_i, scaler_o

def loadData():
    path_datasets = params["path_datasets"]

    test = loadSplit(path_datasets, "test")

    print("Samples size:")
    print(f"Test: {len(test)}.\n")

    return test

def _plotComparison(scaled_x, unscaled_y, unscaled_y_predic, scaler_i):
    """
    N_group must match common.ode.CONFIG["N_z"]: each cosmology occupies a contiguous
    block of N_z rows in the test split, and only the first block is plotted here.
    """
    N_group    = 64
    unscaled_x = scaler_i.inverse_transform(scaled_x)
    idx        = np.argsort(unscaled_x[:N_group, 0])
    z_arr      = unscaled_x[:N_group][idx, 0]
    y_true     = unscaled_y[:N_group][idx]
    y_pred     = unscaled_y_predic[:N_group][idx]
    k1_val, k2_val, x12_val, om0_val, log10fr0_val = unscaled_x[0, 1:]

    _rc_local = {**_RC, "figure.constrained_layout.use": False}
    with plt.rc_context(_rc_local):

        fig = plt.figure(figsize=(11, 10))

        outer = GridSpec(
            2, 2, figure=fig,
            hspace=0.15, wspace=0.30,
            top=0.93, bottom=0.07, left=0.09, right=0.97,
        )

        axes_m, axes_r = [], []
        for i in range(2):
            for j in range(2):
                inner = GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=outer[i, j],
                    height_ratios=[3, 1],
                    hspace=0.06,
                )
                ax_m = fig.add_subplot(inner[0])
                ax_r = fig.add_subplot(inner[1], sharex=ax_m)
                axes_m.append(ax_m)
                axes_r.append(ax_r)
                plt.setp(ax_m.get_xticklabels(), visible=False)

        fig.suptitle(
            r"Comparación entre resultados numéricos vs emulados",
            fontsize=14, fontweight="bold",
        )

        param_text = "\n".join([
            "Configuración de modo",
            rf"$k_1 = {k1_val:.4f}\ h\,\mathrm{{Mpc}}^{{-1}}$",
            rf"$k_2 = {k2_val:.4f}\ h\,\mathrm{{Mpc}}^{{-1}}$",
            rf"$x_{{12}} = \hat{{k}}_1\cdot\hat{{k}}_2 = {x12_val:.4f}$",
            "",
            "Parámetros cosmológicos",
            rf"$\Omega_m = {om0_val:.4f}$",
            rf"$\log_{{10}}(fR_0) = {log10fr0_val:.4f}$",
        ])
        axes_m[1].text(
            0.97, 0.04,
            param_text,
            transform=axes_m[1].transAxes,
            ha="right", va="bottom",
            fontsize=8.2,
            linespacing=1.6,
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor="#FEFCE8",
                alpha=0.93,
                edgecolor="0.55",
                linewidth=0.9,
            ),
        )

        for k, (ax_m, ax_r) in enumerate(zip(axes_m, axes_r)):
            truth   = y_true[:, k]
            pred    = y_pred[:, k]
            rel_pct = (pred / truth - 1.0) * 100.0

            ax_m.plot(z_arr, truth, color=_C_TRUE, lw=1.8,
                      label="Numérico", zorder=3)
            ax_m.plot(z_arr, pred,   color=_C_PRED, lw=1.4, ls="--",
                      label="Emulador", zorder=3)
            ax_m.fill_between(z_arr, truth, pred,
                              color=_C_FILL, alpha=0.30, zorder=2)
            ax_m.set_ylabel(rf"${_NAMES[k]}$", fontsize=12)
            if k == 0:
                ax_m.legend(loc="lower right", fontsize=9)

            ax_r.plot(z_arr, rel_pct, color="0.20", lw=1.2, zorder=3)
            ax_r.axhline(0.0, color="0.50", lw=0.7, ls=":", zorder=2)
            ax_r.fill_between(
                z_arr, rel_pct, 0.0,
                where=(rel_pct >= 0), color=_C_TRUE, alpha=0.15, zorder=1,
            )
            ax_r.fill_between(
                z_arr, rel_pct, 0.0,
                where=(rel_pct < 0),  color=_C_PRED, alpha=0.15, zorder=1,
            )

            amax = max(np.abs(rel_pct).max() * 1.35, 0.02)
            ax_r.set_ylim(-amax, amax)
            ax_r.set_ylabel(rf"{_DELTA_NAMES[k]}", fontsize=9, labelpad=2)
            ax_r.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=3, symmetric=True)
            )
            ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax_r.tick_params(axis="both", labelsize=8)

        for ax_r in axes_r[2:]:
            ax_r.set_xlabel(r"Redshift $z$", fontsize=11)

        plt.savefig(os.path.join(params["path_plots"], "comparation.pdf"))
        plt.show()

def plotRelError(rel_error):
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(
            "Error relativo porcentual del emulador",
            fontsize=14, fontweight="bold",
        )

        for c in range(2):
            for r in range(2):
                k   = r * 2 + c
                ax  = axes[r, c]
                err = rel_error.T[k]

                p99 = float(np.percentile(np.abs(err), 99))
                rng = (-p99 * 1.08, p99 * 1.08)

                n, bins, _ = ax.hist(
                    err, bins=60, range=rng,
                    log=True, orientation="horizontal",
                    color=_C_HIST, alpha=0.75,
                    edgecolor="white", linewidth=0.2,
                )

                ax.axhline(+p99, color=_C_P99, lw=1.6, ls="--", zorder=5,
                           label=rf"$p_{{99}} = {p99:.2f}\%$")
                ax.axhline(-p99, color=_C_P99, lw=1.6, ls="--", zorder=5)
                ax.axhline(0.0,  color="0.40", lw=0.8, ls=":",  zorder=4)

                ax.axhspan(+p99,  rng[1], color=_C_P99, alpha=0.07, zorder=1)
                ax.axhspan(rng[0], -p99,  color=_C_P99, alpha=0.07, zorder=1)

                ax.set_ylabel(
                    rf"$\Delta\,{_NAMES[k]}\;[\%]$", fontsize=12, labelpad=3
                )
                ax.set_xlabel("Frecuencia", fontsize=11)
                ax.set_ylim(rng)
                ax.legend(fontsize=9, loc="lower right",
                          handlelength=1.2, framealpha=0.85)

        plt.savefig(os.path.join(params["path_plots"], "error.pdf"))
        plt.show()

def plotKernels(z_arr, F2, G2, F2_gr=None, G2_gr=None):
    data = [(F2, _C_F2, r"$F_2(z)$"), (G2, _C_G2, r"$G_2(z)$")]
    refs = [F2_gr, G2_gr]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3))
        fig.suptitle(
            r"Kernels $F_2(z)$ y $G_2(z)$ emulados",
            fontsize=13, fontweight="bold",
        )

        _gr_labels = [r"$F_2^{\rm GR}(z)$", r"$G_2^{\rm GR}(z)$"]
        for i, (ax, (vals, color, label), ref) in enumerate(zip(axes, data, refs)):
            ax.plot(z_arr, vals, color=color, lw=2.0,
                    label=label, zorder=3)

            if ref is not None:
                ax.plot(z_arr, ref, color="0.50", lw=1.2, ls="--",
                        label=_gr_labels[i], zorder=2)
                ax.fill_between(z_arr, vals, ref,
                                color=color, alpha=0.12, zorder=1,
                                label=r"$\Delta$ (mod. grav.)")

            ax.axhline(vals[0], color=color, lw=0.8, ls=":",
                       alpha=0.55, zorder=2)
            ax.text(
                0.03, 0.97,
                rf"$z=0$: {vals[0]:.4f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color=color,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", alpha=0.75, edgecolor="none"),
            )

            ax.set_xlabel(r"Redshift $z$", fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.legend(fontsize=9.5, loc="upper right")

        plt.savefig(os.path.join(params["path_plots"], "kernels.pdf"))
        plt.show()

with stage("Loading model"):
    model, scaler_i, scaler_o = importModel()
with stage("Loading test split"):
    test = loadData()

with stage("Evaluating on test split"):
    scaled_x, scaled_y = scaleSplit(test, scaler_i, scaler_o)
    unscaled_y          = unscale(scaled_y, scaler_o)
    unscaled_y_predic    = predictAndUnscale(model, scaled_x, scaler_o, params["batch_size"])

test_error = relativeErrorPct(unscaled_y, unscaled_y_predic)
percentileReport(test_error)
_plotComparison(scaled_x, unscaled_y, unscaled_y_predic, scaler_i)

plotRelError(test_error)

k1       = params["k1"]
k2       = params["k2"]
x12      = params["x12"]
Om0      = params["Om0"]
"""
The ODE solver works with fR0, not log10fR0; the conversion only happens here.
"""
fR0      = 10**params["log10fR0"]
z_max    = params["z_max"]

z_arr = np.linspace(0, z_max, 100)
args = np.array([k1, k2, x12, Om0, fR0])

AB_functions = emulate(z_arr, args, model, scaler_i, scaler_o, params["batch_size"])

F2, G2 = calKernels(
    z_arr, AB_functions,
    k1, k2, x12, Om0, params["invH0"], params["etaini"], params["N_steps"],
)

plotKernels(z_arr, F2, G2)

print("Done. Wrote comparation.pdf, error.pdf, kernels.pdf.")
