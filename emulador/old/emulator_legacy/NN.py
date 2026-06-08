import os
import json
import time
import joblib
import itertools
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import Callback, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# # 1. Configuration & environment

# In[ ]:


def load_config(path):
    """Carga una configuración desde un archivo JSON."""
    with open(path, "r") as f:
        cfg = json.load(f)
    print(f"Config loaded from: {path}")
    return cfg


# In[ ]:


def set_seeds(seed, deterministic=True):
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            print(f"  Deterministic ops enabled (seed={seed})")
        except Exception as e:
            print(f"  Warning: op determinism not available: {e}")


# In[ ]:


def configure_gpu(use_mixed_prec=False):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        print(f"GPUs detected: {[g.name for g in gpus]}")

    if use_mixed_prec:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 enabled")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("Precision: float32 (mixed precision disabled)")


# # 2. Raw data manipulation

# In[ ]:


def signed_log_transform(y):
    return np.sign(y) * np.log1p(np.abs(y))


# In[ ]:


def signed_log_inverse(y_t):
    y_clipped = np.clip(y_t, -50.0, 50.0)
    return np.sign(y_clipped) * np.expm1(np.abs(y_clipped))


# In[ ]:


def load_data(filepath, input_cols):
    """Carga el dataset crudo y extrae la matriz de inputs.

    El slicing de outputs depende del grupo y se hace fuera, en main/run_search.
    """
    raw = np.loadtxt(filepath)
    X_full = raw[:, input_cols]
    print(f"  Loaded {raw.shape[0]} samples  |  X: {X_full.shape}")
    return raw, X_full


# In[ ]:


def preprocess(X, Y, val_size, test_size, seed, transform="none"):
    X_np, Y_np = np.array(X), np.array(Y)

    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_np, Y_np,
        test_size=test_size,
        random_state=seed
    )

    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval,
        test_size=val_relative,
        random_state=seed
    )

    if transform == "signed_log":
        Y_train = signed_log_transform(Y_train)
        Y_val   = signed_log_transform(Y_val)
        Y_test  = signed_log_transform(Y_test)
    elif transform != "none":
        raise ValueError(f"Unknown transform: {transform}")

    scaler_X = StandardScaler().fit(X_train)
    scaler_Y = StandardScaler().fit(Y_train)

    X_train = scaler_X.transform(X_train)
    X_val   = scaler_X.transform(X_val)
    X_test  = scaler_X.transform(X_test)
    Y_train = scaler_Y.transform(Y_train)
    Y_val   = scaler_Y.transform(Y_val)
    Y_test  = scaler_Y.transform(Y_test)

    print(f"  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Y transform: {transform}")
    return (X_train, X_val, X_test,
            Y_train, Y_val, Y_test,
            scaler_X, scaler_Y)


# # 3. Model & training bluiding blocks

# In[ ]:


class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Schedule basado en épocas, no en steps.

    Dos runs con distinto batch_size verán EXACTAMENTE la misma curva LR vs.
    época. Lo único que cambia entre runs es la granularidad (número de
    actualizaciones de LR por época). Esto hace los runs comparables.
    """
    def __init__(self, initial_lr, total_epochs, warmup_epochs,
                 steps_per_epoch, min_lr=1e-7):
        super().__init__()
        self.initial_lr      = float(initial_lr)
        self.total_epochs    = int(total_epochs)
        self.warmup_epochs   = int(warmup_epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.min_lr          = float(min_lr)

    def __call__(self, step):
        step            = tf.cast(step, tf.float32)
        steps_per_epoch = tf.cast(self.steps_per_epoch, tf.float32)
        warmup_epochs   = tf.cast(self.warmup_epochs,   tf.float32)
        total_epochs    = tf.cast(self.total_epochs,    tf.float32)

        epoch = step / tf.maximum(steps_per_epoch, 1.0)

        warmup_lr = self.initial_lr * (epoch / tf.maximum(warmup_epochs, 1.0))

        progress  = (epoch - warmup_epochs) / tf.maximum(total_epochs - warmup_epochs, 1.0)
        progress  = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                    (1.0 + tf.cos(np.pi * progress))

        return tf.where(epoch < warmup_epochs, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr":      self.initial_lr,
            "total_epochs":    self.total_epochs,
            "warmup_epochs":   self.warmup_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "min_lr":          self.min_lr,
        }


# In[ ]:


def build_and_compile(n_inputs, n_outputs, n_layers, n_neurons,
                      activation, output_activation, lr_schedule, delta):
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))

    for _ in range(n_layers):
        model.add(Dense(n_neurons, activation=activation))

    model.add(Dense(n_outputs, activation=output_activation, dtype="float32"))

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss=Huber(delta),
                  jit_compile=True,
                  steps_per_execution=min(50, steps_per_epoch),
                 )
    return model


# In[ ]:


class P99PercentageCallback(Callback):
    def __init__(self, X_val, Y_val_scaled, scaler_Y, transform="none", every=1):
        super().__init__()
        self.X_val        = X_val
        self.Y_val_scaled = Y_val_scaled
        self.scaler_Y     = scaler_Y
        self.transform    = transform
        self.every        = max(1, int(every))
        self._last        = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}

        # En épocas "intermedias" reutilizamos el último p99 calculado.
        # EarlyStopping seguirá viendo la métrica, pero no comparará valores
        # nuevos hasta el siguiente recálculo (lo cual es lo que queremos).
        if (epoch + 1) % self.every != 0:
            logs["val_worst_p99"] = self._last
            return

        # --- Forward pass directo, sin overhead de model.predict ---
        # model(x, training=False) devuelve un tensor; lo pasamos a numpy una sola vez.
        Y_pred_s = self.model(self.X_val, training=False).numpy()

        # --- Inversión de escalado y transform ---
        Y_true = self.scaler_Y.inverse_transform(self.Y_val_scaled)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_s)
        if self.transform == "signed_log":
            Y_true = signed_log_inverse(Y_true)
            Y_pred = signed_log_inverse(Y_pred)

        # --- Diferencia porcentual y p99 ---
        ref_scale = np.mean(np.abs(Y_true), axis=0, keepdims=True)
        ref_scale = np.where(ref_scale < 1e-10, 1.0, ref_scale)
        delta_pct = (Y_pred - Y_true) / ref_scale * 100.0

        p99_per_output = np.percentile(np.abs(delta_pct), 99, axis=0)
        worst_p99      = float(np.max(p99_per_output))

        self._last         = worst_p99
        logs["val_worst_p99"] = worst_p99


# In[ ]:


def get_callbacks(cfg, X_val, Y_val, scaler_Y, transform):
    p99_every = cfg.get("p99_every", 1)

    p99_cb = P99PercentageCallback(
        X_val, Y_val, scaler_Y,
        transform=transform,
        every=p99_every,
    )

    early = EarlyStopping(
        monitor=cfg["early_monitor"],
        mode="min",
        patience=cfg["early_patience"],
        restore_best_weights=True,
        verbose=0,
    )

    return [p99_cb, early]


# In[ ]:


def train(model, X_train, Y_train, X_val, Y_val, cfg, callbacks):
    batch_size = cfg["batch_size"]

    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_train, Y_train))
                .shuffle(buffer_size=min(len(X_train), 100_000), seed=cfg["seed"])
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset
              .from_tensor_slices((X_val, Y_val))
              .batch(batch_size * 2)        # val no necesita gradientes, batch mayor
              .prefetch(tf.data.AUTOTUNE))

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=0,
    )
    print(f"  Training time: {time.time() - t0:.2f} s")
    return history


# In[ ]:


def train_one_model(cfg, group_name, group_cfg, n_layers, n_neurons,
                    activation, lr, delta, batch_size,
                    X_train, Y_train, X_val, Y_val, scaler_Y):

    n_in      = len(cfg["input_cols"])
    n_out     = len(group_cfg["cols"])
    transform = group_cfg["transform"]

    # Schedule parametrizado por épocas: la curva LR vs. época es independiente
    # del batch_size, lo que hace runs comparables entre sí.
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    schedule = CosineWarmupSchedule(
        initial_lr=lr,
        total_epochs=cfg["epochs"],
        warmup_epochs=cfg["warmup_epochs"],
        steps_per_epoch=steps_per_epoch,
        min_lr=cfg["min_lr"],
    )

    model = build_and_compile(
        n_inputs=n_in,
        n_outputs=n_out,
        n_layers=n_layers,
        n_neurons=n_neurons,
        activation=activation,
        output_activation=cfg["output_activation"],
        lr_schedule=schedule,
        delta=delta,
        steps_per_epoch=steps_per_epoch,
    )

    callbacks = get_callbacks(cfg, X_val, Y_val, scaler_Y, transform)

    cfg_run = {**cfg, "batch_size": batch_size}    # local override
    t0 = time.time()
    history = train(model, X_train, Y_train, X_val, Y_val, cfg_run, callbacks)
    elapsed = time.time() - t0

    return model, history, elapsed


# # 4. Evaluation

# In[ ]:


def _compute_delta_pct(model, X, Y_scaled, scaler_Y, transform):
    """Predice, deshace escalado/transform y calcula ΔY% por columna.

    Helper compartido por evaluate, automatic_evaluate y P99PercentageCallback
    para evitar la triple duplicación de la misma lógica.
    """
    Y_pred_s = model.predict(X, verbose=0)

    Y_true = scaler_Y.inverse_transform(Y_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)

    if transform == "signed_log":
        Y_true = signed_log_inverse(Y_true)
        Y_pred = signed_log_inverse(Y_pred)

    ref_scale = np.mean(np.abs(Y_true), axis=0, keepdims=True)
    ref_scale = np.where(ref_scale < 1e-10, 1.0, ref_scale)
    delta_pct = (Y_pred - Y_true) / ref_scale * 100.0

    return Y_true, Y_pred, delta_pct


# In[ ]:


def _compute_stats(delta_pct, output_labels):
    """Resumen estadístico por output: median, std, p95, p99, max_abs."""
    stats = {}
    for i, label in enumerate(output_labels):
        col = delta_pct[:, i]
        stats[label] = {
            "median" : float(np.median(col)),
            "std"    : float(np.std(col)),
            "p95"    : float(np.percentile(np.abs(col), 95)),
            "p99"    : float(np.percentile(np.abs(col), 99)),
            "max_abs": float(np.max(np.abs(col))),
        }
    return stats


# In[ ]:


def automatic_evaluate(model, X_eval, Y_eval_scaled, scaler_Y,
                       output_labels, transform="none"):
    """Versión silenciosa: devuelve (worst_p99, stats, delta_pct)."""
    _, _, delta_pct = _compute_delta_pct(
        model, X_eval, Y_eval_scaled, scaler_Y, transform
    )
    stats = _compute_stats(delta_pct, output_labels)
    worst_error = max(s["p99"] for s in stats.values())
    return worst_error, stats, delta_pct


# In[ ]:


def evaluate(model, X_eval, Y_eval_scaled, scaler_Y, output_labels,
             transform="none", split_name="Val"):
    """Versión verbosa: imprime PASS/REVIEW por output."""
    Y_true, Y_pred, delta_pct = _compute_delta_pct(
        model, X_eval, Y_eval_scaled, scaler_Y, transform
    )
    stats = _compute_stats(delta_pct, output_labels)

    print(f"\n  {split_name} percentage difference (ΔY%):")
    for label, s in stats.items():
        status = "PASS" if s["p99"] < 1.0 else "REVIEW"
        print(f"    {label:>4s}:  median={s['median']:.4f}%  |  "
              f"std={s['std']:.4f}%  |  |95th|={s['p95']:.4f}%  |  "
              f"|99th|={s['p99']:.4f}%  [{status}]")

    return Y_true, Y_pred, delta_pct, stats


# # 5. Plotting

# In[ ]:


def plot_loss(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history.history["loss"],     "r--", lw=1.5, label="Train")
    ax.semilogy(history.history["val_loss"], "b-",  lw=1.5, label="Validation")
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("Loss",   fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Loss plot saved: {save_path}")


# In[ ]:


def plot_percentage_difference(delta_pct, output_labels, save_path):
    n_out = len(output_labels)
    ncols = min(n_out, 2)
    nrows = (n_out + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 9))
    axes_flat = np.array(axes).flat
    for ax in axes_flat:
        ax.set_visible(False)

    for i, (ax, label) in enumerate(zip(axes.flat, output_labels)):
        ax.set_visible(True)
        col    = delta_pct[:, i]
        median = np.median(col)
        std    = np.std(col)

        # --- Histogram ---
        ax.hist(col, bins=1000, color="steelblue", edgecolor="black",
                alpha=0.75, density=True)

        # Reference lines
        ax.axvline(0.0,    color="black", ls="-",  lw=1.2, label="Zero")
        ax.axvline(median, color="green", ls="-",  lw=1.8,
                   label=f"Median: {median:.4f}%")
        ax.axvline(median + std, color="orange", ls="--", lw=1.2,
                   label=f"±1σ: {std:.4f}%")
        ax.axvline(median - std, color="orange", ls="--", lw=1.2)

        # Symmetric x-limits so the distribution is centred
        xlim = max(np.percentile(np.abs(col), 99.5), 1e-6) * 1.2
        ax.set_xlim(-xlim, xlim)

        ax.set_xlabel(r"$\Delta$" + f"{label} (%)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Percentage difference plot saved: {save_path}")


# # 6. Ouput I/O: paths, artifacts, history

# In[ ]:


def group_paths(cfg, group_name):
    """Centraliza la convención de nombres de archivos por grupo."""
    return {
        "model"    : cfg["output_groups"][group_name]["model_file"],
        "scaler_X" : os.path.join(cfg["scalers_dir"], f"scaler_X_{group_name}.pkl"),
        "scaler_Y" : os.path.join(cfg["scalers_dir"], f"scaler_Y_{group_name}.pkl"),
        "loss_plot": os.path.join(cfg["plots_dir"],   f"loss_curves_{group_name}.pdf"),
        "diff_plot": os.path.join(cfg["plots_dir"],   f"percentage_difference_{group_name}.pdf"),
        "history"  : os.path.join(cfg["history_dir"], f"history_{group_name}.json"),
    }


# In[ ]:


def save_artifacts(model, scaler_X, scaler_Y, paths):
    os.makedirs(os.path.dirname(paths["scaler_X"]), exist_ok=True)
    model.save(paths["model"])
    joblib.dump(scaler_X, paths["scaler_X"])
    joblib.dump(scaler_Y, paths["scaler_Y"])
    print(f"  Model   → {paths['model']}")
    print(f"  Scalers → {paths['scaler_X']}, {paths['scaler_Y']}")


# In[ ]:


def save_history(history, path):
    serializable = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  History saved: {path}")


# # 7. Logging to file & terminal

# In[ ]:


def _write_group_header(log_path, group_name, group_cfg):
    with open(log_path, "a") as f:
        f.write(f"\n{'#'*70}\n")
        f.write(f"# GROUP: {group_name}  |  outputs: {group_cfg['labels']}\n")
        f.write(f"# Transform: {group_cfg['transform']}  |  "
                f"Success threshold: {group_cfg['success_threshold']}%\n")
        f.write(f"{'#'*70}\n\n")


# In[ ]:


def _write_group_summary(log_path, group_name, best, success):
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"BEST RUN SUMMARY [{group_name}]\n")
        if best["params"] is None:
            f.write("  No run completed.\n")
        else:
            n_layers, n_neurons, activation, lr, delta, batch_size = best["params"]
            f.write(f"  Layers={n_layers}  Neurons={n_neurons}  "
                    f"Activation={activation}  LR={lr}  Delta={delta}  "
                    f"Batch={batch_size}\n")
            f.write(f"  Worst p99 |ΔY%| (val):  {best['worst_error']:.6f}%\n")
            f.write(f"  Worst p99 |ΔY%| (test): {best['worst_test']:.6f}%\n")
            f.write(f"  Success: {success}\n")
        f.write(f"{'='*70}\n\n")


# In[ ]:


def _print_final_summary(summary):
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for group_name, info in summary.items():
        best = info["best"]
        if best["params"] is None:
            print(f"  [{group_name}] No run completed.")
            continue
        n_layers, n_neurons, activation, lr, delta, batch_size = best["params"]
        print(f"  [{group_name}] best: layers={n_layers} neurons={n_neurons} "
              f"act={activation} lr={lr} delta={delta} batch={batch_size}")
        print(f"    val  worst p99: {best['worst_error']:.4f}%")
        print(f"    test worst p99: {best['worst_test']:.4f}%")
        print(f"    success: {info['success']}")


# In[ ]:


def log_run(log_path, group_name, run_id, params, worst_error,
            stats, elapsed, is_best):
    n_layers, n_neurons, activation, lr, delta, batch_size = params

    with open(log_path, "a") as f:
        tag = " *** BEST ***" if is_best else ""
        f.write(f"{'='*70}\n")
        f.write(f"[{group_name}] Run {run_id}{tag}\n")
        f.write(f"  Layers={n_layers}  Neurons={n_neurons}  "
                f"Activation={activation}  LR={lr}  Delta={delta}  "
                f"Batch={batch_size}\n")
        f.write(f"  Training time: {elapsed:.2f}s\n")
        f.write(f"  Worst p99 |ΔY%|: {worst_error:.6f}%\n")
        for label, s in stats.items():
            f.write(f"    {label:>4s}: median={s['median']:.4f}%  "
                    f"std={s['std']:.4f}%  max={s['max_abs']:.4f}%  "
                    f"p95={s['p95']:.4f}%  p99={s['p99']:.4f}%\n")
        f.write("\n")


# # 8. Helpers

# In[ ]:


def _init_best_dict():
    return {
        "worst_error": np.inf,
        "worst_test" : np.inf,
        "params"     : None,
        "delta_pct"  : None,
        "history"    : None,
        "stats"      : None,
    }


# In[ ]:


def _run_single_search_iteration(cfg, group_name, group_cfg, params,
                                 X_train, Y_train, X_val, Y_val,
                                 X_test, Y_test, scaler_Y,
                                 current_best_worst):
    """Entrena un único modelo y devuelve un dict con todo lo necesario.

    NO muta nada externo. NO guarda artefactos. NO escribe logs.
    `search_group` es el responsable de decidir qué hacer con el resultado.
    """
    n_layers, n_neurons, activation, lr, delta, batch_size = params
    transform = group_cfg["transform"]

    # Re-seed por run: pesos iniciales reproducibles independientemente del
    # orden de exploración. NO mover ni quitar (ver docstring de set_seeds).
    set_seeds(cfg["seed"], deterministic=cfg["deterministic"])

    model, history, elapsed = train_one_model(
        cfg, group_name, group_cfg,
        n_layers=n_layers, n_neurons=n_neurons,
        activation=activation, lr=lr, delta=delta,
        batch_size=batch_size,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val,     Y_val=Y_val,
        scaler_Y=scaler_Y,
    )

    # Selección por VAL
    worst_val, stats_val, _ = automatic_evaluate(
        model, X_val, Y_val, scaler_Y,
        group_cfg["labels"], transform=transform,
    )

    is_best = worst_val < current_best_worst

    # Sólo evaluamos en TEST si el run es candidato a best (ahorro de cómputo)
    if is_best:
        worst_test, stats_test, delta_pct_test = automatic_evaluate(
            model, X_test, Y_test, scaler_Y,
            group_cfg["labels"], transform=transform,
        )
    else:
        worst_test, stats_test, delta_pct_test, model = None, None, None, None
        tf.keras.backend.clear_session()

    return {
        "model"         : model,
        "history"       : history,
        "elapsed"       : elapsed,
        "params"        : params,
        "worst_val"     : worst_val,
        "stats_val"     : stats_val,
        "is_best"       : is_best,
        "worst_test"    : worst_test,
        "stats_test"    : stats_test,
        "delta_pct_test": delta_pct_test,
    }


# In[ ]:


def generate_search_space(group_cfg, max_runs, seed):
    ss = group_cfg["search_space"]
    all_combos = list(itertools.product(
        ss["n_layers"],
        ss["n_neurons"],
        ss["activation"],
        ss["lr"],
        ss["delta"],
        ss["batch_size"],
    ))

    rng = np.random.RandomState(seed)
    rng.shuffle(all_combos)
    selected = all_combos[:max_runs]

    print(f"  Total combinations: {len(all_combos)}  |  Selected: {len(selected)}")
    return selected


# In[ ]:


def check_wall_time(start_time, wall_time, safety_margin):
    """Return True if there is enough time for another run."""
    elapsed = time.time() - start_time
    remaining = wall_time - elapsed
    return remaining > safety_margin


# In[ ]:


def search_group(cfg, group_name, group_cfg,
                 X_train, X_val, X_test,
                 Y_train, Y_val, Y_test,
                 scaler_X, scaler_Y,
                 log_path, job_start):

    print(f"\n{'='*70}")
    print(f"SEARCH GROUP: {group_name}  |  outputs: {group_cfg['labels']}  "
          f"|  transform: {group_cfg['transform']}")
    print(f"{'='*70}")

    print("\nGenerating search space...")
    combos = generate_search_space(group_cfg, group_cfg["max_runs"], cfg["seed"])

    paths = group_paths(cfg, group_name)
    os.makedirs(cfg["scalers_dir"], exist_ok=True)
    os.makedirs(cfg["plots_dir"],   exist_ok=True)
    os.makedirs(cfg["history_dir"], exist_ok=True)

    _write_group_header(log_path, group_name, group_cfg)

    best    = _init_best_dict()
    success = False

    for run_id, params in enumerate(combos, start=1):
        if not check_wall_time(job_start, cfg["wall_time_seconds"],
                               cfg["safety_margin"]):
            print(f"\n[{group_name}] Wall-time safety margin reached. Stopping.")
            break

        n_layers, n_neurons, activation, lr, delta, batch_size = params
        print(f"\n--- [{group_name}] Run {run_id}/{len(combos)}: "
              f"layers={n_layers} neurons={n_neurons} act={activation} "
              f"lr={lr} delta={delta} batch={batch_size} ---")

        result = _run_single_search_iteration(
            cfg, group_name, group_cfg, params,
            X_train, Y_train, X_val, Y_val,
            X_test,  Y_test,
            scaler_Y,
            current_best_worst=best["worst_error"],
        )

        # search_group es el ÚNICO responsable de actualizar best y persistir
        if result["is_best"]:
            best.update({
                "worst_error": result["worst_val"],
                "worst_test" : result["worst_test"],
                "params"     : result["params"],
                "delta_pct"  : result["delta_pct_test"],
                "history"    : result["history"],
                "stats"      : result["stats_test"],
            })
            save_artifacts(result["model"], scaler_X, scaler_Y, paths)
            tf.keras.backend.clear_session()
            print(f"  ✅ New best (val)! worst p99={result['worst_val']:.4f}%  "
                  f"|  test={result['worst_test']:.4f}%")
        else:
            print(f"  worst p99 (val) = {result['worst_val']:.4f}%")

        log_run(log_path, group_name, run_id, params,
                result["worst_val"], result["stats_val"],
                result["elapsed"], result["is_best"])

        if result["worst_val"] < group_cfg["success_threshold"]:
            print(f"\n🎯 [{group_name}] SUCCESS: "
                  f"worst p99 < {group_cfg['success_threshold']}%")
            success = True
            break

    # Plots y history del best
    if best["params"] is not None:
        plot_loss(best["history"], paths["loss_plot"])
        plot_percentage_difference(best["delta_pct"],
                                   group_cfg["labels"],
                                   paths["diff_plot"])
        save_history(best["history"], paths["history"])

    _write_group_summary(log_path, group_name, best, success)

    return best, success


# # 9. Entry points

# In[ ]:


def main(cfg):
    set_seeds(cfg["seed"], deterministic=cfg["deterministic"])
    configure_gpu(cfg["use_mixed_precision"])

    print("\nLoading data...")
    raw, X_full = load_data(cfg["data_file"], cfg["input_cols"])

    os.makedirs(cfg["plots_dir"],   exist_ok=True)
    os.makedirs(cfg["scalers_dir"], exist_ok=True)
    os.makedirs(cfg["history_dir"], exist_ok=True)

    for group_name, group_cfg in cfg["output_groups"].items():
        print(f"\n{'='*70}")
        print(f"GROUP: {group_name}  |  outputs: {group_cfg['labels']}  "
              f"|  transform: {group_cfg['transform']}")
        print(f"{'='*70}")

        Y_full = raw[:, group_cfg["cols"]]

        print("\nPreprocessing...")
        (X_train, X_val, X_test,
         Y_train, Y_val, Y_test,
         scaler_X, scaler_Y) = preprocess(
            X_full, Y_full,
            val_size=cfg["val_size"],
            test_size=cfg["test_size"],
            seed=cfg["seed"],
            transform=group_cfg["transform"],
        )

        print("\nTraining...")
        model, history, elapsed = train_one_model(
            cfg, group_name, group_cfg,
            n_layers   =group_cfg["n_layers"],
            n_neurons  =group_cfg["n_neurons"],
            activation =group_cfg["activation"],
            lr         =group_cfg["learning_rate"],
            delta      =group_cfg["delta"],
            batch_size =cfg["batch_size"],
            X_train=X_train, Y_train=Y_train,
            X_val=X_val,     Y_val=Y_val,
            scaler_Y=scaler_Y,
        )

        print("\nEvaluating on VAL set:")
        evaluate(model, X_val, Y_val, scaler_Y, group_cfg["labels"],
                 transform=group_cfg["transform"], split_name="Val")

        print("\nEvaluating on TEST set (held-out, publication metric):")
        _, _, delta_pct_test, _ = evaluate(
            model, X_test, Y_test, scaler_Y, group_cfg["labels"],
            transform=group_cfg["transform"], split_name="Test",
        )

        paths = group_paths(cfg, group_name)
        plot_loss(history, paths["loss_plot"])
        plot_percentage_difference(delta_pct_test, group_cfg["labels"],
                                   paths["diff_plot"])
        save_history(history, paths["history"])
        save_artifacts(model, scaler_X, scaler_Y, paths)

    print("\nDone.")


# In[ ]:


def run_search(cfg):
    set_seeds(cfg["seed"], deterministic=cfg["deterministic"])
    configure_gpu(cfg["use_mixed_precision"])

    print("\nLoading data...")
    raw, X_full = load_data(cfg["data_file"], cfg["input_cols"])

    log_path = cfg["log_file"]
    with open(log_path, "w") as f:
        f.write("Hyperparameter Search Log (multi-group)\n")
        f.write(f"Seed: {cfg['seed']}\n")
        f.write(f"Deterministic: {cfg['deterministic']}\n")
        f.write(f"Mixed precision: {cfg['use_mixed_precision']}\n")
        f.write(f"{'='*70}\n\n")

    job_start = time.time()
    summary   = {}

    for group_name, group_cfg in cfg["output_groups"].items():
        Y_full = raw[:, group_cfg["cols"]]

        print(f"\nPreprocessing for [{group_name}]...")
        (X_train, X_val, X_test,
         Y_train, Y_val, Y_test,
         scaler_X, scaler_Y) = preprocess(
            X_full, Y_full,
            val_size=cfg["val_size"],
            test_size=cfg["test_size"],
            seed=cfg["seed"],
            transform=group_cfg["transform"],
        )

        best, success = search_group(
            cfg, group_name, group_cfg,
            X_train, X_val, X_test,
            Y_train, Y_val, Y_test,
            scaler_X, scaler_Y,
            log_path, job_start,
        )
        summary[group_name] = {"best": best, "success": success}

        if not check_wall_time(job_start, cfg["wall_time_seconds"],
                               cfg["safety_margin"]):
            print(f"\nWall-time exhausted. Skipping remaining groups.")
            break

    _print_final_summary(summary)
    print("\nDone.")


# # 10.  Script routine

# In[ ]:


if __name__ == "__main__":
    # Elige UNO. Cada config vive en su propio JSON.
    cfg = load_config("automatic_config.json")
    run_search(cfg)

    # Para entrenamiento único:
    # cfg = load_config("single_config.json")
    # main(cfg)

