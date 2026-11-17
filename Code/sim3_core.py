"""
sim3_core.py -- building blocks for Section 3 (required simulations).

Data sets, preprocessing, the training loop with clean evaluation and timing,
and a paired SGD / R-SGD runner.  Uses rsgd.py from Section 2 for the
uncertainty estimate (4), the robust step and the gradient oracles.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

import rsgd as R

MASTER3 = 20261118

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = {
    "logreg": dict(p=3, loss=R.logreg_loss, logits=R.logreg_logits,
                   grads=R.logreg_per_example_grads,
                   init=lambda rng: np.zeros(3)),
    "mlp": dict(p=R.P_MLP, loss=R.mlp_loss, logits=R.mlp_logits,
                grads=R.mlp_per_example_grads, init=R.mlp_init),
}
DATA = {"logreg": R.make_gaussian_blobs, "mlp": R.make_semicircles}


def seed_streams(sim_id: int, kind: int, seed: int):
    """Four independent streams (data, init, batch, noise) for one seed.
    kind = 0: tuning seeds, 1: evaluation seeds (A), 2: training-only seeds (B)."""
    ss = np.random.SeedSequence([MASTER3, sim_id, kind, seed])
    return [int(c.generate_state(1)[0]) for c in ss.spawn(4)]


@dataclass
class Split:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def make_split(model, data_seed, n_tr=400, n_va=400, n_te=4000):
    """Balanced train / validation / test sets drawn independently; inputs
    standardised with the TRAINING mean and standard deviation only."""
    rng = np.random.default_rng(data_seed)
    gen = DATA[model]
    Xtr, ytr = gen(n_tr, rng)
    Xva, yva = gen(n_va, rng)
    Xte, yte = gen(n_te, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    z = lambda X: (X - mu) / sd
    return Split(z(Xtr), ytr, z(Xva), yva, z(Xte), yte, mu, sd)


def accuracy(model, theta, X, y):
    return float(np.mean((MODELS[model]["logits"](theta, X) > 0) == (y == 1)))


def grad_scale(model, split, theta0):
    """RMS over examples and coordinates of the clean per-example gradients
    at the initial point -- the unit in which noise levels are expressed."""
    U = MODELS[model]["grads"](theta0, split.Xtr, split.ytr)
    return float(np.sqrt(np.mean(U ** 2)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass
class Run:
    theta: np.ndarray
    ks: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    zero_freq: float
    step_ratio: float          # mean ||d_k|| / (alpha_k ||g_k||), zero steps count as 0
    newton_iters: float
    update_time: float         # seconds spent in the K updates (no evaluation)
    grad_evals: int
    failed: bool


def train(model, split, theta0, method, alpha0, rho, sigma, b, K,
          seed_batch, seed_noise, n_log=60, fail_factor=50.0):
    """K updates of SGD or R-SGD.  Per update: b indices i.i.d. with
    replacement; clean per-example gradients; corruption sigma*z added to the
    observations only; g_k (and S_k for R-SGD) from those observations.
    Losses are evaluated on clean training / validation data at n_log+1
    checkpoints; evaluation time is excluded from update_time."""
    M = MODELS[model]
    grads, loss = M["grads"], M["loss"]
    X, y = split.Xtr, split.ytr
    n = len(y)
    rb = np.random.default_rng(seed_batch)
    rn = np.random.default_rng(seed_noise)
    theta = np.array(theta0, dtype=float)
    log_at = set(np.unique(np.linspace(0, K, n_log + 1).astype(int)).tolist())
    ks, trl, val = [], [], []
    l0 = loss(theta, X, y)
    nzero, ratio_sum, it_sum, n_nz = 0, 0.0, 0, 0
    t_upd = 0.0
    failed = False
    for k in range(K + 1):
        if k in log_at:
            ks.append(k)
            trl.append(loss(theta, X, y))
            val.append(loss(theta, split.Xva, split.yva))
            if not np.isfinite(trl[-1]) or trl[-1] > fail_factor * l0:
                failed = True
                break
        if k == K:
            break
        t0 = time.perf_counter()
        idx = rb.integers(0, n, size=b)
        V = grads(theta, X[idx], y[idx])
        if sigma > 0:
            V = V + sigma * rn.standard_normal(V.shape)
        alpha = alpha0 / np.sqrt(1.0 + k / 50.0)
        if method == "sgd":
            g = V.mean(0)
            d = -alpha * g
        else:
            g, s, _, _ = R.uncertainty_diag(V)
            d, info = R.robust_step(g, s, alpha, rho)
            if info.zero:
                nzero += 1
            else:
                it_sum += info.iters
                n_nz += 1
        theta = theta + d
        t_upd += time.perf_counter() - t0
        gn = np.linalg.norm(g)
        ratio_sum += np.linalg.norm(d) / (alpha * gn) if gn > 0 else 0.0
    if failed:
        # pad the curves so that every run has the same length
        while len(ks) < len(log_at):
            ks.append(sorted(log_at)[len(ks)])
            trl.append(np.inf)
            val.append(np.inf)
    return Run(theta, np.array(ks), np.array(trl), np.array(val),
               nzero / K, ratio_sum / K, it_sum / max(n_nz, 1), t_upd, b * K, failed)


def summary(model, split, run):
    """Final clean metrics of one run."""
    loss = MODELS[model]["loss"]
    return dict(train_loss=float(loss(run.theta, split.Xtr, split.ytr)),
                val_loss=float(loss(run.theta, split.Xva, split.yva)),
                test_loss=float(loss(run.theta, split.Xte, split.yte)),
                test_acc=accuracy(model, run.theta, split.Xte, split.yte),
                zero_freq=run.zero_freq, step_ratio=run.step_ratio,
                newton_iters=run.newton_iters,
                update_ms=run.update_time * 1e3, grad_evals=run.grad_evals,
                failed=run.failed)
