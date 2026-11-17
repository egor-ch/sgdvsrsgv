"""
rsgd.py -- SGD versus Robust SGD (R-SGD) under gradient uncertainty.

Section 2 of the homework mini-project: construction of g_k and the diagonal
uncertainty matrix S_k (formula (4)), a specialised solver for the robust
displacement subproblem (2), the SGD / R-SGD steps, gradient-observation
oracles (noisy quadratic, logistic regression, two-layer tanh network), and
finite-difference gradient checks.

All computations are in float64.  Only NumPy and SciPy are required.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

# ---------------------------------------------------------------------------
# 1. Gradient estimate and diagonal uncertainty matrix, formula (4)
# ---------------------------------------------------------------------------


def uncertainty_diag(V: np.ndarray):
    """Build g_k and S_k = diag(s_k) from the b x p matrix of observations V.

    Returns
    -------
    g     : (p,) batch mean  g_k = (1/b) sum_i v_{k,i}
    s     : (p,) diagonal of S_k,  s_{k,j} = shat_{k,j} + eps_k
    shat  : (p,) unregularised variance estimate of the batch mean
    eps   : float, the floor eps_k = 1e-3 * max(mean_j shat_{k,j}, 1e-8)
    """
    V = np.asarray(V, dtype=float)
    b, p = V.shape
    if b < 2:
        raise ValueError("formula (4) needs b >= 2 observations")
    g = V.mean(axis=0)
    shat = ((V - g) ** 2).sum(axis=0) / (b * (b - 1))
    eps = 1e-3 * max(shat.mean(), 1e-8)
    return g, shat + eps, shat, eps


# ---------------------------------------------------------------------------
# 2. Solver for the robust subproblem (2) with diagonal S
# ---------------------------------------------------------------------------


def robust_objective(d, g, s, alpha, rho):
    """phi(d) = g^T d + rho*sqrt(d^T S d) + ||d||^2/(2 alpha),  S = diag(s)."""
    d = np.asarray(d, dtype=float)
    return g @ d + rho * np.sqrt(np.sum(s * d * d)) + (d @ d) / (2.0 * alpha)


def optimality_residual(d, g, s, alpha, rho):
    """Relative residual of the 1(d) condition
    || g + rho*S d / sqrt(d^T S d) + d/alpha || / ||g||   (only for d != 0)."""
    tau = np.sqrt(np.sum(s * d * d))
    r = g + rho * s * d / tau + d / alpha
    return np.linalg.norm(r) / np.linalg.norm(g)


@dataclass
class StepInfo:
    zero: bool          # True if the zero-step rule (3) fired
    kappa2: float       # g^T S^{-1} g  (squared Mahalanobis norm of g)
    iters: int          # Newton/bisection iterations used (0 for closed cases)
    tau: float          # sqrt(d^T S d) at the solution
    psi: float          # final value of the secular function psi(tau)
    bisections: int = 0  # how many safeguard bisection steps were taken


def robust_step(g, s, alpha, rho, tol=1e-14, maxit=100):
    """Solve  d* = argmin_d  g^T d + rho*sqrt(d^T S d) + ||d||^2/(2 alpha),
    S = diag(s) > 0, alpha > 0, rho >= 0.

    Method (see report, Section "Solver"):
      * rho == 0                      ->  d = -alpha g          (SGD)
      * g^T S^{-1} g <= rho^2         ->  d = 0                 (rule (3))
      * otherwise d_j(tau) = -alpha*tau*g_j/(tau + alpha*rho*s_j), where
        tau = sqrt(d^T S d) > 0 is the unique root of the secular equation
            q(tau) = sum_j alpha^2 s_j g_j^2 / (tau + alpha rho s_j)^2 = 1 .
        We apply Newton's method to the concave increasing function
            psi(tau) = q(tau)^{-1/2} - 1,
        started at tau_0 = 0 (psi(0) < 0), safeguarded by bisection on the
        bracket [0, alpha*sqrt(g^T S g)].  Stop when |psi| <= tol or the
        relative change of tau is <= 1e-15, or after maxit iterations.
    Cost: O(p) per iteration.
    """
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    kappa2 = float(np.sum(g * g / s))
    if rho == 0.0:
        d = -alpha * g
        return d, StepInfo(False, kappa2, 0, float(np.sqrt(np.sum(s * d * d))), 0.0)
    if kappa2 <= rho * rho:                      # zero-step rule, part 1(c)
        return np.zeros_like(g), StepInfo(True, kappa2, 0, 0.0, 0.0)

    a = alpha * alpha * s * g * g                # numerators
    c = alpha * rho * s                          # shifts (all > 0)

    def psi_and_deriv(tau):
        den = tau + c
        q = np.sum(a / den ** 2)
        dq = -2.0 * np.sum(a / den ** 3)
        psi = q ** -0.5 - 1.0
        dpsi = -0.5 * q ** -1.5 * dq
        return psi, dpsi

    lo, hi = 0.0, alpha * np.sqrt(np.sum(s * g * g))   # psi(lo) < 0 <= psi(hi)
    tau = 0.0
    psi, dpsi = psi_and_deriv(tau)
    it = nbis = 0
    while it < maxit:
        it += 1
        tau_new = tau - psi / dpsi                     # Newton step
        if not (lo < tau_new <= hi) or not np.isfinite(tau_new):
            tau_new = 0.5 * (lo + hi)                  # safeguard: bisection
            nbis += 1
        step = abs(tau_new - tau)
        tau = tau_new
        psi, dpsi = psi_and_deriv(tau)
        if psi < 0:
            lo = tau
        else:
            hi = tau
        if abs(psi) <= tol or step <= 1e-15 * tau:
            break
    d = -alpha * tau * g / (tau + c)
    return d, StepInfo(False, kappa2, it, float(tau), float(psi), nbis)


def worst_case_gradient(d, g, s, rho):
    """Maximiser of the inner problem in (1) for d != 0 (part 1(a)):
    h* = g + rho * S d / sqrt(d^T S d)."""
    tau = np.sqrt(np.sum(s * d * d))
    return g + rho * s * d / tau


# ---------------------------------------------------------------------------
# 3. Optimisers
# ---------------------------------------------------------------------------


def lr_schedule(alpha0, k, kind="sqrt"):
    """alpha_k = alpha0 / sqrt(1 + k/50) (the example schedule of Section 4)."""
    if kind == "const":
        return alpha0
    return alpha0 / np.sqrt(1.0 + k / 50.0)


@dataclass
class RunLog:
    theta: list = field(default_factory=list)      # iterates theta_0..theta_K
    zero: list = field(default_factory=list)       # zero-step flags
    kappa2: list = field(default_factory=list)
    iters: list = field(default_factory=list)
    residual: list = field(default_factory=list)   # 1(d) residual (nonzero steps)
    grad_evals: int = 0                             # per-example gradient evals
    runtime: float = 0.0


def run_optimizer(observe, theta0, method, alpha0, rho, K, seed_batch, seed_noise,
                  schedule="sqrt"):
    """Run K iterations of SGD or R-SGD.

    observe(theta, rng_batch, rng_noise) -> V (b x p) of gradient observations.
    Both methods are driven by fresh generators built from the SAME seeds, so
    a paired SGD / R-SGD comparison uses identical batch indices and identical
    noise draws z_{k,i} at every iteration (common random numbers).
    """
    rb = np.random.default_rng(seed_batch)
    rn = np.random.default_rng(seed_noise)
    theta = np.array(theta0, dtype=float)
    log = RunLog()
    log.theta.append(theta.copy())
    t0 = time.perf_counter()
    for k in range(K):
        V = observe(theta, rb, rn)
        log.grad_evals += V.shape[0]
        g, s, _, _ = uncertainty_diag(V)
        alpha = lr_schedule(alpha0, k, schedule)
        if method == "sgd":
            d = -alpha * g
            log.zero.append(False)
            log.kappa2.append(float(np.sum(g * g / s)))
            log.iters.append(0)
        elif method == "rsgd":
            d, info = robust_step(g, s, alpha, rho)
            log.zero.append(info.zero)
            log.kappa2.append(info.kappa2)
            log.iters.append(info.iters)
            if not info.zero and rho > 0:
                log.residual.append(optimality_residual(d, g, s, alpha, rho))
        else:
            raise ValueError(method)
        theta = theta + d
        log.theta.append(theta.copy())
    log.runtime = time.perf_counter() - t0
    log.theta = np.array(log.theta)
    log.zero = np.array(log.zero)
    return log


# ---------------------------------------------------------------------------
# 4. Gradient-observation oracles
# ---------------------------------------------------------------------------

# 4.1 Illustration: noisy quadratic  f = 0.5*(th1^2 + 4 th2^2)
QUAD_H = np.array([1.0, 4.0])          # Hessian diagonal  -> L = 4, mu = 1
QUAD_A = np.array([2.0, 0.5])          # noise scaling matrix A = diag(2, 1/2)


def quad_f(theta):
    theta = np.asarray(theta)
    return 0.5 * np.sum(QUAD_H * theta ** 2, axis=-1)


def quad_grad(theta):
    return QUAD_H * np.asarray(theta)


def make_quad_observe(b, sigma):
    """v_{k,i} = grad f(theta_k) + sigma * A z_{k,i},  z ~ N(0, I_2).
    rng_batch is unused (no data set), but kept for a uniform interface."""
    def observe(theta, rng_batch, rng_noise):
        Z = rng_noise.standard_normal((b, 2))
        return quad_grad(theta)[None, :] + sigma * Z * QUAD_A[None, :]
    return observe


# 4.2 Logistic regression  t = w^T a + c   (p = 3)
def bce_from_logits(t, y):
    """Stable l(t, y) = log(1 + e^t) - y t."""
    return np.logaddexp(0.0, t) - y * t


def logreg_logits(theta, X):
    return X @ theta[:2] + theta[2]


def logreg_loss(theta, X, y):
    return bce_from_logits(logreg_logits(theta, X), y).mean()


def logreg_per_example_grads(theta, X, y):
    """u_i = (sigmoid(t_i) - y_i) * (a_i, 1);  returns b x 3."""
    delta = expit(logreg_logits(theta, X)) - y
    return np.column_stack([delta[:, None] * X, delta])


# 4.3 Two-layer tanh network  t = w2^T tanh(W1 a + b1) + b2   (p = 33)
H_MLP = 8
P_MLP = H_MLP * 2 + H_MLP + H_MLP + 1    # = 33


def mlp_unpack(theta):
    W1 = theta[:16].reshape(H_MLP, 2)
    b1 = theta[16:24]
    w2 = theta[24:32]
    b2 = theta[32]
    return W1, b1, w2, b2


def mlp_logits(theta, X):
    W1, b1, w2, b2 = mlp_unpack(theta)
    return np.tanh(X @ W1.T + b1) @ w2 + b2


def mlp_loss(theta, X, y):
    return bce_from_logits(mlp_logits(theta, X), y).mean()


def mlp_per_example_grads(theta, X, y):
    """Per-example back-propagation, vectorised over the batch; b x 33.
    delta_i = sigmoid(t_i) - y_i,  dW2 = delta h,  db2 = delta,
    dz = delta * w2 * (1 - h^2),  dW1 = dz a^T,  db1 = dz."""
    W1, b1, w2, b2 = mlp_unpack(theta)
    Hh = np.tanh(X @ W1.T + b1)                    # b x 8
    delta = expit(Hh @ w2 + b2) - y                 # b
    dZ = delta[:, None] * w2[None, :] * (1.0 - Hh ** 2)   # b x 8
    dW1 = (dZ[:, :, None] * X[:, None, :]).reshape(len(y), 16)
    return np.column_stack([dW1, dZ, delta[:, None] * Hh, delta])


def mlp_init(rng):
    """Documented initialisation: W1 ~ N(0, 1/2) (fan-in 2), b1 = 0,
    w2 ~ N(0, 1/8) (fan-in 8), b2 = 0."""
    W1 = rng.normal(0.0, np.sqrt(1 / 2), size=(H_MLP, 2))
    w2 = rng.normal(0.0, np.sqrt(1 / 8), size=H_MLP)
    return np.concatenate([W1.ravel(), np.zeros(H_MLP), w2, [0.0]])


def make_data_observe(per_example_grads, X, y, b, sigma):
    """Learning models: sample b indices i.i.d. WITH replacement, compute the
    clean per-example gradients u_i(theta) and add sigma * z_{k,i}.
    The corruption touches only the observations V; the data X, y and every
    loss / accuracy evaluation stay clean."""
    n = len(y)

    def observe(theta, rng_batch, rng_noise):
        idx = rng_batch.integers(0, n, size=b)
        U = per_example_grads(theta, X[idx], y[idx])
        if sigma > 0:
            U = U + sigma * rng_noise.standard_normal(U.shape)
        return U
    return observe


# 4.4 Data generators (balanced classes)
def make_gaussian_blobs(n, rng):
    """a | y ~ N((2y-1)(1,1)^T, I_2), n/2 points per class."""
    y = np.repeat([0.0, 1.0], n // 2)
    X = (2 * y - 1)[:, None] * np.ones(2)[None, :] + rng.standard_normal((len(y), 2))
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_semicircles(n, rng, noise=0.1):
    """Two interleaving noisy semicircles, t ~ U[0, pi], e ~ N(0, 0.1^2 I)."""
    y = np.repeat([0.0, 1.0], n // 2)
    t = rng.uniform(0.0, np.pi, size=len(y))
    X = np.where(y[:, None] == 0,
                 np.column_stack([np.cos(t), np.sin(t)]),
                 np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]))
    X = X + noise * rng.standard_normal(X.shape)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# 5. Finite-difference gradient check
# ---------------------------------------------------------------------------


def fd_check(loss, per_example_grads, theta, X, y, h=1e-6):
    """Compare the mean of analytic per-example gradients with a central
    finite-difference gradient of the mean loss, and each per-example
    gradient with central differences of the single-example loss.
    Returns (rel. error of mean gradient, max rel. error per example)."""
    p = theta.size
    U = per_example_grads(theta, X, y)
    E = np.eye(p)
    fd_mean = np.array([(loss(theta + h * E[j], X, y) - loss(theta - h * E[j], X, y))
                        / (2 * h) for j in range(p)])
    err_mean = np.linalg.norm(U.mean(0) - fd_mean) / np.linalg.norm(fd_mean)
    errs = []
    for i in range(min(len(y), 20)):
        Xi, yi = X[i:i + 1], y[i:i + 1]
        fd_i = np.array([(loss(theta + h * E[j], Xi, yi) - loss(theta - h * E[j], Xi, yi))
                         / (2 * h) for j in range(p)])
        errs.append(np.linalg.norm(U[i] - fd_i) / max(np.linalg.norm(fd_i), 1e-12))
    return err_mean, max(errs)
