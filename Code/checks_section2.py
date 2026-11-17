"""
checks_section2.py -- every numerical check of Section 2 of the homework.

Run:  python checks_section2.py
Writes: results_section2.json (all numbers quoted in the report) and the
figures fig/*.pdf.  Master seed 20261117; every experiment derives its own
independent stream from numpy.random.SeedSequence, so any single check can be
re-run on its own and gives the same numbers.
"""
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

import rsgd as R

MASTER = 20261117
SS = np.random.SeedSequence(MASTER)
(SEED_C0, SEED_C1, SEED_C2, SEED_C3, SEED_C4,
 SEED_C5, SEED_C6, SEED_C7) = SS.spawn(8)

# validated categorical palette (slots 1-4) + text inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#777777"
plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "savefig.bbox": "tight", "figure.dpi": 150,
})

RES = {"master_seed": MASTER,
       "versions": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "matplotlib": matplotlib.__version__}}


def rand_problem(rng, p, decades=6):
    """Random anisotropic instance: g ~ N(0, I), s log-uniform over `decades`."""
    g = rng.standard_normal(p)
    s = 10.0 ** rng.uniform(-decades / 2, decades / 2, p)
    alpha = 10.0 ** rng.uniform(-2, 0)
    return g, s, alpha


# ===========================================================================
# C0. Gradient oracles of the two learning models (finite differences)
# ===========================================================================
rng = np.random.default_rng(SEED_C0)
Xl, yl = R.make_gaussian_blobs(200, rng)
Xm, ym = R.make_semicircles(200, rng)
c0 = {"logreg": [], "mlp": []}
for _ in range(5):
    th = rng.standard_normal(3)
    c0["logreg"].append(R.fd_check(R.logreg_loss, R.logreg_per_example_grads, th, Xl, yl))
    th = R.mlp_init(rng)
    c0["mlp"].append(R.fd_check(R.mlp_loss, R.mlp_per_example_grads, th, Xm, ym))
RES["C0_fd"] = {k: {"max_rel_err_mean_grad": float(max(e[0] for e in v)),
                    "max_rel_err_per_example": float(max(e[1] for e in v))}
                for k, v in c0.items()}

# ===========================================================================
# C1. Solver: accuracy, iterations, CVXPY cross-check, certificate, cost
# ===========================================================================
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore", module="cvxpy")
RES["versions"]["cvxpy"] = cp.__version__

rng = np.random.default_rng(SEED_C1)
c1 = {}
# (i) iterations and residuals on hard random anisotropic problems
iters, resid, nbis, minimax, monotone_viol = [], [], 0, [], 0
for trial in range(20000):
    p = int(rng.choice([2, 3, 33, 1000]))
    g, s, alpha = rand_problem(rng, p)
    kappa = np.sqrt(np.sum(g * g / s))
    rho = kappa * rng.uniform(0.001, 0.999)
    d, info = R.robust_step(g, s, alpha, rho)
    iters.append(info.iters)
    nbis += info.bisections
    resid.append(R.optimality_residual(d, g, s, alpha, rho))
    h = R.worst_case_gradient(d, g, s, rho)
    minimax.append(np.linalg.norm(d + alpha * h) / np.linalg.norm(d))
iters = np.array(iters)
c1["n_problems"] = 20000
c1["iters_mean"] = float(iters.mean())
c1["iters_median"] = float(np.median(iters))
c1["iters_p99"] = float(np.percentile(iters, 99))
c1["iters_max"] = int(iters.max())
c1["iters_hist"] = {int(k): int(v) for k, v in zip(*np.unique(iters, return_counts=True))}
c1["bisection_steps_total"] = int(nbis)
c1["residual_max"] = float(np.max(resid))
c1["minimax_saddle_max"] = float(np.max(minimax))

# (ii) monotonicity of unsafeguarded Newton (concavity of psi)
for trial in range(5000):
    p = int(rng.choice([2, 3, 33]))
    g, s, alpha = rand_problem(rng, p)
    kappa = np.sqrt(np.sum(g * g / s))
    rho = kappa * rng.uniform(0.001, 0.999)
    a = alpha ** 2 * s * g * g
    c = alpha * rho * s
    tau = 0.0
    for _ in range(60):
        q = np.sum(a / (tau + c) ** 2)
        dq = -2 * np.sum(a / (tau + c) ** 3)
        psi = q ** -0.5 - 1
        if abs(psi) < 1e-14:
            break
        tau_new = tau - psi / (-0.5 * q ** -1.5 * dq)
        if tau_new < tau * (1 - 1e-14):
            monotone_viol += 1
        tau = tau_new
        if psi > 1e-12:            # an iterate to the right of the root
            monotone_viol += 1
c1["newton_monotonicity_violations"] = monotone_viol

# (iii) CVXPY (Clarabel, SOCP) and strong-convexity certificate
cv_dd, cv_obj, cert_min, cv_status = [], [], np.inf, []
for trial in range(300):
    p = int(rng.choice([2, 3, 33]))
    g, s, alpha = rand_problem(rng, p, decades=4)
    kappa = np.sqrt(np.sum(g * g / s))
    rho = kappa * rng.uniform(0.05, 0.95)
    d, _ = R.robust_step(g, s, alpha, rho)
    x, t = cp.Variable(p), cp.Variable()
    prob = cp.Problem(cp.Minimize(g @ x + rho * t + cp.sum_squares(x) / (2 * alpha)),
                      [cp.norm(cp.multiply(np.sqrt(s), x), 2) <= t])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12,
               tol_feas=1e-12)
    dc = x.value
    cv_status.append(prob.status)
    phi = R.robust_objective(d, g, s, alpha, rho)
    phic = R.robust_objective(dc, g, s, alpha, rho)
    cv_dd.append(np.linalg.norm(d - dc) / np.linalg.norm(d))
    cv_obj.append((phi - phic) / abs(phi))
    # certificate: phi(d+delta) - phi(d) >= ||delta||^2/(2 alpha)
    for scale in [1e-6, 1e-3, 1e-1, 1.0, 10.0]:
        for _ in range(10):
            delta = scale * np.linalg.norm(d) * rng.standard_normal(p) / np.sqrt(p)
            gap = (R.robust_objective(d + delta, g, s, alpha, rho) - phi
                   - delta @ delta / (2 * alpha))
            cert_min = min(cert_min, gap / max(abs(phi), 1e-300))
c1["cvxpy_rel_diff_d_max"] = float(np.max(cv_dd))
c1["cvxpy_rel_diff_d_median"] = float(np.median(cv_dd))
c1["cvxpy_n_not_optimal"] = int(sum(st != "optimal" for st in cv_status))
c1["cvxpy_rel_obj_ours_minus_cvx_max"] = float(np.max(cv_obj))
c1["certificate_min_rel_gap"] = float(cert_min)

# (iv) cost: solver time vs p, compared with the per-example gradient cost
timing = {}
for p in [3, 33, 1000, 100000]:
    g, s, alpha = rand_problem(rng, p)
    rho = 0.5 * np.sqrt(np.sum(g * g / s))
    reps = 2000 if p < 1e4 else 50
    t0 = time.perf_counter()
    for _ in range(reps):
        R.robust_step(g, s, alpha, rho)
    timing[p] = (time.perf_counter() - t0) / reps * 1e6
b = 32
th = R.mlp_init(rng)
t0 = time.perf_counter()
for _ in range(2000):
    U = R.mlp_per_example_grads(th, Xm[:b], ym[:b])
t_grad = (time.perf_counter() - t0) / 2000 * 1e6
t0 = time.perf_counter()
for _ in range(2000):
    R.uncertainty_diag(U)
t_S = (time.perf_counter() - t0) / 2000 * 1e6
c1["time_us_solver_vs_p"] = {str(k): float(v) for k, v in timing.items()}
c1["time_us_mlp_grads_b32"] = float(t_grad)
c1["time_us_build_S_b32_p33"] = float(t_S)
RES["C1_solver"] = c1

# ===========================================================================
# C2. Zero-radius limit
# ===========================================================================
rng = np.random.default_rng(SEED_C2)
c2 = {}
exact = 0
for _ in range(1000):
    g, s, alpha = rand_problem(rng, int(rng.integers(1, 50)))
    d, _ = R.robust_step(g, s, alpha, 0.0)
    exact += int(np.array_equal(d, -alpha * g))
c2["rho0_bitwise_equal"] = f"{exact}/1000"
rhos = 10.0 ** -np.arange(1, 9)
curves = []
for _ in range(5):
    g, s, alpha = rand_problem(rng, 33, decades=4)
    dev = []
    for r in rhos:
        d, _ = R.robust_step(g, s, alpha, r)
        dev.append(np.linalg.norm(d + alpha * g) / (alpha * np.linalg.norm(g)))
    curves.append(dev)
curves = np.array(curves)
slopes = [np.polyfit(np.log10(rhos), np.log10(cv), 1)[0] for cv in curves]
c2["loglog_slopes"] = [float(x) for x in slopes]
c2["dev_at_rho_1e-8_max"] = float(curves[:, -1].max())
# trajectory: SGD vs R-SGD(rho=0), identical seeds
obs = R.make_quad_observe(8, 1.0)
la = R.run_optimizer(obs, [2, 2], "sgd", 0.2, 0.0, 300, 1, 2)
lb = R.run_optimizer(obs, [2, 2], "rsgd", 0.2, 0.0, 300, 1, 2)
c2["trajectory_max_abs_diff"] = float(np.max(np.abs(la.theta - lb.theta)))
RES["C2_zero_radius"] = c2

# ===========================================================================
# C3. Isotropic closed form
# ===========================================================================
rng = np.random.default_rng(SEED_C3)
errs, agree, it3, n_zero = [], 0, [], 0
N3 = 10000
for _ in range(N3):
    p = int(rng.integers(1, 51))
    g = rng.standard_normal(p)
    sv = 10.0 ** rng.uniform(-4, 2)
    alpha = 10.0 ** rng.uniform(-2, 0)
    ratio = rng.uniform(0, 2)                         # rho*sqrt(s)/||g||
    rho = ratio * np.linalg.norm(g) / np.sqrt(sv)
    d, info = R.robust_step(g, np.full(p, sv), alpha, rho)
    d_cf = -alpha * max(1 - rho * np.sqrt(sv) / np.linalg.norm(g), 0.0) * g
    agree += int(info.zero == (ratio >= 1))
    n_zero += int(info.zero)
    if not info.zero:
        errs.append(np.linalg.norm(d - d_cf) / np.linalg.norm(d_cf))
        it3.append(info.iters)
c3 = {"n": N3, "n_zero_steps": n_zero, "branch_agreement": f"{agree}/{N3}",
      "max_rel_err": float(np.max(errs)),
      "iters_hist": {int(k): int(v) for k, v in zip(*np.unique(it3, return_counts=True))}}
# g = 0 case
d0, i0 = R.robust_step(np.zeros(4), np.ones(4), 0.1, 0.5)
c3["g_zero_gives_zero_step"] = bool(i0.zero and not d0.any())
RES["C3_isotropic"] = c3

# ===========================================================================
# C4. Zero-step rule (anisotropic), threshold sweep and direction
# ===========================================================================
rng = np.random.default_rng(SEED_C4)
mults = [0.5, 0.9, 0.99, 0.999, 1 - 1e-6, 1 + 1e-6, 1.001, 1.01, 1.1, 2.0]
c4 = {"mults": mults, "agreement": {}}
for m in mults:
    ok = 0
    for _ in range(1000):
        g, s, alpha = rand_problem(rng, int(rng.choice([2, 3, 33])))
        rho = m * np.sqrt(np.sum(g * g / s))
        d, info = R.robust_step(g, s, alpha, rho)
        ok += int((info.zero and not d.any()) == (m > 1))
    c4["agreement"][str(m)] = f"{ok}/1000"
RES["C4_zero_rule"] = c4

# sweep for the figure: anisotropic s vs isotropic s with same g
g = np.array([1.0, 0.6])
s_an = np.array([4.0, 0.05])
s_is = np.full(2, np.mean(s_an))
alpha = 0.5
sweep = {}
for name, s in [("aniso", s_an), ("iso", s_is)]:
    rc = np.sqrt(np.sum(g * g / s))
    rr = np.linspace(0, 1.15, 400) * rc
    nrm, ang_g, ang_n = [], [], []
    for r in rr:
        d, _ = R.robust_step(g, s, alpha, r)
        nrm.append(np.linalg.norm(d) / (alpha * np.linalg.norm(g)))
        if d.any():
            cosg = -d @ g / np.linalg.norm(d) / np.linalg.norm(g)
            v = g / s
            cosn = -d @ v / np.linalg.norm(d) / np.linalg.norm(v)
            ang_g.append(np.degrees(np.arccos(np.clip(cosg, -1, 1))))
            ang_n.append(np.degrees(np.arccos(np.clip(cosn, -1, 1))))
        else:
            ang_g.append(np.nan)
            ang_n.append(np.nan)
    sweep[name] = (rr / rc, np.array(nrm), np.array(ang_g), np.array(ang_n), rc)
v = g / s_an
c4["example"] = {"g": g.tolist(), "s_aniso": s_an.tolist(), "alpha": alpha,
                 "rho_c_aniso": float(sweep["aniso"][4]),
                 "rho_c_iso": float(sweep["iso"][4]),
                 "angle_between_g_and_Sinv_g_deg": float(np.degrees(np.arccos(
                     g @ v / np.linalg.norm(g) / np.linalg.norm(v))))}

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for name, col, ls, lab in [("aniso", C1, "-", r"anisotropic $s=(4,\,0.05)$"),
                           ("iso", C2, "--", r"isotropic $s=(2.025,\,2.025)$")]:
    x, nrm, ag, an, _ = sweep[name]
    ax[0].plot(x, nrm, color=col, ls=ls, label=lab)
ax[0].axvline(1, color=MUTED, lw=0.8, ls=":")
ax[0].set_xlabel(r"$\rho/\rho_c$,   $\rho_c=\sqrt{g^TS^{-1}g}$")
ax[0].set_ylabel(r"$\|d\|_2 \,/\, (\alpha\|g\|_2)$")
ax[0].set_title("(a) step length; zero for $\\rho\\geq\\rho_c$")
ax[0].legend(frameon=False, loc="lower left")
x, nrm, ag, an, _ = sweep["aniso"]
ax[1].plot(x, ag, color=C1, label=r"angle$(d,\,-g)$")
ax[1].plot(x, an, color=C3, ls="-.", label=r"angle$(d,\,-S^{-1}g)$")
x2, _, ag2, _, _ = sweep["iso"]
ax[1].plot(x2, ag2, color=C2, ls="--", label=r"angle$(d,\,-g)$, isotropic")
ax[1].axvline(1, color=MUTED, lw=0.8, ls=":")
ax[1].set_xlabel(r"$\rho/\rho_c$")
ax[1].set_ylabel("angle, degrees")
ax[1].set_title("(b) direction rotates from $-g$ to $-S^{-1}g$")
ax[1].legend(frameon=False, loc="center left", bbox_to_anchor=(0.3, 0.45))
fig.tight_layout()
fig.savefig("fig/zero_step_sweep.pdf")
plt.close(fig)

# ===========================================================================
# C5. Noisy-quadratic trajectories: SGD vs R-SGD, zero steps far from theta*
# ===========================================================================
rng = np.random.default_rng(SEED_C5)
B5, A0, K5, NSEED = 8, 0.2, 300, 20
sigmas = [0.0, 1.0, 4.0]
rhos5 = [0.0, 1.0, 2.45]
seeds5 = rng.integers(0, 2**31, size=(NSEED, 2))
c5 = {"b": B5, "alpha0": A0, "K": K5, "n_seeds": NSEED, "runs": {}}
traj = {}
for sg in sigmas:
    obs = R.make_quad_observe(B5, sg)
    for rho in rhos5:
        meth = "sgd" if rho == 0 else "rsgd"
        F, Z, res, fz, rt, ge, its = [], [], [], [], [], 0, []
        for sb, sn in seeds5:
            lg = R.run_optimizer(obs, [2.0, 2.0], meth, A0, rho, K5, int(sb), int(sn))
            F.append(R.quad_f(lg.theta))
            Z.append(lg.zero.mean())
            res += lg.residual
            fz += list(R.quad_f(lg.theta[:-1][lg.zero]))
            rt.append(lg.runtime)
            ge += lg.grad_evals
            its += [i for i, z in zip(lg.iters, lg.zero) if not z]
            if (sb, sn) == tuple(seeds5[0]):
                traj[(sg, rho)] = lg
        F = np.array(F)
        c5["runs"][f"sigma={sg},rho={rho}"] = {
            "final_f_median": float(np.median(F[:, -1])),
            "final_f_q25": float(np.percentile(F[:, -1], 25)),
            "final_f_q75": float(np.percentile(F[:, -1], 75)),
            "mean_f_last50": float(F[:, -50:].mean()),
            "zero_step_freq": float(np.mean(Z)),
            "f_at_zero_steps_max": float(max(fz)) if fz else None,
            "f_at_zero_steps_median": float(np.median(fz)) if fz else None,
            "max_residual_1d": float(max(res)) if res else None,
            "newton_iters_mean": float(np.mean(its)) if (its and rho > 0) else None,
            "grad_evals_per_run": ge // NSEED,
            "runtime_ms_per_run": float(np.mean(rt) * 1e3),
        }
        traj.setdefault(("F", sg, rho), F)
# sigma = 0 degenerate case: R-SGD vs SGD identical up to the eps floor
obs = R.make_quad_observe(B5, 0.0)
la = R.run_optimizer(obs, [2, 2], "sgd", A0, 0.0, K5, 1, 2)
lb = R.run_optimizer(obs, [2, 2], "rsgd", A0, 2.45, K5, 1, 2)
c5["sigma0_max_abs_diff_sgd_vs_rsgd"] = float(np.max(np.abs(la.theta - lb.theta)))
RES["C5_quadratic"] = c5

fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-2.2, 2.4, 300))
ax[0].contour(xx, yy, 0.5 * (xx ** 2 + 4 * yy ** 2), levels=[0.05, 0.25, 1, 2.5, 5, 10],
              colors="#cccccc", linewidths=0.7)
sg = 4.0
t = np.linspace(0, 2 * np.pi, 200)
rr = 2.45
ax[0].plot(rr * 2 * sg / np.sqrt(B5) * np.cos(t), rr * sg / (8 * np.sqrt(B5)) * np.sin(t),
           color=C3, ls="--", lw=1.1, label=r"$b\,\nabla f^T\Sigma^{-1}\nabla f=\rho^2$")
for rho, col, lab, z_ in [(0.0, C2, "SGD", 2), (2.45, C1, r"R-SGD, $\rho=2.45$", 3)]:
    th = traj[(sg, rho)].theta
    ax[0].plot(th[:, 0], th[:, 1], color=col, lw=1.0, label=lab, zorder=z_)
    if rho > 0:
        z = traj[(sg, rho)].zero
        ax[0].plot(th[:-1][z, 0], th[:-1][z, 1], "o", ms=3.5, mfc="white", mec=INK,
                   mew=0.8, label="zero step", ls="none", zorder=4)
ax[0].plot([2], [2], "s", color=INK, ms=4)
ax[0].annotate(r"$\theta_0$", (2, 2), xytext=(4, -10), textcoords="offset points")
ax[0].set_xlim(-3, 3); ax[0].set_ylim(-2.2, 2.4)
ax[0].set_xlabel(r"$\theta_1$"); ax[0].set_ylabel(r"$\theta_2$")
ax[0].set_title(r"(a) one seed, $\sigma=4$, $b=8$")
ax[0].legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper left", fontsize=6.8)
for i, sg in [(1, 1.0), (2, 4.0)]:
    for rho, col, ls in [(0.0, C2, "-"), (1.0, C3, "-."), (2.45, C1, "--")]:
        F = traj[("F", sg, rho)]
        lab = "SGD" if rho == 0 else rf"R-SGD $\rho={rho:g}$"
        ax[i].semilogy(np.median(F, 0), color=col, ls=ls, lw=1.2, label=lab)
    ax[i].set_xlabel("update $k$")
    ax[i].set_title(rf"({'bc'[i-1]}) median $f(\theta_k)$, $\sigma={sg:g}$")
    ax[i].set_ylim(5e-4, 15)
ax[1].set_ylabel(r"median over 20 seeds")
ax[2].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig("fig/quadratic_trajectories.pdf")
plt.close(fig)

# ===========================================================================
# C6. Batch size and the uncertainty estimate (Section 2(c))
# ===========================================================================
rng = np.random.default_rng(SEED_C6)
SIG = 1.0
Sigma = SIG ** 2 * R.QUAD_A ** 2              # per-observation variances (4, 1/4)
bs = [2, 3, 4, 8, 16, 32, 64, 128, 256]
NREP = 20000
points = {"theta*": np.array([0.0, 0.0]), "theta_mid": np.array([1.0, 0.1]),
          "theta0": np.array([2.0, 2.0])}
RHO6, ALPHA6 = 2.45, 0.1
c6 = {"sigma": SIG, "n_rep": NREP, "rho": RHO6, "alpha": ALPHA6, "rows": []}
for b in bs:
    row = {"b": b}
    mu = R.quad_grad(points["theta*"])
    Z = rng.standard_normal((NREP, b, 2))
    V = mu + SIG * Z * R.QUAD_A
    g = V.mean(1)
    shat = ((V - g[:, None, :]) ** 2).sum(1) / (b * (b - 1))
    eps = 1e-3 * np.maximum(shat.mean(1), 1e-8)
    s = shat + eps[:, None]
    row["bias_ratio"] = (b * shat.mean(0) / Sigma).tolist()          # -> 1
    row["rel_sd"] = (shat.std(0) / shat.mean(0)).tolist()
    row["rel_sd_theory"] = float(np.sqrt(2 / (b - 1)))
    k2 = np.sum(g * g / s, 1)
    row["kappa2_mean_at_opt"] = float(k2.mean())
    row["kappa2_theory_at_opt"] = float(2 * (b - 1) / (b - 3)) if b > 3 else None
    for name, th in points.items():
        mu = R.quad_grad(th)
        V = mu + SIG * Z * R.QUAD_A
        g = V.mean(1)
        shat = ((V - g[:, None, :]) ** 2).sum(1) / (b * (b - 1))
        s = shat + 1e-3 * np.maximum(shat.mean(1), 1e-8)[:, None]
        k2 = np.sum(g * g / s, 1)
        row[f"zero_freq_{name}"] = float(np.mean(k2 <= RHO6 ** 2))
        if name == "theta_mid":
            # relative change of the step w.r.t. SGD, rho = 1 (nonzero steps)
            rel = []
            for r in range(2000):
                d, info = R.robust_step(g[r], s[r], ALPHA6, 1.0)
                rel.append(np.linalg.norm(d + ALPHA6 * g[r]) /
                           (ALPHA6 * np.linalg.norm(g[r])))
            row["rel_shrink_mid_rho1_mean"] = float(np.mean(rel))
            row["ratio_rho_sqrt_s_over_g_median"] = float(np.median(
                np.sqrt(s[:, 0]) / np.abs(g[:, 0])))
    c6["rows"].append(row)
c6["mu_Sigma_inv_mu"] = {k: float(np.sum(R.quad_grad(v) ** 2 / Sigma))
                         for k, v in points.items()}
# sigma = 0: all observations equal -> shat = 0, s = eps = 1e-11
V = np.tile(R.quad_grad([2.0, 2.0]), (8, 1))
g, s, shat, eps = R.uncertainty_diag(V)
c6["sigma0"] = {"shat": shat.tolist(), "eps": eps, "kappa2": float(np.sum(g * g / s))}

# sampling with vs without replacement (logistic regression, sigma = 0)
Xs, ys = R.make_gaussian_blobs(64, rng)
th = np.array([0.3, -0.2, 0.1])
Upop = R.logreg_per_example_grads(th, Xs, ys)
n = len(ys)
c6["replacement"] = []
for b in [8, 32]:
    true_with = Upop.var(0) / b                                  # population var / b
    true_wo = Upop.var(0) / b * (n - b) / (n - 1)
    est_w, est_wo = [], []
    for _ in range(20000):
        idx = rng.integers(0, n, b)
        est_w.append(R.uncertainty_diag(Upop[idx])[2])
        idx = rng.choice(n, b, replace=False)
        est_wo.append(R.uncertainty_diag(Upop[idx])[2])
    c6["replacement"].append({
        "n": n, "b": b,
        "with_E_shat_over_true": (np.mean(est_w, 0) / true_with).tolist(),
        "without_E_shat_over_true": (np.mean(est_wo, 0) / true_wo).tolist(),
        "predicted_without": n / (n - b)})
RES["C6_batch"] = c6

rows = c6["rows"]
bb = np.array([r["b"] for r in rows])
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6))
for j, col, mk, lab in [(0, C1, "o", r"$j=1$ ($\Sigma_{11}=4$)"),
                        (1, C2, "s", r"$j=2$ ($\Sigma_{22}=1/4$)")]:
    ax[0].loglog(bb, [r["bias_ratio"][j] * Sigma[j] / b_ for r, b_ in zip(rows, bb)],
                 mk, color=col, ms=5, mfc="white", mew=1.2, label=lab)
    ax[0].loglog(bb, Sigma[j] / bb, color=col, lw=1)
ax[0].set_xlabel("batch size $b$"); ax[0].set_ylabel(r"$\mathbb{E}\,\hat s_j$")
ax[0].set_title(r"(a) mean: markers; $\Sigma_{jj}/b$: lines")
ax[0].legend(frameon=False, fontsize=7)
ax[1].semilogx(bb, [r["rel_sd"][0] for r in rows], "o", color=C1, ms=5, mfc="white",
               mew=1.2, label="empirical, $j=1$")
ax[1].semilogx(bb, [r["rel_sd"][1] for r in rows], "s", color=C2, ms=4, mfc="white",
               mew=1.2, label="empirical, $j=2$")
ax[1].semilogx(bb, [r["rel_sd_theory"] for r in rows], color=INK, lw=1,
               label=r"$\sqrt{2/(b-1)}$")
ax[1].set_xlabel("batch size $b$"); ax[1].set_ylabel(r"sd$(\hat s_j)/\mathbb{E}\hat s_j$")
ax[1].set_title("(b) relative spread of the estimate")
ax[1].legend(frameon=False, fontsize=7)
for name, col, mk, lab in [("theta*", C1, "o", r"$\theta^\star=(0,0)$"),
                           ("theta_mid", C3, "^", r"$\theta=(1,\,0.1)$"),
                           ("theta0", C2, "s", r"$\theta_0=(2,2)$")]:
    ax[2].semilogx(bb, [r[f"zero_freq_{name}"] for r in rows], marker=mk, color=col,
                   ms=4.5, mfc="white", mew=1.2, lw=1.2, label=lab)
ax[2].axhline(0.95, color=MUTED, lw=0.8, ls=":", label=r"$\chi^2_2$ limit, 0.95")
ax[2].set_xlabel("batch size $b$"); ax[2].set_ylabel("zero-step frequency")
ax[2].set_title(r"(c) $P(g^TS^{-1}g\leq\rho^2)$, $\rho=2.45$")
ax[2].legend(frameon=False, fontsize=7, loc="center right")
ax[2].set_ylim(-0.03, 1.03)
fig.tight_layout()
fig.savefig("fig/batch_size.pdf")
plt.close(fig)

with open("results_section2.json", "w") as fh:
    json.dump(RES, fh, indent=1)
print(json.dumps(RES, indent=1))
