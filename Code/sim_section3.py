"""
sim_section3.py -- Section 3: required simulations (with the Section 4 protocol
needed for a fair comparison).

  python sim_section3.py run    # all experiments -> results_section3.json, curves_section3.npz
  python sim_section3.py plot   # figures fig3/*.pdf from the saved results

Protocol (identical for both simulations)
  data      : balanced; n_train = n_val = 400, n_test = 4000; standardised with
              training mean / std only
  budget    : b = 16, K updates (500 logistic, 3000 network), alpha_k = alpha0/sqrt(1+k/50)
  noise     : v = u_i(theta) + sigma z,  sigma = c * gamma,  c in {0, 1, 4, 16},
              gamma = RMS of clean per-example gradients at initialisation
  tuning    : 3 tuning seeds (own data, init, batches, noise), noise level c = 1,
              criterion = mean validation loss over the last 5 checkpoints;
              SGD: alpha0 grid (6); R-SGD: alpha0 grid x rho/sqrt(p) grid (36, incl. 0)
  evaluation: frozen settings; 10 seeds "A" (new data set per seed) and 10 seeds
              "B" (data of A-seed 0 fixed; init, batches, noise vary); paired
              SGD / R-SGD with common random numbers
  extras    : rho sweep at SGD's alpha0 (sensitivity), step-length-matched SGD
"""
import json
import sys
import time

import numpy as np
from scipy import stats

import sim3_core as C

SIMS = {"logreg": dict(sim_id=1, K=500), "mlp": dict(sim_id=2, K=3000)}
B = 16
C_LEVELS = [0, 1, 4, 16]
C_TUNE = 1
ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
R_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]          # rho = r * sqrt(p)
R_SENS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
N_TUNE, N_EVAL, N_SENS = 3, 10, 5


def setup(model, kind, seed, data_from=None):
    """Data split, initial point and the two training streams for one seed.
    data_from=(kind, seed) overrides the data stream (used by the B seeds)."""
    sid = SIMS[model]["sim_id"]
    sd, si, sb, sn = C.seed_streams(sid, kind, seed)
    if data_from is not None:
        sd = C.seed_streams(sid, *data_from)[0]
    split = C.make_split(model, sd)
    theta0 = C.MODELS[model]["init"](np.random.default_rng(si))
    return split, theta0, sb, sn


def crit(run):
    return np.inf if run.failed else float(np.mean(run.val_loss[-5:]))


def run_all():
    out, curves = {}, {}
    for model, cfg in SIMS.items():
        K, p = cfg["K"], C.MODELS[model]["p"]
        sq = np.sqrt(p)
        res = {"K": K, "b": B, "p": p}
        t_start = time.time()
        # ---------------- noise unit gamma (frozen constant) ----------------
        gam = [C.grad_scale(model, *setup(model, 0, s)[:2]) for s in range(N_TUNE)]
        gamma = float(np.mean(gam))
        res["gamma"] = gamma
        res["sigmas"] = {str(c): c * gamma for c in C_LEVELS}
        # ---------------- tuning at c = C_TUNE ----------------
        sig = C_TUNE * gamma
        tune = {}
        for a0 in ALPHA_GRID:
            for r in R_GRID:
                vals, fails = [], 0
                for s in range(N_TUNE):
                    split, th0, sb, sn = setup(model, 0, s)
                    meth = "sgd" if r == 0 else "rsgd"
                    run = C.train(model, split, th0, meth, a0, r * sq, sig, B, K, sb, sn)
                    vals.append(crit(run))
                    fails += run.failed
                tune[(a0, r)] = (float(np.mean(vals)), fails)
        sgd_best = min(ALPHA_GRID, key=lambda a: tune[(a, 0.0)][0])
        r_best = min(tune, key=lambda k: tune[k][0])
        res["tuning"] = {
            "grid_alpha0": ALPHA_GRID, "grid_r": R_GRID, "noise_c": C_TUNE,
            "n_seeds": N_TUNE, "budget_runs_sgd": len(ALPHA_GRID) * N_TUNE,
            "budget_runs_rsgd": len(ALPHA_GRID) * len(R_GRID) * N_TUNE,
            "table": {f"{a}|{r}": {"crit": v[0], "fails": v[1]} for (a, r), v in tune.items()},
            "sgd_alpha0": sgd_best, "rsgd_alpha0": r_best[0], "rsgd_r": r_best[1],
            "rsgd_rho": r_best[1] * sq,
            "sgd_at_edge": sgd_best in (ALPHA_GRID[0], ALPHA_GRID[-1]),
            "rsgd_alpha_at_edge": r_best[0] in (ALPHA_GRID[0], ALPHA_GRID[-1]),
            "rsgd_r_at_edge": r_best[1] == R_GRID[-1],
        }
        conf = {"sgd": ("sgd", sgd_best, 0.0), "rsgd": ("rsgd", r_best[0], r_best[1] * sq)}
        # ---------------- evaluation ----------------
        ev = {}
        for c in C_LEVELS:
            sig = c * gamma
            for design, kind, df in [("A", 1, None), ("B", 2, (1, 0))]:
                for name, (meth, a0, rho) in conf.items():
                    rows, trc, vac = [], [], []
                    for s in range(N_EVAL):
                        split, th0, sb, sn = setup(model, kind, s, df)
                        run = C.train(model, split, th0, meth, a0, rho, sig, B, K, sb, sn)
                        sm = C.summary(model, split, run)
                        if model == "logreg":
                            # Bayes rule in original coordinates: t* = 2(a1 + a2)
                            Xo = split.Xte * split.std + split.mean
                            tb = 2 * Xo.sum(1)
                            sm["bayes_acc"] = float(np.mean((tb > 0) == (split.yte == 1)))
                            sm["bayes_test_loss"] = float(np.mean(np.logaddexp(0, tb) - split.yte * tb))
                        rows.append(sm)
                        trc.append(run.train_loss)
                        vac.append(run.val_loss)
                        if design == "A" and s == 0:
                            curves[f"{model}|theta|{c}|{name}"] = run.theta
                    ev[f"{c}|{design}|{name}"] = rows
                    curves[f"{model}|train|{c}|{design}|{name}"] = np.array(trc)
                    curves[f"{model}|val|{c}|{design}|{name}"] = np.array(vac)
                    curves[f"{model}|ks"] = run.ks
            # step-length-matched SGD (design A): alpha0' = alpha0_R * mean step ratio
            ratio = float(np.mean([r_["step_ratio"] for r_ in ev[f"{c}|A|rsgd"]]))
            a_m = conf["rsgd"][1] * ratio
            rows = []
            for s in range(N_EVAL):
                split, th0, sb, sn = setup(model, 1, s)
                run = C.train(model, split, th0, "sgd", a_m, 0.0, sig, B, K, sb, sn)
                rows.append(C.summary(model, split, run))
            ev[f"{c}|A|sgd_matched"] = rows
            ev[f"{c}|A|sgd_matched_alpha0"] = a_m
        res["eval"] = ev
        # ---------------- sensitivity: rho sweep at SGD's alpha0 ----------------
        sens = {}
        for c in C_LEVELS:
            for r in R_SENS:
                rows = []
                for s in range(N_SENS):
                    split, th0, sb, sn = setup(model, 1, s)
                    meth = "sgd" if r == 0 else "rsgd"
                    run = C.train(model, split, th0, meth, sgd_best, r * sq, c * gamma, B, K, sb, sn)
                    rows.append(C.summary(model, split, run))
                sens[f"{c}|{r}"] = rows
        res["sensitivity"] = {"alpha0": sgd_best, "r_values": R_SENS, "n_seeds": N_SENS,
                              "rows": sens}
        # ---------------- gradient check at the actual initial points -------------
        split, th0, _, _ = setup(model, 1, 0)
        import rsgd as R
        res["fd_check"] = R.fd_check(C.MODELS[model]["loss"], C.MODELS[model]["grads"],
                                     th0 if model == "mlp" else np.array([0.3, -0.2, 0.1]),
                                     split.Xtr[:50], split.ytr[:50])
        res["wall_seconds"] = time.time() - t_start
        out[model] = res
        print(model, "done in", round(res["wall_seconds"], 1), "s;",
              "SGD alpha0", sgd_best, "| R-SGD", r_best, flush=True)
    out["protocol"] = {"master_seed": C.MASTER3, "b": B, "c_levels": C_LEVELS,
                       "c_tune": C_TUNE, "n_tune": N_TUNE, "n_eval": N_EVAL,
                       "n_sens": N_SENS, "n_train": 400, "n_val": 400, "n_test": 4000,
                       "schedule": "alpha0/sqrt(1+k/50)", "criterion":
                       "mean validation loss over last 5 of 61 checkpoints"}
    with open("results_section3.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    np.savez("curves_section3.npz", **curves)


def run_sens_matched():
    """For every point (c, r>0) of the radius sweep: SGD with the step size
    scaled by R-SGD's mean step ratio (same seeds) -- the step-length control."""
    res = json.load(open("results_section3.json"))
    for model, cfg in SIMS.items():
        K = cfg["K"]
        r = res[model]
        a0 = r["sensitivity"]["alpha0"]
        out = {}
        for c in C_LEVELS:
            for rv in R_SENS[1:]:
                rows_r = r["sensitivity"]["rows"][f"{c}|{rv}"]
                ratio = float(np.mean([x["step_ratio"] for x in rows_r]))
                rows = []
                for s in range(N_SENS):
                    split, th0, sb, sn = setup(model, 1, s)
                    run = C.train(model, split, th0, "sgd", a0 * ratio, 0.0, c * r["gamma"],
                                  B, K, sb, sn)
                    rows.append(C.summary(model, split, run))
                out[f"{c}|{rv}"] = {"alpha0": a0 * ratio, "rows": rows}
        r["sensitivity_matched"] = out
    with open("results_section3.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)


# ===========================================================================
# Tables (aggregates used in the report)
# ===========================================================================
def agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std(ddof=1))


def paired(rows_r, rows_s, key):
    d = np.array([a[key] - b[key] for a, b in zip(rows_r, rows_s)])
    ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(ci), d


def tables():
    res = json.load(open("results_section3.json"))
    T = {}
    for model in SIMS:
        r = res[model]
        T[model] = {}
        for c in C_LEVELS:
            for design in ["A", "B"]:
                S, Rr = r["eval"][f"{c}|{design}|sgd"], r["eval"][f"{c}|{design}|rsgd"]
                row = {}
                for k in ["train_loss", "test_loss", "test_acc", "zero_freq",
                          "update_ms", "step_ratio"]:
                    row[f"sgd_{k}"] = agg(S, k)
                    row[f"rsgd_{k}"] = agg(Rr, k)
                for k in ["train_loss", "test_loss", "test_acc"]:
                    m, ci, d = paired(Rr, S, k)
                    better = (d > 0) if k == "test_acc" else (d < 0)
                    row[f"diff_{k}"] = (m, ci, int(better.sum()), int((d == 0).sum()))
                dtr = np.array([a["train_loss"] - b["train_loss"] for a, b in zip(Rr, S)])
                dte = np.array([a["test_loss"] - b["test_loss"] for a, b in zip(Rr, S)])
                dac = np.array([a["test_acc"] - b["test_acc"] for a, b in zip(Rr, S)])
                row["agree_train_testloss"] = int(np.sum(np.sign(dtr) == np.sign(dte)))
                row["agree_train_acc"] = int(np.sum((dtr < 0) == (dac > 0)))
                row["fails"] = (sum(x["failed"] for x in S), sum(x["failed"] for x in Rr))
                if model == "logreg":
                    row["bayes_acc"] = agg(S, "bayes_acc")
                    row["bayes_test_loss"] = agg(S, "bayes_test_loss")
                if design == "A":
                    M = r["eval"][f"{c}|A|sgd_matched"]
                    row["matched_alpha0"] = r["eval"][f"{c}|A|sgd_matched_alpha0"]
                    for k in ["train_loss", "test_loss", "test_acc"]:
                        row[f"matched_{k}"] = agg(M, k)
                        m, ci, d = paired(Rr, M, k)
                        better = (d > 0) if k == "test_acc" else (d < 0)
                        row[f"diff_vs_matched_{k}"] = (m, ci, int(better.sum()))
                T[model][f"{c}|{design}"] = row
        sens = {}
        for c in C_LEVELS:
            for rv in R_SENS:
                rows = r["sensitivity"]["rows"][f"{c}|{rv}"]
                sens[f"{c}|{rv}"] = {k: agg(rows, k) for k in
                                     ["train_loss", "test_loss", "test_acc", "zero_freq",
                                      "step_ratio"]}
        if "sensitivity_matched" in r:
            for c in C_LEVELS:
                for rv in R_SENS[1:]:
                    mm = r["sensitivity_matched"][f"{c}|{rv}"]
                    sens[f"{c}|{rv}"]["matched_test_loss"] = agg(mm["rows"], "test_loss")
                    sens[f"{c}|{rv}"]["matched_test_acc"] = agg(mm["rows"], "test_acc")
                    sens[f"{c}|{rv}"]["matched_train_loss"] = agg(mm["rows"], "train_loss")
                    sens[f"{c}|{rv}"]["matched_alpha0"] = mm["alpha0"]
                    rr = r["sensitivity"]["rows"][f"{c}|{rv}"]
                    d = np.array([a["test_loss"] - b["test_loss"] for a, b in zip(rr, mm["rows"])])
                    sens[f"{c}|{rv}"]["diff_vs_matched_test_loss"] = (float(d.mean()), int((d < 0).sum()))
        T[model]["sens"] = sens
    with open("tables_section3.json", "w") as fh:
        json.dump(T, fh, indent=1)
    return res, T


# ===========================================================================
# Figures
# ===========================================================================
def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    INK, MUTED = "#222222", "#777777"
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "savefig.bbox": "tight", "figure.dpi": 150})
    res, T = tables()
    cv = np.load("curves_section3.npz")
    import os
    os.makedirs("fig3", exist_ok=True)
    titles = {"logreg": "logistic regression", "mlp": "two-layer network"}
    for model in SIMS:
        r = res[model]
        ks = cv[f"{model}|ks"]
        # ---------- loss curves ----------
        fig, ax = plt.subplots(2, 4, figsize=(8.2, 4.0), sharex=True)
        for j, c in enumerate(C_LEVELS):
            for i, kind in enumerate(["train", "val"]):
                a = ax[i, j]
                for name, col, ls, lab in [("sgd", C2, "-", "SGD"), ("rsgd", C1, "--", "R-SGD")]:
                    Y = cv[f"{model}|{kind}|{c}|A|{name}"]
                    Y = np.where(np.isfinite(Y), Y, np.nan)
                    med = np.nanmedian(Y, 0)
                    lo, hi = np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)
                    a.plot(ks, med, color=col, ls=ls, label=lab)
                    a.fill_between(ks, lo, hi, color=col, alpha=0.18, lw=0)
                a.set_yscale("log")
                if i == 0:
                    a.set_title(rf"$c={c}$, $\sigma={c * r['gamma']:.3g}$")
                if j == 0:
                    a.set_ylabel(("training" if kind == "train" else "validation") + " loss")
                if i == 1:
                    a.set_xlabel("update $k$")
        ax[0, 0].legend(frameon=False, loc="upper right")
        fig.suptitle(f"{titles[model]}: median and interquartile band over {N_EVAL} "
                     "evaluation seeds (design A)", fontsize=9, y=1.0)
        fig.tight_layout()
        fig.savefig(f"fig3/{model}_curves.pdf")
        plt.close(fig)
        # ---------- decision boundaries (evaluation seed A-0) ----------
        split, th0, _, _ = setup(model, 1, 0)
        Xo = split.Xte * split.std + split.mean
        rng = np.random.default_rng(0)
        sub = rng.choice(len(split.yte), 600, replace=False)
        lo_, hi_ = Xo.min(0) - 0.3, Xo.max(0) + 0.3
        gx, gy = np.meshgrid(np.linspace(lo_[0], hi_[0], 300), np.linspace(lo_[1], hi_[1], 300))
        G = (np.column_stack([gx.ravel(), gy.ravel()]) - split.mean) / split.std
        fig, ax = plt.subplots(1, 4, figsize=(8.2, 2.4), sharey=True)
        for j, c in enumerate(C_LEVELS):
            a = ax[j]
            ys = split.yte[sub]
            a.scatter(Xo[sub][ys == 0, 0], Xo[sub][ys == 0, 1], s=4, color="#9a9a9a", lw=0,
                      label="class 0")
            a.scatter(Xo[sub][ys == 1, 0], Xo[sub][ys == 1, 1], s=4, marker="^",
                      color="#cfcfcf", edgecolor="#8a8a8a", lw=0.3, label="class 1")
            for name, col, ls, lab in [("sgd", C2, "-", "SGD"), ("rsgd", C1, "--", "R-SGD")]:
                th = cv[f"{model}|theta|{c}|{name}"]
                Z = C.MODELS[model]["logits"](th, G).reshape(gx.shape)
                a.contour(gx, gy, Z, levels=[0], colors=col, linestyles=ls, linewidths=1.5)
                a.plot([], [], color=col, ls=ls, label=lab)
            if model == "logreg":
                xx = np.linspace(lo_[0], hi_[0], 10)
                a.plot(xx, -xx, color=INK, lw=0.8, ls=":", label="Bayes")
            a.set_xlim(lo_[0], hi_[0]); a.set_ylim(lo_[1], hi_[1])
            accS = r["eval"][f"{c}|A|sgd"][0]["test_acc"]
            accR = r["eval"][f"{c}|A|rsgd"][0]["test_acc"]
            a.set_title(rf"$c={c}$: acc SGD {accS:.3f}, R {accR:.3f}", fontsize=7.8)
            a.set_xlabel("$a_1$")
        ax[0].set_ylabel("$a_2$")
        ax[-1].legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5,
                      loc="lower right" if model == "logreg" else "upper right")
        fig.tight_layout()
        fig.savefig(f"fig3/{model}_boundaries.pdf")
        plt.close(fig)
        # ---------- sensitivity ----------
        S = T[model]["sens"]
        fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.5))
        cols = [C3, C1, C4, C2]
        mks = ["o", "s", "^", "D"]
        for c, col, mk in zip(C_LEVELS, cols, mks):
            rv = np.array(R_SENS)
            ax[0].plot(rv, [S[f"{c}|{x}"]["test_loss"][0] for x in R_SENS], marker=mk,
                       color=col, ms=4, mfc="white", mew=1.1, label=f"$c={c}$")
            if "matched_test_loss" in S[f"{c}|0.25"]:
                ax[0].plot(rv, [S[f"{c}|0.0"]["test_loss"][0]] +
                           [S[f"{c}|{x}"]["matched_test_loss"][0] for x in R_SENS[1:]],
                           color=col, ls=":", lw=1.2)
            ax[1].plot(rv, [S[f"{c}|{x}"]["zero_freq"][0] for x in R_SENS], marker=mk,
                       color=col, ms=4, mfc="white", mew=1.1)
            ax[2].plot(rv, [S[f"{c}|{x}"]["step_ratio"][0] for x in R_SENS], marker=mk,
                       color=col, ms=4, mfc="white", mew=1.1)
        ax[0].set_yscale("log")
        ax[0].set_ylabel("test loss"); ax[1].set_ylabel("zero-step frequency")
        ax[2].set_ylabel(r"mean $\|d_k\|/(\alpha_k\|g_k\|)$")
        for a in ax:
            a.set_xlabel(r"$r=\rho/\sqrt{p}$")
        ax[0].plot([], [], color=MUTED, ls=":", label="step-matched SGD")
        ax[0].legend(frameon=False, fontsize=6.5)
        fig.suptitle(rf"{titles[model]}: radius sweep at SGD's $\alpha_0={r['sensitivity']['alpha0']:g}$, "
                     f"{N_SENS} seeds", fontsize=9, y=1.02)
        fig.tight_layout()
        fig.savefig(f"fig3/{model}_sensitivity.pdf")
        plt.close(fig)
        # ---------- tuning heat map (text table figure) ----------
        tb = r["tuning"]["table"]
        M = np.array([[tb[f"{a}|{x}"]["crit"] for x in R_GRID] for a in ALPHA_GRID])
        fig, a = plt.subplots(figsize=(4.2, 2.6))
        Mp = np.where(np.isfinite(M), M, np.nan)
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Rectangle
        cmap = LinearSegmentedColormap.from_list("ramp", ["#6da7ec", "#cde2fb", "#ffffff"])
        a.imshow(np.log10(Mp), cmap=cmap, aspect="auto")
        for i in range(len(ALPHA_GRID)):
            for j in range(len(R_GRID)):
                v = M[i, j]
                a.text(j, i, "fail" if not np.isfinite(v) else f"{v:.3f}", ha="center",
                       va="center", fontsize=6.5, color=INK)
        tsel = r["tuning"]
        i_s = ALPHA_GRID.index(tsel["sgd_alpha0"])
        a.add_patch(Rectangle((-0.5, i_s - 0.5), 1, 1, fill=False, ec=C2, lw=2))
        i_r, j_r = ALPHA_GRID.index(tsel["rsgd_alpha0"]), R_GRID.index(tsel["rsgd_r"])
        a.add_patch(Rectangle((j_r - 0.5, i_r - 0.5), 1, 1, fill=False, ec=C1, lw=2, ls="--"))
        a.set_xticks(range(len(R_GRID)), [f"{x:g}" for x in R_GRID])
        a.set_yticks(range(len(ALPHA_GRID)), [f"{x:g}" for x in ALPHA_GRID])
        a.set_xlabel(r"$r=\rho/\sqrt{p}$  ($r=0$: SGD)"); a.set_ylabel(r"$\alpha_0$")
        a.grid(False)
        a.set_title("validation loss, $c=1$, 3 seeds; boxes: selected SGD (solid), R-SGD (dashed)", fontsize=7.5)
        fig.tight_layout()
        fig.savefig(f"fig3/{model}_tuning.pdf")
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_all()
        run_sens_matched()
    if len(sys.argv) > 1 and sys.argv[1] == "matched":
        run_sens_matched()
    plot()
