"""
check_tuning_budget.py -- Section 4 checks for the report:
  1. equal search budget: SGD tuned on a 36-point log grid of alpha0 in [0.1, 30]
     (108 runs, the same as R-SGD's 6 x 6 grid), same 3 tuning seeds, c = 1,
     same validation criterion; if the choice changes, the new SGD is evaluated
     on the same 10 evaluation seeds (design A) at every noise level;
  2. measured cost of the original searches (runs, gradient evaluations, seconds).

  python check_tuning_budget.py   -> results_tuning_budget.json
"""
import json
import time

import numpy as np

import sim3_core as C
from sim_section3 import SIMS, B, C_LEVELS, C_TUNE, N_TUNE, N_EVAL, ALPHA_GRID, R_GRID, setup, crit

FINE = [float(x) for x in np.geomspace(0.1, 30.0, 36)]


def timed_search(model, K, sigma, configs):
    """Run (method, alpha0, rho) configs on the tuning seeds; return crit table and seconds."""
    tab, t0 = {}, time.perf_counter()
    for meth, a0, rho in configs:
        vals = []
        for s in range(N_TUNE):
            split, th0, sb, sn = setup(model, 0, s)
            vals.append(crit(C.train(model, split, th0, meth, a0, rho, sigma, B, K, sb, sn)))
        tab[(meth, a0, rho)] = float(np.mean(vals))
    return tab, time.perf_counter() - t0


def main():
    res3 = json.load(open("results_section3.json"))
    out = {}
    for model, cfg in SIMS.items():
        K, p = cfg["K"], C.MODELS[model]["p"]
        sq = np.sqrt(p)
        gamma = res3[model]["gamma"]
        sig = C_TUNE * gamma
        old = res3[model]["tuning"]
        # --- cost of the original searches (re-run to time them) ---
        _, t_sgd = timed_search(model, K, sig, [("sgd", a, 0.0) for a in ALPHA_GRID])
        _, t_rsgd = timed_search(model, K, sig, [("sgd" if r == 0 else "rsgd", a, r * sq)
                                                 for a in ALPHA_GRID for r in R_GRID])
        # --- equal-budget SGD search ---
        fine, t_fine = timed_search(model, K, sig, [("sgd", a, 0.0) for a in FINE])
        best = min(fine, key=fine.get)[1]
        idx = FINE.index(best)
        r = {
            "grad_evals_per_run": B * K,
            "orig_sgd": {"runs": len(ALPHA_GRID) * N_TUNE,
                         "grad_evals": len(ALPHA_GRID) * N_TUNE * B * K, "seconds": t_sgd},
            "orig_rsgd": {"runs": len(ALPHA_GRID) * len(R_GRID) * N_TUNE,
                          "grad_evals": len(ALPHA_GRID) * len(R_GRID) * N_TUNE * B * K,
                          "seconds": t_rsgd},
            "fine_sgd": {"runs": len(FINE) * N_TUNE, "grad_evals": len(FINE) * N_TUNE * B * K,
                         "seconds": t_fine, "grid": FINE,
                         "crit": [fine[("sgd", a, 0.0)] for a in FINE],
                         "alpha0": best, "at_edge": idx in (0, len(FINE) - 1),
                         "best_crit": fine[("sgd", best, 0.0)]},
            "orig_sgd_alpha0": old["sgd_alpha0"],
            "orig_sgd_crit": old["table"][f"{old['sgd_alpha0']}|0.0"]["crit"],
            "rsgd_crit": old["table"][f"{old['rsgd_alpha0']}|{old['rsgd_r']}"]["crit"],
        }
        # --- evaluate the equal-budget SGD on the evaluation seeds (design A) ---
        ev = {}
        for c in C_LEVELS:
            rows = []
            for s in range(N_EVAL):
                split, th0, sb, sn = setup(model, 1, s)
                run = C.train(model, split, th0, "sgd", best, 0.0, c * gamma, B, K, sb, sn)
                rows.append(C.summary(model, split, run))
            rs = res3[model]["eval"][f"{c}|A|rsgd"]
            so = res3[model]["eval"][f"{c}|A|sgd"]
            tl = np.array([x["test_loss"] for x in rows])
            ev[str(c)] = {
                "test_loss_mean": float(tl.mean()), "test_loss_sd": float(tl.std(ddof=1)),
                "acc_mean": float(np.mean([x["test_acc"] for x in rows])),
                "orig_sgd_test_loss": float(np.mean([x["test_loss"] for x in so])),
                "rsgd_test_loss": float(np.mean([x["test_loss"] for x in rs])),
                "rsgd_better_seeds": int(sum(a["test_loss"] < b_ for a, b_ in zip(rs, tl))),
            }
        r["fine_sgd_eval"] = ev
        out[model] = r
        print(model, json.dumps({k: v for k, v in r.items() if k != "fine_sgd"}, indent=1, default=float))
        print(model, "fine SGD alpha0", best, "edge", r["fine_sgd"]["at_edge"])
    json.dump(out, open("results_tuning_budget.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
