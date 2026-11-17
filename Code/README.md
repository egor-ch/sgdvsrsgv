# Code

- `rsgd.py` — formula (4) estimator of g_k and diagonal S_k; exact solver for the robust step (2)
  (secular equation, monotone Newton); SGD / R-SGD loop; gradient oracles (noisy quadratic,
  logistic regression, two-layer tanh network); data generators; finite-difference check.
- `checks_section2.py` — all numerical checks of Section 2 (zero radius, isotropic formula,
  zero-step rule, 1(d) residual, CVXPY cross-check, batch size, quadratic trajectories).
- `sim3_core.py` — balanced train/val/test splits, train-only standardisation, training loop with
  clean evaluation and timing.
- `sim_section3.py` — tuning (validation, c = 1), evaluation designs A/B (10 paired seeds),
  radius sweep, step-length-matched SGD controls, figures. `python sim_section3.py run`.
- `make_tables3.py` — the LaTeX tables used in the report.

Requirements: see `requirements.txt` (Python 3.11+; float64 throughout).
