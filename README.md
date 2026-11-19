# Mini-project: SGD versus Robust SGD under Gradient Uncertainty

Course: Optimisation methods for artificial intelligence. Assignment: `HW_robust_sgd.pdf` (A.-H. Phan).

Team: Egor Chulkov, Vladislav Dvornikov, Daria Demeshko.

## What is where

| Folder | Content |
|---|---|
| `Report/` | **Report to submit**: `HW_RSGD_report_EN.pdf` and `HW_RSGD_report_RU.pdf` (10 pages each) |
| `Report/Detailed notes/` | Our longer working notes (not for submission): full 30+ page version, 1(c) (RU), Section 2 (EN, RU), Section 3 (EN, RU) |
| `Report/source/` | LaTeX sources of the report, with `fig/`, `fig3/`, `tabs3/`; sources of the notes are in `source/detailed_notes/` |
| `Code/` | `rsgd.py`, `checks_section2.py`, `sim3_core.py`, `sim_section3.py`, `make_tables3.py`, `check_tuning_budget.py`, `requirements.txt` |
| `Results/` | `results_section2.json`, `results_section3.json`, `results_tuning_budget.json`, `tables_section3.json`, `curves_section3.npz` |
| `Presentation/` | `SGD_vs_RSGD_slides.pdf` (8 slides, 5 minutes; Egor: slides 1–2, Vladislav: 3–5, Daria: 6–8) and `Presentation_QA_prep.md` (likely questions with answers) |

The editable slide deck (with speaker notes, exportable to PowerPoint/PDF):
https://claude.ai/artifact/PhzAZpsfVfVebssBzWMC5u

## Reproduce

```
cd Code
pip install -r requirements.txt
python checks_section2.py      # Section 2 checks, ~20 s  -> results_section2.json, fig/
python sim_section3.py run     # Sections 3-4, ~3 min     -> results_section3.json, curves_section3.npz, fig3/
python check_tuning_budget.py  # search costs + equal-budget SGD check, ~1 min -> results_tuning_budget.json
python make_tables3.py         # LaTeX tables             -> tables3_tex.json (copied to Report/source/tabs3/)
```
Outputs are written to the working directory; the copies used in the reports are in `Results/`
and `Report/source/`. Reports: run `xelatex` twice in `Report/source/` (notes: in `Report/source/detailed_notes/`) (fonts: Liberation Serif/Sans/Mono).

Seeds: Section 2 master seed 20261117; Sections 3–4 master seed 20261118
(`SeedSequence([20261118, sim, kind, seed])`, see report Section 6).

## Still to fill in before submission
- Each member's contribution in the "Team and contributions" line of both reports (`Report/source/HW_RSGD_report_*.tex`, then recompile).
