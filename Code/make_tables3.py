"""Generate the LaTeX table bodies of the Section 3 report from tables_section3.json."""
import json
T = json.load(open("tables_section3.json"))
R = json.load(open("results_section3.json"))
C = [0, 1, 4, 16]

def f(x, d=4):
    return f"{x:.{d}f}"

def pm(t, d=4):
    return f"{t[0]:.{d}f} $\\pm$ {t[1]:.{d}f}"

out = {}
for m in ["logreg", "mlp"]:
    K = R[m]["K"]
    dl = 3 if m == "logreg" else 4
    # main table (design A)
    rows = []
    for c in C:
        r = T[m][f"{c}|A"]
        for meth, lab in [("sgd", "SGD"), ("rsgd", "R-SGD")]:
            z = "--" if meth == "sgd" else f"{100*r['rsgd_zero_freq'][0]:.1f}\\%"
            ms = r[f"{meth}_update_ms"][0]
            rows.append(f"{c} & {lab} & {pm(r[f'{meth}_train_loss'])} & {pm(r[f'{meth}_test_loss'])} & "
                        f"{pm(r[f'{meth}_test_acc'])} & {z} & {ms:.0f} ({1000*ms/K:.0f}) \\\\")
        if c != C[-1]:
            rows.append("\\midrule")
    out[f"{m}_main"] = "\n".join(rows)
    # paired table (design A and B)
    rows = []
    for c in C:
        for d in ["A", "B"]:
            r = T[m][f"{c}|{d}"]
            a, b_, cc = r["diff_train_loss"], r["diff_test_loss"], r["diff_test_acc"]
            rows.append(f"{c} & {d} & {a[0]:+.4f} $\\pm$ {a[1]:.4f} & {a[2]}/10 & "
                        f"{b_[0]:+.4f} $\\pm$ {b_[1]:.4f} & {b_[2]}/10 & "
                        f"{cc[0]:+.4f} $\\pm$ {cc[1]:.4f} & {cc[2]}/10 ({cc[3]}) & "
                        f"{r['agree_train_testloss']}/10 & {r['agree_train_acc']}/10 \\\\")
        if c != C[-1]:
            rows.append("\\midrule")
    out[f"{m}_paired"] = "\n".join(rows)
    # variability A vs B
    rows = []
    for c in C:
        a, b_ = T[m][f"{c}|A"], T[m][f"{c}|B"]
        rows.append(f"{c} & {a['sgd_test_loss'][1]:.4f} & {b_['sgd_test_loss'][1]:.4f} & "
                    f"{a['rsgd_test_loss'][1]:.4f} & {b_['rsgd_test_loss'][1]:.4f} & "
                    f"{a['sgd_test_acc'][1]:.4f} & {b_['sgd_test_acc'][1]:.4f} & "
                    f"{a['rsgd_test_acc'][1]:.4f} & {b_['rsgd_test_acc'][1]:.4f} \\\\")
    out[f"{m}_var"] = "\n".join(rows)
    # matched control at the frozen setting
    rows = []
    for c in C:
        a = T[m][f"{c}|A"]
        dm = a["diff_vs_matched_test_loss"]
        rows.append(f"{c} & {a['rsgd_step_ratio'][0]:.3f} & {a['matched_alpha0']:.3g} & "
                    f"{pm(a['rsgd_test_loss'])} & {pm(a['matched_test_loss'])} & "
                    f"{dm[0]:+.4f} $\\pm$ {dm[1]:.4f} & {dm[2]}/10 & "
                    f"{a['rsgd_test_acc'][0]:.4f} / {a['matched_test_acc'][0]:.4f} \\\\")
    out[f"{m}_matched"] = "\n".join(rows)
    # sensitivity (selected r)
    rows = []
    for c in C:
        for rv in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            s = T[m]["sens"][f"{c}|{rv}"]
            if rv == 0:
                mt, wins = "--", "--"
            else:
                mt = f"{s['matched_test_loss'][0]:.4f}"
                wins = f"{s['diff_vs_matched_test_loss'][1]}/5"
            rows.append(f"{c} & {rv:g} & {100*s['zero_freq'][0]:.1f}\\% & {s['step_ratio'][0]:.3f} & "
                        f"{s['train_loss'][0]:.4f} & {s['test_loss'][0]:.4f} & {mt} & {wins} & "
                        f"{s['test_acc'][0]:.4f} \\\\")
        if c != C[-1]:
            rows.append("\\midrule")
    out[f"{m}_sens"] = "\n".join(rows)
json.dump(out, open("tables3_tex.json", "w"), indent=1)
for k, v in out.items():
    print("%%", k); print(v)
