"""LaTeX rows for the repeated-partition table and the effect-decomposition table (from reports/revision/*.csv)."""
import csv, sys
from collections import defaultdict

NAMES = {"ECG_cinc2021": ("ECG", 136), "eICU_expl_k12": ("eICU-CRD", 787), "ExtraSensory": ("ExtraSensory", 225)}
ORDER = ["FedWrap-MLFS", "base NSGA-II", "local-top-k union", "fed-rank", "top-frequency", "FMLFS", "all features"]
LAB = {"FedWrap-MLFS": r"\method{} (ours)", "base NSGA-II": "base NSGA-II", "local-top-k union": r"local-top-$k$ union",
       "fed-rank": "fed-rank", "top-frequency": "top-frequency", "FMLFS": r"FMLFS \cite{FMLFS_2024}",
       "all features": "all features (no selection)"}


def pfmt(p):
    p = float(p)
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


rows = defaultdict(dict)
for r in csv.DictReader(open("reports/revision/resplit_table.csv")):
    rows[r["dataset"]][r["method"]] = r
out = []
for ds, (name, D) in NAMES.items():
    if ds not in rows:
        continue
    for i, m in enumerate(ORDER):
        r = rows[ds][m]
        first = rf"\multirow{{7}}{{*}}{{\shortstack[l]{{{name}\\$D{{=}}{D}$}}}}" if i == 0 else ""
        macro = f"{float(r['macro']):.3f}\\,$\\pm$\\,{float(r['macro_std']):.3f}"
        if i == 0:
            macro = r"\textbf{" + f"{float(r['macro']):.3f}" + r"}\,$\pm$\," + f"{float(r['macro_std']):.3f}"
            tail = "--- & ---"
        else:
            tail = f"${float(r['diff']):+.3f}$ & {pfmt(r['p_holm'])}"
        out.append(f"{first} & {LAB[m]} & {macro} & {float(r['micro']):.3f} & {float(r['nfeat']):.0f} & {float(r['ratio']):.2f} & {tail} \\\\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"
print("% ---- tab:resplit rows"); print("\n".join(out))

print("\n% ---- tab:decomposition rows")
for r in csv.DictReader(open("reports/revision/resplit_decomposition.csv")):
    name = NAMES[r["dataset"]][0]
    ev_f, ev_b = float(r["unique_evals_fedaware"]), float(r["unique_evals_base"])
    print(f"{name} & {r['filter_at_base']} & {float(r['filter_at_base_macro']):.3f} & {float(r['base']):.3f} & {float(r['fedwrap']):.3f} & "
          f"${float(r['wrapper_effect']):+.3f}$ ({pfmt(r['wrapper_p'])}) & ${float(r['fedaware_effect']):+.3f}$ ({pfmt(r['fedaware_p'])}) & "
          f"{ev_b:,.0f} & {ev_f:,.0f} & ${100*(ev_f/ev_b-1):+.0f}\\%$ ({pfmt(r['unique_evals_p'])}) \\\\".replace(",", "{,}"))
