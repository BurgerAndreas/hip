import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

csv_path = "results/eval_horm_wandb_export.csv"

name_to_name = {
    "hesspred": "EquiformerV2 (ours)",
    "eqv2": "EquiformerV2",
    "left-df": "LeftNet-DF",
    "left": "LeftNet",
    "alpha": "AlphaNet",
}
hessian_method_to_name = {
    "autograd": "Autograd",
    "predict": "Learned",
}

df0 = pd.read_csv(csv_path, quotechar='"')
df = df0[["Name", "hessian_method", "model_is_ts", "true_is_ts", "is_ts_agree"]].copy()
df["delta"] = df["model_is_ts"] - df["true_is_ts"]

# rename
df["Name"] = df["Name"].map(name_to_name)
df["hessian_method"] = df["hessian_method"].map(hessian_method_to_name)

print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Confusion counts (TP, FP, FN, TN) derived from rates and sample size
N = df0["max_samples"].astype(int)
tp = 0.5 * (df["model_is_ts"] + df["true_is_ts"] - 1 + df["is_ts_agree"]) * N
conf = pd.DataFrame({"Name": df["Name"], "TP": tp.round().astype(int)})
conf["FP"] = (df["model_is_ts"] * N - conf["TP"]).round().astype(int)
conf["FN"] = (df["true_is_ts"] * N - conf["TP"]).round().astype(int)
conf["TN"] = N - (conf["TP"] + conf["FP"] + conf["FN"])
conf["N"] = N
conf[["TP", "FP", "FN", "TN"]] = conf[["TP", "FP", "FN", "TN"]].clip(lower=0)

print()
print("Confusion counts (per model):")
print(conf[["Name", "TP", "FP", "FN", "TN", "N"]].to_string(index=False))

# Confusion rates (conditional percentages)
conf_rates = conf.copy()
conf_rates["P"] = conf_rates["TP"] + conf_rates["FN"]
conf_rates["Nneg"] = conf_rates["TN"] + conf_rates["FP"]
conf_rates["PP"] = conf_rates["TP"] + conf_rates["FP"]
conf_rates["PN"] = conf_rates["TN"] + conf_rates["FN"]
conf_rates["TPR"] = conf_rates["TP"] / conf_rates["P"].replace(0, pd.NA)
conf_rates["FNR"] = conf_rates["FN"] / conf_rates["P"].replace(0, pd.NA)
conf_rates["TNR"] = conf_rates["TN"] / conf_rates["Nneg"].replace(0, pd.NA)
conf_rates["FPR"] = conf_rates["FP"] / conf_rates["Nneg"].replace(0, pd.NA)
conf_rates["Precision"] = conf_rates["TP"] / conf_rates["PP"].replace(0, pd.NA)
conf_rates["NPV"] = conf_rates["TN"] / conf_rates["PN"].replace(0, pd.NA)
conf_rates["Accuracy"] = (conf_rates["TP"] + conf_rates["TN"]) / conf_rates[
    "N"
].replace(0, pd.NA)


def fmt_pct(x: float) -> str:
    # return f"{x*100:.1f}%"
    return f"{round(x * 100)}%"


print()
print("Confusion rates (per model):")
print(
    conf_rates[
        ["Name", "TPR", "FPR", "TNR", "FNR", "Precision", "NPV", "Accuracy"]
    ].to_string(
        index=False,
        formatters={
            k: fmt_pct
            for k in ["TPR", "FPR", "TNR", "FNR", "Precision", "NPV", "Accuracy"]
        },
    )
)

# LaTeX table for confusion rates
print()
# conf_frac_tbl = conf.copy()
# for c in ["TP", "FP", "FN", "TN"]:
# 	conf_frac_tbl[c] = conf_frac_tbl[c] / conf_frac_tbl["N"]

table_df = pd.DataFrame(
    {
        "Hessian": df["hessian_method"],
        "Name": df["Name"],
        "TPR": conf_rates["TPR"],
        "FPR": conf_rates["FPR"],
        "TNR": conf_rates["TNR"],
        "FNR": conf_rates["FNR"],
        "Precision": conf_rates["Precision"],
        "Accuracy": conf_rates["Accuracy"],
    }
)
# Order: autograd first, then learned; then by Name for stability
table_df = table_df.sort_values(["Hessian", "Name"]).reset_index(drop=True)
# Append rates: Precision and Accuracy
# prec_acc = conf_rates[["Name", "Precision", "Accuracy"]].copy()
# prec_acc["Precision"] = prec_acc["Precision"].fillna(0.0)
# prec_acc["Accuracy"] = prec_acc["Accuracy"].fillna(0.0)
# table_df = table_df.merge(prec_acc, on="Name", how="left")

best_tp_idx = int(table_df["TPR"].idxmax())
best_fp_idx = int(table_df["FPR"].idxmin())
best_fn_idx = int(table_df["FNR"].idxmin())
best_tn_idx = int(table_df["TNR"].idxmax())
best_prec_idx = int(table_df["Precision"].idxmax())
best_acc_idx = int(table_df["Accuracy"].idxmax())


def fmt_pct_int(x: float) -> str:
    return f"{round(x * 100)}\\%"


lines = []
lines.append("\\begin{tabular}{llrrrrrr}")
lines.append("\\hline")
lines.append(
    r"Hessian & Name & TPR $\uparrow$ & FPR $\downarrow$ & FNR $\downarrow$ & TNR $\uparrow$ & Precision $\uparrow$ & Accuracy $\uparrow$ \\"
)
lines.append("\\hline")
for i, row in table_df.iterrows():
    tp = fmt_pct_int(row["TPR"])
    fp = fmt_pct_int(row["FPR"])
    fn = fmt_pct_int(row["FNR"])
    tn = fmt_pct_int(row["TNR"])
    prec = fmt_pct_int(row["Precision"])
    acc = fmt_pct_int(row["Accuracy"])
    if i == best_tp_idx:
        tp = f"\\textbf{{{tp}}}"
    if i == best_fp_idx:
        fp = f"\\textbf{{{fp}}}"
    if i == best_fn_idx:
        fn = f"\\textbf{{{fn}}}"
    if i == best_tn_idx:
        tn = f"\\textbf{{{tn}}}"
    if i == best_prec_idx:
        prec = f"\\textbf{{{prec}}}"
    if i == best_acc_idx:
        acc = f"\\textbf{{{acc}}}"
    lines.append(
        f"{row['Hessian']} & {row['Name']}  & {tp} & {fp}  & {fn}  & {tn} & {prec} & {acc} \\\\"
    )
lines.append("\\hline")
lines.append("\\end{tabular}")
print("\n".join(lines))

#####################################################################
# Plots
#####################################################################
print()

# Bar plot of delta (model - true) only
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="Name", y="delta")
bound = max(0.05, float(df["delta"].abs().max()) * 1.15)
plt.ylim(-bound, bound)
plt.axhline(0, color="0.2", lw=1)
plt.ylabel("Delta vs true (rate)")
plt.title("TS rate delta (model - true)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
fname = "results/plots/ts_rate_comparison.png"
plt.savefig(fname, dpi=200)
print(f"Saved to {fname}")
plt.close()

# Optional: dumbbell plot to highlight deltas
plt.figure(figsize=(7, 3.8))
for i, row in df.reset_index().iterrows():
    y = i
    plt.plot([row["true_is_ts"], row["model_is_ts"]], [y, y], color="0.6", lw=2)
    plt.scatter(row["true_is_ts"], y, label="true_is_ts" if i == 0 else None)
    plt.scatter(row["model_is_ts"], y, label="model_is_ts" if i == 0 else None)
plt.yticks(range(len(df)), df["Name"])
plt.xlabel("TS rate")
plt.xlim(0, 1)
plt.title("TS rate difference (model - true)")
plt.legend()
plt.tight_layout()
fname = "results/plots/ts_rate_dumbbell.png"
plt.savefig(fname, dpi=200)
print(f"Saved to {fname}")
plt.close()

# Plot: horizontal stacked bars for confusion fractions per model
conf_frac = conf.copy()
for c in ["TP", "FP", "FN", "TN"]:
    conf_frac[c] = conf_frac[c] / conf_frac["N"]
order = conf_frac.sort_values("TP", ascending=False)["Name"].tolist()
plot_df = conf_frac.set_index("Name").loc[order][["TP", "FP", "FN", "TN"]]
plt.figure(figsize=(8, 3.8 + 0.2 * len(plot_df)))
left = None
for i, comp in enumerate(["TP", "FP", "FN", "TN"]):
    vals = plot_df[comp].values
    if left is None:
        left = vals * 0
    plt.barh(plot_df.index, vals, left=left, label=comp)
    left = left + vals
plt.xlim(0, 1)
plt.xlabel("Fraction of samples")
plt.title("Confusion breakdown per model")
plt.legend(ncol=4, bbox_to_anchor=(1, 1.02), loc="lower right")
plt.tight_layout()
fname = "results/plots/ts_confusion_stacked.png"
plt.savefig(fname, dpi=200)
print(f"Saved to {fname}")
plt.close()
