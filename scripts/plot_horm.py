import numpy as np
import matplotlib.pyplot as plt
import os

fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

# (a) grouped bars
names = ["AlphaNet", "LEFTNet", "LEFTNet-df", "EquiformerV2"]
x = np.arange(len(names))
w = 0.35
a_vals = np.array([828, 852, 895, 888])  # GSM Success
b_vals = np.array([650, 661, 55, 3])  # Intended

ax = axs[0, 0]
ax.bar(x - w / 2, a_vals, width=w, label="GSM Success (E-F)", alpha=0.3)
ax.bar(x + w / 2, b_vals, width=w, label="Intended (E-F)", alpha=0.3)
for i, (a, b) in enumerate(zip(a_vals, b_vals)):
    ax.text(i - w / 2, a + 5, str(a), ha="center", va="bottom", fontsize=8)
    ax.text(i + w / 2, b + 5, str(b), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, names)
ax.set_ylabel("Number of Reactions")
ax.set_title("(a) TS search performances of E-F models.")
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
ax.legend()

# (b) same idea for E-F-H
# ... repeat with other numbers
axs[0, 1].set_title("(b) TS search performances of E-F-H models.")
axs[0, 1].grid(True, axis="y", linestyle="--", alpha=0.4)

# (c) boxplot of RMSD
ax = axs[1, 0]
data_EF = [np.random.lognormal(-3, 0.4, 100) for _ in names]
data_EFH = [np.random.lognormal(-2.8, 0.4, 100) for _ in names]
ax.boxplot(data_EF, positions=x - 0.15, widths=0.25, patch_artist=True)
ax.boxplot(data_EFH, positions=x + 0.15, widths=0.25, patch_artist=True)
ax.set_xticks(x, names)
ax.set_yscale("log")
ax.set_ylabel("TS RMSD (Å)")
ax.set_title("(c) RMSD of predicted TS.")
ax.legend(["E-F", "E-F-H"])

# (d) boxplot with log scale for barrier errors
ax = axs[1, 1]
err_EF = [np.random.lognormal(0, 0.6, 100) for _ in names]
err_EFH = [np.random.lognormal(-0.2, 0.6, 100) for _ in names]
ax.boxplot(err_EF, positions=x - 0.15, widths=0.25, patch_artist=True)
ax.boxplot(err_EFH, positions=x + 0.15, widths=0.25, patch_artist=True)
ax.set_xticks(x, names)
ax.set_yscale("log")
ax.set_ylabel("Absolute Barrier Error (kcal/mol)")
ax.set_title("(d) Barrier prediction errors.")
ax.legend(["E-F", "E-F-H"])
fname = "results/plots/horm.png"
plt.savefig(fname, dpi=200)
print(f"Saved to {fname}")
plt.close()
