#!/usr/bin/env python
"""Replot HIP vs AD Hessian time per sample versus batch size."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_style import (  # noqa: E402
    AD_COLOR,
    HIP_COLOR,
    LINE_WIDTH,
    MARKER_SIZE,
    apply_plot_style,
    finish_axis,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    ROOT
    / "paper"
    / "hessian_prediction"
    / "plots"
    / "time_batching.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "paper"
    / "hessian_prediction"
    / "plots"
    / "time_batching.pdf"
)

DISPLAY = {
    "prediction": "HIP Hessians",
    "autograd": "AD Hessians",
}
COLOUR = {
    "prediction": HIP_COLOR,
    "autograd": AD_COLOR,
}
MARKER = {
    "prediction": "D",
    "autograd": "o",
}
LINESTYLE = {
    "prediction": "-",
    "autograd": "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()

    df = pd.read_csv(args.csv)
    df["time_per_sample"] = df["time"] / df["batch_size"]
    summary = (
        df.groupby(["method", "batch_size"], as_index=False)["time_per_sample"]
        .mean()
        .sort_values(["method", "batch_size"])
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for method in ("autograd", "prediction"):
        sub = summary[summary["method"] == method]
        ax.plot(
            sub["batch_size"],
            sub["time_per_sample"],
            marker=MARKER[method],
            linestyle=LINESTYLE[method],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=COLOUR[method],
            label=DISPLAY[method],
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time per sample (ms)")
    ax.set_xlim(0, 68)

    ad1 = float(
        summary.loc[
            (summary["method"] == "autograd") & (summary["batch_size"] == 1),
            "time_per_sample",
        ].iloc[0]
    )
    hip_max_bz = int(summary.loc[summary["method"] == "prediction", "batch_size"].max())
    hip = float(
        summary.loc[
            (summary["method"] == "prediction") & (summary["batch_size"] == hip_max_bz),
            "time_per_sample",
        ].iloc[0]
    )
    ratio = ad1 / hip
    ax.annotate(
        f"{ratio:.0f}$\\times$",
        xy=(hip_max_bz, hip),
        xytext=(18, ad1 * 0.45),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#5A5A5A", lw=1.2),
        ha="center",
        va="bottom",
        fontsize=14,
        color="#2F4565",
        zorder=4,
    )
    ax.legend(loc="upper right", frameon=True, edgecolor="none")
    finish_axis(ax)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"AD batch-1 mean {ad1:.1f} ms; HIP batch-{hip_max_bz} mean {hip:.2f} ms; {ratio:.1f}x")


if __name__ == "__main__":
    main()
