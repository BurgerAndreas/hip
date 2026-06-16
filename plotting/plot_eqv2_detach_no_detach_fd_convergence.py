#!/usr/bin/env python
"""Plot EquiformerV2 FD-vs-autodiff Hessian convergence with hard-coded values."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import HIP_COLOR, LINE_WIDTH, MARKER_SIZE, finish_axis, model_color


H_VALUES = np.array(
    [
        1.0e-05,
        1.7716e-05,
        3.13857e-05,
        5.56029e-05,
        9.85061e-05,
        1.74513e-04,
        3.09168e-04,
        5.47723e-04,
        9.70346e-04,
        1.71906e-03,
        3.0455e-03,
        5.3954e-03,
        9.5585e-03,
        1.69338e-02,
        3.0e-02,
    ],
    dtype=float,
)

DETACH_FULL_REL_ERR = np.array(
    [
        0.036393118,
        0.037625557,
        0.036817149,
        0.036380044,
        0.036306411,
        0.036156305,
        0.036152108,
        0.036147215,
        0.036120478,
        0.036118063,
        0.036092287,
        0.035999852,
        0.035801053,
        0.035191621,
        0.033338879,
    ],
    dtype=float,
)

NO_DETACH_FULL_REL_ERR = np.array(
    [
        0.0071265881,
        0.0095586396,
        0.0047428702,
        0.0033699738,
        0.0010778534,
        0.00091641092,
        0.00054895545,
        0.0002463602,
        0.00018764891,
        0.000078730562,
        0.00015200649,
        0.0003878404,
        0.00092586288,
        0.0021291019,
        0.0054443074,
    ],
    dtype=float,
)


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "plots"
        / "eqv2_fd_convergence_detach_vs_no_detach_sample35070_dense.png"
    )


def plot(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sns.lineplot(
        x=H_VALUES,
        y=DETACH_FULL_REL_ERR,
        ax=ax,
        label="detached rotations",
        color=HIP_COLOR,
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
    )
    sns.lineplot(
        x=H_VALUES,
        y=NO_DETACH_FULL_REL_ERR,
        ax=ax,
        label="differentiable rotations",
        color=model_color("AD"),
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FD step size h [Angstrom]")
    ax.set_ylabel(r"$||H_{FD}-H_{AD}||_F \ / \ ||H_{AD}||_F$")
    # ax.set_title(
    #     "EquiformerV2 FD vs autodiff Hessian\n"
    #     "TS1x-val idx=35070"
    # )
    finish_axis(ax, legend=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    plot(default_output_path())
    print(default_output_path())


if __name__ == "__main__":
    main()
