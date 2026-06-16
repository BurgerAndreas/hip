#!/usr/bin/env python
"""Insight figures for the smoothness of the *eqv2* (baseline EquiformerV2) force field.

Reads the arrays/summaries produced by ``scripts/force_smoothness_scan.py`` for the
``eqv2`` checkpoint (tags ``eqv2_test_s0_{transition_state,reactant}_ext``) and renders
several standalone figures, each focused on one insight. No model evaluation is needed
here -- everything is computed from the saved arrays, so this runs on CPU in seconds.

Key fact this set of figures is built around: for eqv2's *direct, (nearly) non-conservative*
forces, the autograd Hessian is H = -dF/dx (the Jacobian of the force field). Its reliability
is therefore entirely governed by how smooth F(x) is.

Figures:
  1. fd_convergence      - autograd-H vs finite-difference error vs step size h (the headline:
                           a smooth field would track O(h^2) down to the float-noise floor).
  2. force_spectrum      - power spectrum of the directional force d.F (broadband high-frequency
                           content == roughness).
  3. conservativeness    - directional force d.F vs -dE/dl along the scan, and their residual
                           (non-conservativeness of the direct forces).
  4. hessian_asymmetry   - heatmaps of the autograd Hessian, its symmetric and antisymmetric
                           parts (a true PES Hessian is symmetric; the differentiated force
                           field is not).
  5. vib_eigenspectrum   - mass-weighted, Eckart-projected vibrational eigenvalues from the
                           (symmetrised) autograd Hessian, and the implied stationary-point
                           classification (TS should have exactly one negative mode).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from hip.frequency_analysis import analyze_frequencies_torch
from plot_style import AD_COLOR, ACCENT_COLOR, GUIDE_COLOR, GUIDE_LINE_WIDTH, HIP_COLOR, LINE_WIDTH, THIN_LINE_WIDTH, finish_axis

GEOMS = ["transition_state", "reactant"]
GEOM_LABEL = {"transition_state": "transition state (saddle)", "reactant": "reactant (minimum)"}
DIRS = ["lowest_hess", "random"]
DIR_COLOR = {"lowest_hess": AD_COLOR, "random": HIP_COLOR}
MODEL = "EquiformerV2 baseline (eqv2.ckpt)"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_summary(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    s["_atomic_numbers"] = np.array([int(x) for x in re.findall(r"-?\d+", s["atomic_numbers"])])
    coords = np.array([float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", s["coords0"])])
    s["_coords0"] = coords.reshape(-1, 3)
    return s


def _load(run_dir: Path, geom: str) -> tuple[dict, dict]:
    tag = f"eqv2_test_s0_{geom}_ext"
    npz = dict(np.load(run_dir / f"{tag}_arrays.npz"))
    summary = _parse_summary(run_dir / f"{tag}_summary.json")
    return npz, summary


# ---------------------------------------------------------------------------
def fig_fd_convergence(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, geom in zip(axes, GEOMS):
        npz, summ = data[geom]
        h = npz["h_values"]
        # ax.loglog(h, npz["dir_err"], "o-", color="C0", ms=4,
        #           label=r"directional  $\|H_{fd}\,d - H d\| / \|H d\|$")
        sns.lineplot(
            x=npz["h_full"],
            y=npz["full_err"],
            ax=ax,
            marker="s",
            color=HIP_COLOR,
            label=r"full Jacobian  $\|H_{fd} - H\| / \|H\|$",
        )
        # ax.loglog(h, npz["noise_floor"], ":", color="grey",
        #           label=r"$\sim$ float-noise floor ($\epsilon|F|/h$)")
        # O(h^2) guide anchored at the smallest-h directional point
        anchor = npz["dir_err"][0]
        sns.lineplot(x=h, y=anchor * (h / h[0]) ** 2, ax=ax, color=GUIDE_COLOR, linestyle="--", linewidth=GUIDE_LINE_WIDTH, alpha=0.7, label=r"$O(h^2)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{GEOM_LABEL[geom]}", fontsize=10)
        ax.set_xlabel(r"FD step size  $h$  [$\AA$]")
        finish_axis(ax)
    axes[0].set_ylabel("relative error vs autograd Hessian")
    axes[0].legend(fontsize=8, loc="lower right", frameon=True, edgecolor="none")
    # fig.suptitle(
    #     f"Autograd Hessian never converges to finite differences — {MODEL}\n"
    #     "flat & far above the noise floor (ignores the $O(h^2)$ guide) ⇒ force field is not smooth",
    #     fontsize=11,
    # )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_fd_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_force_spectrum(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, geom in zip(axes, GEOMS):
        npz, summ = data[geom]
        for kind in DIRS:
            freqs = npz[f"scan_{kind}_freqs"]
            mag = npz[f"scan_{kind}_mag"]
            hf = summ["hf_power_fraction"][kind]
            sns.lineplot(x=freqs, y=mag + 1e-30, ax=ax, lw=THIN_LINE_WIDTH, color=DIR_COLOR[kind], label=f"{kind}  (HF frac={hf:.1e})")
        ax.set_yscale("log")
        ax.axvline(20.0, color="grey", ls=":", lw=1)
        ax.set_title(GEOM_LABEL[geom], fontsize=10)
        ax.set_xlabel("spatial frequency [cycles/$\\AA$]")
        finish_axis(ax)
        ax.legend(fontsize=8, frameon=True, edgecolor="none")
    axes[0].set_ylabel(r"$|\mathrm{FFT}(d\cdot F)|$")
    fig.suptitle(
        f"Force power spectrum along a line scan"
    )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_force_spectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_conservativeness(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for col, geom in enumerate(GEOMS):
        npz, summ = data[geom]
        kind = "random"  # cleaner, well-excited direction
        lam = npz[f"scan_{kind}_lam"]
        g = npz[f"scan_{kind}_g"]            # d.F  (directional force)
        E = npz[f"scan_{kind}_E"]
        dl = float(lam[1] - lam[0])
        minus_dEdl = -np.gradient(E, dl)     # for a conservative field == g
        resid = g - minus_dEdl

        top, bot = axes[0, col], axes[1, col]
        sns.lineplot(x=lam, y=g, ax=top, lw=LINE_WIDTH, color=AD_COLOR, label=r"$d\cdot F$ (directional force)")
        sns.lineplot(x=lam, y=minus_dEdl, ax=top, lw=LINE_WIDTH, ls="--", color=HIP_COLOR, label=r"$-dE/d\lambda$ (from energy)")
        top.set_title(f"{GEOM_LABEL[geom]}", fontsize=10)
        finish_axis(top)
        top.legend(fontsize=10, frameon=True, edgecolor="none")

        sns.lineplot(x=lam, y=resid, ax=bot, lw=THIN_LINE_WIDTH, color=ACCENT_COLOR)
        bot.axhline(0, color="k", lw=GUIDE_LINE_WIDTH, alpha=0.5)
        bot.set_xlabel(r"random displacement $\lambda$ [$\AA$]")
        finish_axis(bot)
    axes[0, 0].set_ylabel("Force [eV/$\\AA$]")
    axes[1, 0].set_ylabel(r"residual $d\cdot F\  -\  (-dE/d\lambda)$ [eV/$\AA$]")
    # fig.suptitle(
    #     f"Are the direct forces conservative? — {MODEL}\n"
    #     r"gap between $d\cdot F$ and $-dE/d\lambda$ = non-conservativeness (random direction)",
    #     fontsize=11,
    # )
    fig.tight_layout(pad=0.01, h_pad=0.5)
    p = out / "eqv2_conservativeness.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_hessian_asymmetry(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, geom in enumerate(GEOMS):
        npz, summ = data[geom]
        H = npz["H_autograd"]
        sym = 0.5 * (H + H.T)
        anti = 0.5 * (H - H.T)
        vmax = float(np.abs(H).max())
        panels = [(H, "autograd H  $(-dF/dx)$"), (sym, "symmetric part"), (anti, "antisymmetric part")]
        for col, (M, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{GEOM_LABEL[geom]}\nrow index (3N)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        anti_frac = float(np.linalg.norm(anti) / (np.linalg.norm(H) + 1e-30))
        axes[row, 2].text(0.5, -0.14, f"antisym fraction = {anti_frac:.3f}",
                          transform=axes[row, 2].transAxes, ha="center", fontsize=9, color=HIP_COLOR)
    fig.suptitle(
        f"Autograd Hessian symmetry",
        fontsize=11,
    )
    fig.tight_layout(pad=0.01, rect=[0, 0.02, 1, 1])
    p = out / "eqv2_hessian_asymmetry.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_vib_eigenspectrum(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for ax, geom in zip(axes, GEOMS):
        npz, summ = data[geom]
        H = torch.tensor(npz["H_autograd"], dtype=torch.float64)
        Hsym = 0.5 * (H + H.T)  # symmetrise before physical analysis
        coords = torch.tensor(summ["_coords0"], dtype=torch.float64)
        z = torch.tensor(summ["_atomic_numbers"], dtype=torch.long)
        freq = analyze_frequencies_torch(Hsym, coords, z)
        ev = freq["eigvals"].detach().cpu().numpy()
        neg = int(freq["neg_num"])
        idx = np.arange(ev.size)
        colors = np.where(ev < 0, HIP_COLOR, AD_COLOR)
        sns.barplot(x=idx, y=ev, ax=ax, palette=colors.tolist(), hue=idx, legend=False)
        ax.axhline(0, color="k", lw=0.7)
        expected = 1 if geom == "transition_state" else 0
        ok = "✓" if neg == expected else "✗"
        ax.set_title(f"{GEOM_LABEL[geom]}\nnegative modes: {neg} (expected {expected}) {ok}", fontsize=10)
        ax.set_xlabel("mode index (sorted)")
        finish_axis(ax)
        rows.append((geom, neg, expected))
    axes[0].set_ylabel("Eigenvalue (mass-weighted, Eckart-projected)")
    # fig.suptitle(
    #     f"Stationary-point classification from the autograd Hessian — {MODEL}\n"
    #     "red = negative (imaginary) modes; non-smoothness can inject/remove spurious curvature",
    #     fontsize=11,
    # )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_vib_eigenspectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, rows


def fig_residual_source(data: dict, out: Path) -> Path:
    """Show that the conservativeness-residual spikes come from the energy channel
    (-dE/dl), not the directly-predicted force (d.F)."""
    from scipy.signal import savgol_filter

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    kind = "random"  # well-excited direction
    for col, geom in enumerate(GEOMS):
        npz, summ = data[geom]
        lam = npz[f"scan_{kind}_lam"]
        g = npz[f"scan_{kind}_g"]          # d.F  (direct force)
        E = npz[f"scan_{kind}_E"]
        dl = float(lam[1] - lam[0])
        mdedl = -np.gradient(E, dl)        # energy-derived force
        win = 51
        g_hf = g - savgol_filter(g, win, 3)
        m_hf = mdedl - savgol_filter(mdedl, win, 3)
        ratio = m_hf.std() / (g_hf.std() + 1e-30)

        # row 1: the two force signals
        a0 = axes[0, col]
        a0.plot(lam, g, lw=1.1, color="C0", label=r"$d\cdot F$ (direct force) — smooth")
        a0.plot(lam, mdedl, lw=1.0, ls="--", color="C3", label=r"$-dE/d\lambda$ (energy) — spiky")
        a0.set_title(f"{GEOM_LABEL[geom]}", fontsize=10)
        a0.grid(alpha=0.3)
        a0.legend(fontsize=8)

        # row 2: high-pass (spike) content of each channel
        a1 = axes[1, col]
        a1.plot(lam, m_hf, lw=0.8, color="C3", label=r"HF$(-dE/d\lambda)$ energy")
        a1.plot(lam, g_hf, lw=0.9, color="C0", label=r"HF$(d\cdot F)$ force")
        a1.set_title(f"spike content: energy is {ratio:.0f}× the force", fontsize=10)
        a1.grid(alpha=0.3)
        a1.legend(fontsize=8)

        # row 3: zoom on the largest energy spike -> E steps while d.F stays smooth
        i = int(np.argmax(np.abs(m_hf)))
        lo, hi = max(0, i - 40), min(lam.size, i + 41)
        a2 = axes[2, col]
        a2.plot(lam[lo:hi], E[lo:hi] - E[lo:hi].min(), color="C2", lw=1.2, marker=".", ms=3,
                label=r"$E(\lambda)$ (left)")
        a2.set_ylabel("E - min(E) [eV]", color="C2")
        a2.tick_params(axis="y", labelcolor="C2")
        a2b = a2.twinx()
        a2b.plot(lam[lo:hi], g[lo:hi], color="C0", lw=1.4, label=r"$d\cdot F$ (right)")
        a2b.set_ylabel(r"$d\cdot F$ [eV/$\AA$]", color="C0")
        a2b.tick_params(axis="y", labelcolor="C0")
        a2.axvline(lam[i], color="C3", ls=":", lw=1)
        a2.set_title(r"zoom at largest spike: $E$ has a kink, $d\cdot F$ does not", fontsize=10)
        a2.set_xlabel(r"displacement $\lambda$ [$\AA$]")
        a2.grid(alpha=0.3)
    axes[0, 0].set_ylabel("force [eV/$\\AA$]")
    axes[1, 0].set_ylabel("high-pass force [eV/$\\AA$]")
    fig.suptitle(
        f"Residual spikes come from the energy head, not the forces — {MODEL}\n"
        r"the direct force $d\cdot F$ is smooth; $-dE/d\lambda$ is spiky because the predicted energy is slightly kinky",
        fontsize=11,
    )
    fig.tight_layout(pad=0.01, rect=[0, 0, 1, 0.97])
    p = out / "eqv2_residual_source.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_autograd_fidelity(data: dict, out: Path) -> tuple[Path, list]:
    """Does the autograd Hessian capture the model's *own* curvature?

    Compares, for each geometry/direction, the autograd directional curvature
    c_ag = d^T H_sym d  against the curvature read off the smooth fine line scan
    c_scan = -dg/dl (g = d.F). For a faithful derivative of a smooth force field
    these agree; large relative gaps only appear for soft (near-zero-curvature)
    modes where the *relative* metric is ill-conditioned.
    """
    from scipy.signal import savgol_filter

    rows = []
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    # ---- top row: tangent-slope overlay on the well-excited (random) direction ----
    for col, geom in enumerate(GEOMS):
        npz, summ = data[geom]
        H = npz["H_autograd"]; Hs = 0.5 * (H + H.T)
        kind = "random"
        lam = npz[f"scan_{kind}_lam"]; g = npz[f"scan_{kind}_g"]; d = npz[f"scan_{kind}_d"]
        dl = float(lam[1] - lam[0]); c = lam.size // 2
        gs_smooth = savgol_filter(g, 51, 3)
        c_scan = -float(np.gradient(gs_smooth, dl)[c])     # -dg/dl
        c_ag = float(d @ Hs @ d)                            # d^T H d
        g0 = float(gs_smooth[c])
        ax = fig.add_subplot(gs[0, col])
        m = np.abs(lam) <= 0.05
        ax.plot(lam[m], g[m], lw=1.4, color="C0", label=r"$d\cdot F$ (scan)")
        ax.plot(lam[m], g0 - c_scan * lam[m], "--", color="C2", lw=1.6,
                label=fr"scan slope $-dg/d\lambda$ = {c_scan:.2f}")
        ax.plot(lam[m], g0 - c_ag * lam[m], ":", color="C3", lw=1.8,
                label=fr"autograd $d^THd$ = {c_ag:.2f}")
        rel = abs(c_ag - c_scan) / (abs(c_scan) + 1e-12)
        ax.set_title(f"{GEOM_LABEL[geom]} — random direction\n"
                     f"curvature agreement: {rel:.1%}", fontsize=10)
        ax.set_xlabel(r"displacement $\lambda$ [$\AA$]")
        ax.set_ylabel(r"$d\cdot F$ [eV/$\AA$]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # ---- bottom row: bar comparison across geometries & directions ----
    ax = fig.add_subplot(gs[1, :])
    labels, c_ags, c_scans, rels, hf = [], [], [], [], []
    for geom in GEOMS:
        npz, summ = data[geom]
        H = npz["H_autograd"]; Hs = 0.5 * (H + H.T)
        for kind in DIRS:
            lam = npz[f"scan_{kind}_lam"]; g = npz[f"scan_{kind}_g"]; d = npz[f"scan_{kind}_d"]
            dl = float(lam[1] - lam[0]); c = lam.size // 2
            gss = savgol_filter(g, 51, 3)
            cs = -float(np.gradient(gss, dl)[c]); ca = float(d @ Hs @ d)
            g_hf = (g - gss).std()
            labels.append(f"{geom.split('_')[0]}\n{kind}")
            c_ags.append(ca); c_scans.append(cs); rels.append(abs(ca - cs) / (abs(cs) + 1e-12)); hf.append(g_hf)
            rows.append((geom, kind, ca, cs, abs(ca - cs) / (abs(cs) + 1e-12), g_hf))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, c_scans, w, color="C2", label=r"scan  $-dg/d\lambda$")
    ax.bar(x + w / 2, c_ags, w, color="C3", label=r"autograd  $d^THd$")
    for xi, (ca, cs, r) in enumerate(zip(c_ags, c_scans, rels)):
        top = max(ca, cs)
        tag = f"{r:.1%}" if abs(cs) > 1.0 else f"{r:.0%}\n(soft)"
        ax.annotate(tag, (xi, top), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="k")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("directional curvature [eV/$\\AA^2$]")
    ax.set_title("autograd curvature vs smooth-scan curvature "
                 "(stiff/random dirs agree to ~1–2%; soft modes are ill-conditioned in the relative metric)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Autograd Hessian faithfully captures the model's own curvature — {MODEL}\n"
        r"smooth forces (HF$(d\cdot F)\sim10^{-5}$) ⇒ autograd $d^THd$ matches finite differences of the scan",
        fontsize=11,
    )
    fig.tight_layout(pad=0.01, rect=[0, 0, 1, 0.96])
    p = out / "eqv2_autograd_fidelity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=_project_root() / "runs" / "force_smoothness")
    ap.add_argument("--out-dir", type=Path, default=_project_root() / "plots" / "force_smoothness" / "eqv2_figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = {g: _load(args.run_dir, g) for g in GEOMS}

    written = []
    written.append(fig_fd_convergence(data, args.out_dir))
    written.append(fig_force_spectrum(data, args.out_dir))
    written.append(fig_conservativeness(data, args.out_dir))
    written.append(fig_hessian_asymmetry(data, args.out_dir))
    written.append(fig_residual_source(data, args.out_dir))
    fid_png, fid_rows = fig_autograd_fidelity(data, args.out_dir)
    written.append(fid_png)
    vib_png, vib_rows = fig_vib_eigenspectrum(data, args.out_dir)
    written.append(vib_png)

    print("Autograd vs smooth-scan directional curvature:")
    for geom, kind, ca, cs, rel, ghf in fid_rows:
        print(f"  {geom:17s} {kind:11s}: d^THd={ca:+8.3f}  -dg/dl={cs:+8.3f}  "
              f"rel={rel:7.2%}  HF(d.F) std={ghf:.2e}")
    print("\nNegative-mode check (autograd Hessian, symmetrised):")
    for geom, neg, exp in vib_rows:
        print(f"  {geom:18s}: {neg} negative (expected {exp})")
    print("\nWrote figures:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
