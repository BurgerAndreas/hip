"""
Plot (a) the predicted Hessian (b) the ground truth DFT Hessian (c) the difference between the two as heatmaps
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as TGDataLoader
import seaborn as sns

from hip.training_module import PotentialModule, SchemaUniformDataset
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from nets.prediction_utils import compute_extra_props
from ocpmodels.hessian_graph_transform import HessianGraphTransform

from hip.colours import (
    ANNOTATION_BOLD_FONT_SIZE,
    ANNOTATION_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
)


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    n_comp = forces.reshape(-1).shape[0]
    hess = []
    for f in forces.reshape(-1):
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)

    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def load_hessians(ckpt_path, dataset_path, hessian_method, sample_idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]

    model = (
        PotentialModule.load_from_checkpoint(ckpt_path, strict=False)
        .potential.to(device)
        .eval()
    )

    do_autograd = hessian_method == "autograd"

    if hessian_method == "predict":
        transform = HessianGraphTransform(
            cutoff=model.cutoff,
            cutoff_hessian=model.cutoff_hessian,
            max_neighbors=model.max_neighbors,
            use_pbc=model.use_pbc,
        )
    else:
        transform = None

    dataset = LmdbDataset(fix_dataset_path(dataset_path), transform=transform)
    dataset = SchemaUniformDataset(dataset)
    dataset = Subset(dataset, indices=[sample_idx])
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    target_batch = next(iter(dataloader))
    batch = target_batch.to(device)
    batch = compute_extra_props(batch)

    n_atoms = batch.pos.shape[0]

    if model_name == "LEFTNet":
        batch.pos.requires_grad_()
        energy_model, force_model = model.forward_autograd(batch)
        hessian_model = compute_hessian(batch.pos, energy_model, force_model)
    elif "equiformer" in model_name.lower():
        if do_autograd:
            batch.pos.requires_grad_()
            energy_model, force_model, _ = model.forward(
                batch, otf_graph=False, hessian=False
            )
            hessian_model = compute_hessian(batch.pos, energy_model, force_model)
        else:
            energy_model, force_model, out = model.forward(
                batch, otf_graph=False, hessian=True, add_props=True
            )
            hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
    else:
        batch.pos.requires_grad_()
        energy_model, force_model = model.forward(batch)
        hessian_model = compute_hessian(batch.pos, energy_model, force_model)

    hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)

    return (
        hessian_model.detach().cpu().numpy(),
        hessian_true.detach().cpu().numpy(),
        {
            "n_atoms": int(n_atoms),
            "model_name": model_name,
        },
    )


def plot_heatmaps(h_pred, h_true, save_path=None, absolute=False, show_relative=False):
    if absolute:
        h_pred = np.abs(h_pred)
        h_true = np.abs(h_true)
        diff = np.abs(h_pred - h_true)
    else:
        diff = h_pred - h_true

    vmax_pt = float(max(np.abs(h_pred).max(), np.abs(h_true).max()))
    if absolute:
        vmin_pt = 0.0
    else:
        vmin_pt = -vmax_pt
    vmax_diff = float(np.abs(diff).max())
    if absolute:
        vmin_diff = 0.0
    else:
        vmin_diff = -vmax_diff

    # Prepare figure and axes (optionally include relative difference subplot)
    if show_relative:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    cmap_pred_true = "Reds" if absolute else "RdBu_r"
    cmap_diff = "Reds" if absolute else "RdBu_r"
    cmap_rel = "Reds"

    sns.heatmap(
        h_pred,
        ax=axes[0],
        cmap=cmap_pred_true,
        vmin=vmin_pt,
        vmax=vmax_pt,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.7, "pad": 0.01},
    )
    sns.heatmap(
        h_true,
        ax=axes[1],
        cmap=cmap_pred_true,
        vmin=vmin_pt,
        vmax=vmax_pt,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.7, "pad": 0.01},
    )
    sns.heatmap(
        diff,
        ax=axes[2],
        cmap=cmap_diff,
        vmin=vmin_diff,
        vmax=vmax_diff,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.7, "pad": 0.01},
    )

    if show_relative:
        # Relative difference: |pred - true| / |true|, add epsilon to avoid division by zero
        denom = np.maximum(np.abs(h_true), 1e-12)
        rel = np.abs(h_pred - h_true) / denom
        vmin_rel = 0.0
        vmax_rel = float(np.abs(rel).max())

        sns.heatmap(
            rel,
            ax=axes[3],
            cmap=cmap_rel,
            vmin=vmin_rel,
            vmax=vmax_rel,
            square=True,
            cbar=True,
            cbar_kws={"shrink": 0.7, "pad": 0.01},
        )

    # Add subplot labels outside, further top-left of each axes
    labels = ["a", "b", "c", "d"] if show_relative else ["a", "b", "c"]
    for ax, label in zip(axes, labels):
        ax.text(
            -0.10,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
            clip_on=False,
        )

    axes[0].set_title("Predicted Hessian", pad=2)
    axes[1].set_title("Ground Truth DFT Hessian", pad=2)
    axes[2].set_title("Absolute Difference", pad=2)
    if show_relative:
        axes[3].set_title("(d) Relative Difference", pad=2)
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("3N Degrees of Freedom")
        ax.set_ylabel("")
    axes[0].set_ylabel("3N Degrees of Freedom")  # only leftmost axis
    ax.tick_params(axis="both", which="both", labelsize=6, length=2, pad=1)

    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    print(f"Saved heatmaps to {save_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hessian heatmaps for a single sample"
    )
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        # default="ckpt/eqv2.ckpt"
        # default="/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746.ckpt",
        # default="/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetmaebz128-676539-20250906-003158.ckpt",
        default="/ssd/Code/hip/ckpt/hesspred_v1.ckpt",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train_big.lmdb, RGD1.lmdb)",
    )
    parser.add_argument(
        "--hessian_method",
        type=str,
        default="predict",
        help="Hessian computation method: autograd or predict",
    )
    parser.add_argument(
        "--show_relative",
        action="store_true",
        help="Include relative difference subplot",
    )

    args = parser.parse_args()

    for sample_idx in [0, 1234, 12345]:
        h_pred, h_true, meta = load_hessians(
            ckpt_path=args.ckpt_path,
            dataset_path=args.dataset,
            hessian_method=args.hessian_method,
            sample_idx=sample_idx,
        )

        for absolute in [True, False]:
            # name = (
            #     f"{os.path.basename(args.ckpt_path).split('.')[0]}_"
            #     f"{os.path.basename(args.dataset).split('.')[0]}_"
            #     f"{args.hessian_method}_idx{args.sample_idx}"
            # )
            save_path = os.path.join(
                "plots",
                "hessian_heatmap",
                f"hessian_heatmap_idx{sample_idx}{'_abs' if absolute else ''}.png",
            )

            plot_heatmaps(
                h_pred,
                h_true,
                save_path=save_path,
                absolute=absolute,
                show_relative=args.show_relative,
            )


if __name__ == "__main__":
    main()
