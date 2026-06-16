#!/usr/bin/env python
"""Evaluate autograd Hessians on selected force-scan center geometries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
from hip.frequency_analysis import analyze_frequencies_np
from hip.path_config import fix_dataset_path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_SCAN_DIR = project_root() / "runs" / "t1x_val_force_spectra_100x2x51"


def rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def selected_indices(manifest_path: Path) -> list[tuple[int, int]]:
    manifest = pd.read_csv(manifest_path)
    frame = (
        manifest[["geom_rank", "dataset_idx"]]
        .drop_duplicates()
        .sort_values("geom_rank")
        .reset_index(drop=True)
    )
    return [(int(row.geom_rank), int(row.dataset_idx)) for row in frame.itertuples(index=False)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "ckpt" / "eqv2_orig.ckpt")
    parser.add_argument("--dataset", default="ts1x-val.lmdb")
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tag", default="eqv2_orig")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = args.output_dir or args.scan_dir / "ad_hessians" / args.tag
    sample_dir = output_dir / "sample_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    indices = selected_indices(args.scan_dir / "scan_manifest.csv")
    dataset_path = fix_dataset_path(args.dataset)
    dataset = LmdbDataset(dataset_path)
    calc = EquiformerTorchCalculator(
        checkpoint_path=str(args.checkpoint),
        hessian_method="autograd",
        device=device,
    )

    rows: list[dict[str, object]] = []
    for sample_idx, (geom_rank, dataset_idx) in enumerate(indices):
        out_path = sample_dir / f"g{geom_rank:04d}_idx{dataset_idx:06d}_{args.tag}_ad_hessian.npz"
        if out_path.exists() and not args.overwrite:
            with np.load(out_path) as data:
                rows.append(json.loads(str(data["metrics_json"].item())))
            print(f"[{sample_idx + 1:3d}/{len(indices)}] skipped existing {out_path.name}", flush=True)
            continue

        data = dataset[dataset_idx]
        coords = data.pos.detach().cpu().numpy().astype(np.float64)
        z = data.z.detach().cpu().numpy().astype(np.int64)
        symbols = [Z_TO_ATOM_SYMBOL[int(zz)] for zz in z]
        n_atoms = int(len(z))
        n3 = 3 * n_atoms

        out = calc.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=device),
            atomic_nums=torch.tensor(z, dtype=torch.long, device=device),
            hessian_method="autograd",
            do_hessian=True,
        )
        energy_model = float(out["energy"].detach().cpu().reshape(-1)[0].item())
        forces_model = out["forces"].detach().cpu().numpy().reshape(n_atoms, 3).astype(np.float64)
        hessian_ad = out["hessian"].detach().cpu().numpy().reshape(n3, n3).astype(np.float64)
        hessian_sym = 0.5 * (hessian_ad + hessian_ad.T)

        hessian_dft = data.hessian.detach().cpu().numpy().reshape(n3, n3).astype(np.float64)
        forces_dft = data.forces.detach().cpu().numpy().reshape(n_atoms, 3).astype(np.float64)
        energy_true = float((data.energy if "energy" in data else data.ae).detach().cpu().reshape(-1)[0].item())

        eigvals_model = np.linalg.eigvalsh(hessian_sym)
        eigvals_dft = np.linalg.eigvalsh(hessian_dft)
        vib_model = analyze_frequencies_np(hessian_sym, coords, symbols)
        vib_dft = analyze_frequencies_np(hessian_dft, coords, symbols)

        force_diff = forces_model - forces_dft
        metrics = {
            "sample_idx": sample_idx,
            "geom_rank": geom_rank,
            "dataset_idx": dataset_idx,
            "natoms": n_atoms,
            "energy_model": energy_model,
            "energy_true": energy_true,
            "energy_error": abs(energy_model - energy_true),
            "forces_error": float(np.mean(np.abs(force_diff))),
            "force_l2_error": float(np.linalg.norm(force_diff.reshape(-1))),
            "force_model_norm": float(np.linalg.norm(forces_model.reshape(-1))),
            "force_true_norm": float(np.linalg.norm(forces_dft.reshape(-1))),
            "hessian_error": float(np.mean(np.abs(hessian_sym - hessian_dft))),
            "hess_rel_err": rel_fro(hessian_sym, hessian_dft),
            "hessian_model_fro_norm": float(np.linalg.norm(hessian_sym)),
            "hessian_true_fro_norm": float(np.linalg.norm(hessian_dft)),
            "hessian_asym_rel": float(np.linalg.norm(hessian_ad - hessian_ad.T) / (np.linalg.norm(hessian_ad) + 1e-30)),
            "hessian_asym_mae": float(np.mean(np.abs(hessian_ad - hessian_ad.T))),
            "eigval_mae": float(np.mean(np.abs(np.sort(eigvals_model) - np.sort(eigvals_dft)))),
            "vib_eigval_mae": float(np.mean(np.abs(np.sort(vib_model["eigvals"]) - np.sort(vib_dft["eigvals"])))),
            "model_neg_num": int(vib_model["neg_num"]),
            "true_neg_num": int(vib_dft["neg_num"]),
            "neg_num_agree": int(int(vib_model["neg_num"]) == int(vib_dft["neg_num"])),
            "sample_npz_path": str(out_path.resolve()),
        }

        np.savez_compressed(
            out_path,
            geom_rank=np.asarray(geom_rank, dtype=np.int64),
            dataset_idx=np.asarray(dataset_idx, dtype=np.int64),
            atomic_numbers=z,
            coords_ang=coords,
            energy_model_ev=np.asarray(energy_model, dtype=np.float64),
            energy_true_ev=np.asarray(energy_true, dtype=np.float64),
            forces_model_ev_ang=forces_model,
            forces_dft_ev_ang=forces_dft,
            hessian_ad_ev_ang2=hessian_ad,
            hessian_sym_ev_ang2=hessian_sym,
            hessian_dft_ev_ang2=hessian_dft,
            metrics_json=np.asarray(json.dumps(metrics)),
        )
        rows.append(metrics)
        print(
            f"[{sample_idx + 1:3d}/{len(indices)}] idx={dataset_idx:6d} "
            f"hess_rel={metrics['hess_rel_err']:.4g} asym={metrics['hessian_asym_rel']:.4g} "
            f"neg {metrics['model_neg_num']}/{metrics['true_neg_num']}",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("geom_rank")
    csv = output_dir / f"{args.tag}_selected_ad_hessian_metrics.csv"
    df.to_csv(csv, index=False)
    df.to_parquet(output_dir / f"{args.tag}_selected_ad_hessian_metrics.parquet", index=False)

    print(f"\nWrote {len(df)} rows to {csv}")
    print(f"Median hess_rel_err: {df['hess_rel_err'].median():.6g}")
    print(f"Median hessian_asym_rel: {df['hessian_asym_rel'].median():.6g}")
    print(f"Neg-mode agreement: {df['neg_num_agree'].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
