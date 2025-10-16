import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import time

import json
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from hip.training_module import SchemaUniformDataset
from hip.frequency_analysis import analyze_frequencies_torch
from nets.prediction_utils import Z_TO_ATOM_SYMBOL
from hip.colours import HESSIAN_METHOD_TO_COLOUR

# https://plotly.com/python/templates/
PLOTLY_TEMPLATE = "plotly_white"


def time_forward_pass_with_frequency_analysis(model, batch, device="cuda", method="gpu"):
    """Times forward pass and frequency analysis on specified device."""
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass always on GPU
        ener, force, out = model.forward(
            batch.to("cuda"), otf_graph=True, hessian=True,
        )
        
        # Get hessian from output
        hessian = out.get("hessian")
        
        # Convert coordinates and atomic numbers for frequency analysis
        cart_coords = batch.pos
        atomsymbols = [Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z]
        
        # Perform frequency analysis
        if device == "cuda":
            # Use torch version for GPU
            freq_results = analyze_frequencies_torch(
                hessian=hessian,
                cart_coords=cart_coords,
                atomsymbols=atomsymbols
            )
        else:
            # Use CPU
            freq_results = analyze_frequencies_torch(
                hessian=hessian.detach().cpu(),
                cart_coords=cart_coords.detach().cpu(),
                atomsymbols=atomsymbols
            )
    
    end_time = time.time()
    time_taken = (end_time - start_time) * 1000  # Convert to ms
    
    if device == "cuda":
        memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    else:
        memory_usage = 0  # CPU doesn't have GPU memory tracking
    
    return time_taken, memory_usage, freq_results


def time_forward_pass_with_frequency_analysis_batch(model, batch, device="cuda", method="gpu", max_workers=12):
    """Times forward pass and frequency analysis for a batch on specified device."""
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass always on GPU
        ener, force, out = model.forward(
            batch.to("cuda"), otf_graph=True, hessian=True,
        )
        
        # Get hessian from output
        hessian = out.get("hessian")
        
        # Process each sample in the batch individually
        B = batch.batch.max() + 1
        natoms = batch.natoms
        # all_pos = batch.pos.reshape(-1, 3).to(hessian.device)
        
        # Calculate pointer for hessian matrices
        numels = batch.natoms.pow(2).mul(9)
        ptr_hessian = torch.cat([torch.tensor([0], device=numels.device), numels], dim=0)
        ptr_hessian = torch.cumsum(ptr_hessian, dim=0)
        hessian = hessian.view(-1)
        
        # Output containers for min eigenvalues and corresponding eigenvectors
        # Keep eigenvalues as a fixed-size tensor [B]; eigenvectors as a list of length-B tensors
        min_eigs = torch.empty(B, dtype=hessian.dtype, device="cpu")
        min_vecs = [None] * B
        
        if method == "gpu":
            for _b in range(B):
                _start = ptr_hessian[_b].item()
                ND = natoms[_b] * 3
                _numel = ND**2
                _end = _numel + _start
                hessian_b = hessian[_start:_end].reshape(ND, ND)
                # freq_results = analyze_frequencies_torch(
                #     hessian=hessian_b,
                #     cart_coords=cart_coords,
                #     atomsymbols=atomsymbols
                # )
                # GPU eigendecomposition
                w, v = torch.linalg.eigh(hessian_b)
                # smallest eigenpair (ascending)
                w0 = w[0].detach().to("cpu")
                v0 = v[:, 0].detach().to("cpu")
                min_eigs[_b] = w0
                min_vecs[_b] = v0
        elif method == "gpu_padding":
            # Pad each matrix to ND_MAX x ND_MAX on GPU, then select eigenpair from original subspace
            ND_MAX = 23 * 3
            eps = 1e-12
            for _b in range(B):
                _start = ptr_hessian[_b].item()
                ND = natoms[_b] * 3
                _numel = ND**2
                _end = _numel + _start
                hessian_b = hessian[_start:_end].reshape(ND, ND)
                # Allocate padded matrix
                H_pad = torch.zeros((ND_MAX, ND_MAX), dtype=hessian_b.dtype, device=hessian_b.device)
                H_pad[:ND, :ND] = hessian_b
                w, V = torch.linalg.eigh(H_pad)
                # Restrict to original ND rows to avoid padded-only eigenvectors
                V_head = V[:ND, :]
                valid = (V_head.norm(dim=0) > eps)
                if valid.any():
                    w_valid = w[valid]
                    idx_valid = torch.nonzero(valid, as_tuple=False).view(-1)
                    j = torch.argmin(w_valid)
                    col = idx_valid[j]
                    v0 = V_head[:, col].detach().to("cpu")
                    w0 = w_valid[j].detach().to("cpu")
                else:
                    # Fallback should not happen; default to original top-left diag
                    w0 = torch.tensor(0.0, dtype=hessian.dtype)
                    v0 = torch.zeros(ND, dtype=hessian.dtype)
                min_eigs[_b] = w0
                min_vecs[_b] = v0
        elif method == "lobpcg":
            for _b in range(B):
                _start = ptr_hessian[_b].item()
                ND = natoms[_b] * 3
                _numel = ND**2
                _end = _numel + _start
                hessian_b = hessian[_start:_end].reshape(ND, ND)
                w, v = torch.lobpcg(hessian_b, k=2, niter=None, tol=None, largest=False)
        elif method == "cpu":
            # Prepare CPU tensors for eigendecomposition
            for _b in range(B):
                _start = ptr_hessian[_b].item()
                ND = natoms[_b] * 3
                _numel = ND**2
                _end = _numel + _start
                hessian_b = hessian[_start:_end].reshape(ND, ND)
                m = hessian_b.detach().cpu()
                w, v = torch.linalg.eigh(m)
                w0 = w[0]
                v0 = v[:, 0]
                min_eigs[_b] = w0
                min_vecs[_b] = v0
        elif method == "cpu_parallel":
            # Prepare CPU tensors for eigendecomposition
            cpu_mats = []
            for _b in range(B):
                _start = ptr_hessian[_b].item()
                ND = natoms[_b] * 3
                _numel = ND**2
                _end = _numel + _start
                hessian_b = hessian[_start:_end].reshape(ND, ND)
                cpu_mats.append(hessian_b.detach().cpu())

            workers = max_workers or os.cpu_count() or 1
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [(i, ex.submit(torch.linalg.eigh, cpu_mats[i])) for i in range(B)]
                for i, f in futures:
                    w, v = f.result()
                    ND = natoms[i] * 3
                    min_eigs[i] = w[0]
                    min_vecs[i] = v[:, 0]
    
    end_time = time.time()
    time_taken = (end_time - start_time) * 1000  # Convert to ms
    
    if device == "cuda":
        memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    else:
        memory_usage = 0  # CPU doesn't have GPU memory tracking
    
    return time_taken, memory_usage


def eigendecomp_benchmark(
    checkpoint_path,
    dataset_name,
    max_samples_per_n,
    output_dir,
    output_path,
    device="cuda",
):
    """Benchmarks forward pass + frequency analysis on GPU vs CPU."""
    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Get indices by number of atoms
    fixed_dataset_path = Path(fix_dataset_path(dataset_name))
    indices_file = (
        fixed_dataset_path.parent
        / f"{fixed_dataset_path.stem}_indices_by_natoms_small.json"
    )

    if indices_file.exists():
        print(f"Loading indices from {indices_file}")
        with open(indices_file, "r") as f:
            indices_by_natoms = json.load(f)
    else:
        print(f"Indices file not found. Generating new indices for {dataset_name}")
        from scripts.speed_comparison import save_idx_by_natoms
        results = save_idx_by_natoms({"dataset_path": dataset_name})
        indices_by_natoms = results["small_idx"]

    # Prepare dataset and dataloader
    dataset = LmdbDataset(fix_dataset_path(dataset_name))
    dataset = SchemaUniformDataset(dataset)

    # Warm up the model
    print("Warming up model...")
    loader = TGDataLoader(
        dataset, batch_size=1, shuffle=False, 
    )
    for i, sample in enumerate(loader):
        batch = sample.to(device)
        time_forward_pass_with_frequency_analysis(model, batch, device)
        torch.cuda.empty_cache() if device == "cuda" else None
        if i > 5:
            break
    print("Model warmed up")

    results = []

    for n_atoms, indices in tqdm(indices_by_natoms.items(), desc="Processing N_atoms"):
        if len(indices) == 0:
            continue
        n_atoms = int(n_atoms)

        # Limit number of samples
        indices_to_test = indices[:max_samples_per_n]

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(
            subset, batch_size=1, shuffle=False, 
        )

        for _batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            batch = _batch.clone()

            # Time GPU version
            time_gpu, mem_gpu, freq_results_gpu = time_forward_pass_with_frequency_analysis(
                model, batch, "cuda", method="gpu"
            )
            results.append(
                {
                    "n_atoms": n_atoms,
                    "device": "cuda",
                    "method": "gpu",
                    "time": time_gpu,
                    "memory": mem_gpu,
                    "neg_num": freq_results_gpu["neg_num"],
                    "natoms": freq_results_gpu["natoms"],
                }
            )

            # Clear memory
            torch.cuda.empty_cache()

            # Time CPU version
            time_cpu, mem_cpu, freq_results_cpu = time_forward_pass_with_frequency_analysis(
                model, batch, "cpu", method="cpu"
            )
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "cpu",
                    "device": "cpu",
                    "time": time_cpu,
                    "memory": mem_cpu,
                    "neg_num": freq_results_cpu["neg_num"],
                    "natoms": freq_results_cpu["natoms"],
                }
            )

            # # Clear memory
            # torch.cuda.empty_cache()
            
            # # Time CPU-parallel version
            # time_cpu, mem_cpu, freq_results_cpu = time_forward_pass_with_frequency_analysis(
            #     model, batch, "cpu", method="cpu_parallel"
            # )
            # results.append(
            #     {
            #         "n_atoms": n_atoms,
            #         "method": "cpu_parallel",
            #         "device": "cpu",
            #         "time": time_cpu,
            #         "memory": mem_cpu,
            #         "neg_num": freq_results_cpu["neg_num"],
            #         "natoms": freq_results_cpu["natoms"],
            #     }
            # )

            # Clear memory
            torch.cuda.empty_cache()

    # Save results
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return results_df


def eigendecomp_batchsize_benchmark(
    checkpoint_path,
    dataset_name,
    batch_sizes=(1, 2, 4, 8, 16, 32),
    num_batches=10,
    device="cuda",
    output_path="./results_eigendecomp/eigendecomp_batchsize.csv",
):
    """Benchmark eigendecomposition speed vs batch size using random batches."""
    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Prepare dataset
    dataset = LmdbDataset(fix_dataset_path(dataset_name))
    dataset = SchemaUniformDataset(dataset)

    # Light warm-up
    warm_loader = TGDataLoader(
        dataset, batch_size=1, shuffle=True, 
    )
    for i, sample in enumerate(warm_loader):
        batch = sample.to(device)
        time_forward_pass_with_frequency_analysis(model, batch, device)
        torch.cuda.empty_cache() if device == "cuda" else None
        if i >= 3:
            break
    print("Model warmed up")

    results = []
    dataset_len = len(dataset)
    
    for bsz in batch_sizes:
        print(f"\n# Batch size: {bsz}")
        # Prepare a subset with random indices
        num_needed = num_batches * bsz
        if dataset_len == 0:
            break
        rand_idx = torch.randint(
            low=0, high=dataset_len, size=(num_needed,), dtype=torch.long
        ).tolist()
        subset = Subset(dataset, rand_idx)
        loader = TGDataLoader(
            subset, batch_size=bsz, shuffle=True
        )

        torch.cuda.empty_cache() if device == "cuda" else None

        measured = 0
        for sample in loader:
            batch = sample.clone().to(device)

            # Time GPU version
            time_gpu, mem_gpu = time_forward_pass_with_frequency_analysis_batch(
                model, batch, "cuda", method="gpu"
            )
            results.append(
                {
                    "method": "gpu",
                    "time": time_gpu,
                    "memory": mem_gpu,
                    "batch_size": bsz,
                    "avg_natoms": batch.natoms.float().mean().item(),
                }
            )

            # Time GPU with padding to ND_MAX
            time_gpu_pad, mem_gpu_pad = time_forward_pass_with_frequency_analysis_batch(
                model, batch, "cuda", method="gpu_padding"
            )
            results.append(
                {
                    "method": "gpu_padding",
                    "time": time_gpu_pad,
                    "memory": mem_gpu_pad,
                    "batch_size": bsz,
                    "avg_natoms": batch.natoms.float().mean().item(),
                }
            )

            # # Time GPU sparse block (LOBPCG on block-diagonal)
            # time_gpu_sp_blk, mem_gpu_sp_blk = time_forward_pass_with_frequency_analysis_batch(
            #     model, batch, "cuda", method="lobpcg"
            # )
            # results.append(
            #     {
            #         "method": "lobpcg",
            #         "time": time_gpu_sp_blk,
            #         "memory": mem_gpu_sp_blk,
            #         "batch_size": bsz,
            #         "avg_natoms": batch.natoms.float().mean().item(),
            #     }
            # )

            # Clear memory
            torch.cuda.empty_cache()

            # Time CPU version
            time_cpu, mem_cpu = time_forward_pass_with_frequency_analysis_batch(
                model, batch, "cpu", method="cpu"
            )
            results.append(
                {
                    "method": "cpu",
                    "time": time_cpu,
                    "memory": mem_cpu,
                    "batch_size": bsz,
                    "avg_natoms": batch.natoms.float().mean().item(),
                }
            )

            # Time CPU-parallel version
            time_cpu_par, mem_cpu_par = time_forward_pass_with_frequency_analysis_batch(
                model, batch, "cpu", method="cpu_parallel"
            )
            results.append(
                {
                    "method": "cpu_parallel",
                    "time": time_cpu_par,
                    "memory": mem_cpu_par,
                    "batch_size": bsz,
                    "avg_natoms": batch.natoms.float().mean().item(),
                }
            )

            # Clear memory
            torch.cuda.empty_cache()

            msg = f"Bz={bsz}, avg natoms={batch.natoms.clone().to(torch.float32).mean():.1f}"
            msg += f", gpu={time_gpu:.1f} ms"
            msg += f", gpu_pad={time_gpu_pad:.1f} ms"
            # msg += f", sp_blk={time_gpu_sp_blk:.1f} ms"
            msg += f", cpu={time_cpu:.1f} ms"
            msg += f", cpu_par={time_cpu_par:.1f} ms"
            print(msg)

            measured += 1
            if measured >= num_batches:
                break

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Batch size benchmark results saved to {output_path}")
    return results_df


def plot_eigendecomp_comparison(results_df, output_dir):
    """Plot GPU vs CPU comparison for eigendecomposition."""
    output_dir = Path(output_dir)
    
    # Plot results for speed
    avg_times = results_df.groupby(["n_atoms", "device"])["time"].mean().unstack()

    fig = go.Figure()
    for device in avg_times.columns:
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[device],
                mode="lines+markers",
                name=device.upper(),
                line=dict(color=HESSIAN_METHOD_TO_COLOUR.get(device, None)),
            )
        )

    fig.update_layout(
        title="Eigendecomposition Speed: GPU vs CPU",
        xaxis_title="Number of Atoms (N)",
        yaxis_title="Average Time (ms)",
        legend_title="Device",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # Increase line width slightly for readability
    fig.update_traces(line=dict(width=3))
    output_path = output_dir / "eigendecomp_speed_plot.png"
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n{output_path}")

    # Plot results for memory (GPU only)
    gpu_results = results_df[results_df["device"] == "gpu"]
    if len(gpu_results) > 0:
        avg_memory = gpu_results.groupby(["n_atoms"])["memory"].mean()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory.values,
                mode="lines+markers",
                name="GPU Memory",
                line=dict(color=HESSIAN_METHOD_TO_COLOUR.get("gpu", None)),
            )
        )

        fig.update_layout(
            title="GPU Memory Usage for Eigendecomposition",
            xaxis_title="Number of Atoms (N)",
            yaxis_title="Peak Memory (MB)",
            template=PLOTLY_TEMPLATE,
            margin=dict(l=40, r=40, b=40, t=40),
        )
        fig.update_traces(line=dict(width=3))
        output_path = output_dir / "eigendecomp_memory_plot.png"
        fig.write_image(output_path, scale=2)
        print(f"Plot saved to \n{output_path}")


def plot_eigendecomp_batchsize(results_df, output_dir):
    """Plot eigendecomposition speed vs batch size."""
    output_dir = Path(output_dir)
    
    # Plot total time vs batch size
    avg_times = results_df.groupby(["batch_size", "method"])["time"].mean().unstack()
    
    fig = go.Figure()
    for method in avg_times.columns:
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=method.upper(),
                line=dict(color=HESSIAN_METHOD_TO_COLOUR.get(method, None)),
            )
        )

    fig.update_layout(
        title="Eigendecomposition Speed vs Batch Size",
        xaxis_title="Batch Size",
        yaxis_title="Average Time (ms)",
        legend_title="Device",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    fig.update_traces(line=dict(width=3))
    output_path = output_dir / "eigendecomp_batchsize_plot.png"
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n{output_path}")

    # Plot time per sample vs batch size
    fig = go.Figure()
    for method in avg_times.columns:
        # Calculate time per sample
        time_per_sample = avg_times[method] / avg_times.index
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=time_per_sample,
                mode="lines+markers",
                name=f"{method.upper()} (per sample)",
                line=dict(color=HESSIAN_METHOD_TO_COLOUR.get(method, None)),
            )
        )

    fig.update_layout(
        title="Eigendecomposition Time per Sample vs Batch Size",
        xaxis_title="Batch Size",
        yaxis_title="Average Time per Sample (ms)",
        legend_title="Device",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    fig.update_traces(line=dict(width=3))
    output_path = output_dir / "eigendecomp_batchsize_per_sample_plot.png"
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n{output_path}")

    # Plot memory usage (GPU only)
    gpu_results = results_df[results_df["method"] == "gpu"]
    if len(gpu_results) > 0:
        avg_memory = gpu_results.groupby(["batch_size"])["memory"].mean()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory.values,
                mode="lines+markers",
                name="GPU Memory",
                line=dict(color=HESSIAN_METHOD_TO_COLOUR.get("gpu", None)),
            )
        )

        fig.update_layout(
            title="GPU Memory Usage vs Batch Size",
            xaxis_title="Batch Size",
            yaxis_title="Peak Memory (MB)",
            template=PLOTLY_TEMPLATE,
            margin=dict(l=40, r=40, b=40, t=40),
        )
        fig.update_traces(line=dict(width=3))
        output_path = output_dir / "eigendecomp_batchsize_memory_plot.png"
        fig.write_image(output_path, scale=2)
        print(f"Plot saved to \n{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eigendecomposition benchmark")

    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/hesspred_v1.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name",
    )
    parser.add_argument(
        "--max_samples_per_n",
        type=int,
        default=10,
        help="Maximum number of samples per N atoms to test.",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the benchmark. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--redobz",
        type=bool,
        default=False,
        help="Redo the batch size benchmark. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test",
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    redo = args.redo
    ckpt_name = args.ckpt_path.split("/")[-1].split(".")[0]

    output_dir = "./results_eigendecomp"
    output_dir = Path(output_dir)
    output_path = (
        output_dir
        / f"{args.dataset}_eigendecomp_results_{ckpt_name}_{args.max_samples_per_n}.csv"
    )
    
    if not redo:
        if output_path.exists():
            results_df = pd.read_csv(output_path)
            print(f"Loaded existing results from {output_path}")
        else:
            redo = True

    if redo:
        results_df = eigendecomp_benchmark(
            checkpoint_path=args.ckpt_path,
            dataset_name=args.dataset,
            max_samples_per_n=args.max_samples_per_n,
            output_dir=output_dir,
            output_path=output_path,
        )

    # Plot results
    plot_eigendecomp_comparison(results_df, output_dir)

    ##############################################################
    # Batch size benchmark
    print("\nStarting batch size benchmark...")
    
    output_path_batchsize = (
        output_dir
        / f"{args.dataset}_eigendecomp_batchsize_results_{ckpt_name}_{args.max_samples_per_n}.csv"
    )
    
    if output_path_batchsize.exists() and not args.redobz:
        bz_results_df = pd.read_csv(output_path_batchsize)
        print(f"Loaded existing batch size results from {output_path_batchsize}")
    else:
        bz_results_df = eigendecomp_batchsize_benchmark(
            checkpoint_path=args.ckpt_path,
            dataset_name=args.dataset,
            batch_sizes=args.batch_sizes,
            output_path=output_path_batchsize,
        )

    # Plot batch size results
    plot_eigendecomp_batchsize(bz_results_df, output_dir)

    print("\nDone!")
