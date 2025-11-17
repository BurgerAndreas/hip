import torch
import argparse
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
import numpy as np
from pathlib import Path

from hip.training_module import PotentialModule


def compute_vjp(coords, forces, v=None, retain_graph=False):
    """Compute a single VJP: v^T @ J where J is d(forces)/d(coords).

    If v is None, a random vector is generated.
    """
    if v is None:
        v = torch.randn_like(forces)

    # VJP: v^T @ J = grad(v^T @ forces, coords)
    vjp = torch.autograd.grad(
        [(v * forces).sum()],
        [coords],
        retain_graph=retain_graph,
        create_graph=False,
    )[0]
    return vjp


def create_random_batch(n_atoms, device="cuda", seed=None):
    """Create a random batch with n_atoms atoms."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Random positions in a 1x1x1 box
    positions = torch.rand(n_atoms, 3, dtype=torch.float32) * 0.5

    # Random atomic numbers (common elements: H=1, C=6, N=7, O=8)
    atomic_numbers = torch.randint(1, 9, (n_atoms,), dtype=torch.int64)

    # Create torch_geometric Data object
    data = TGData(
        pos=positions.to(device),
        z=atomic_numbers.to(device),
        charges=atomic_numbers.to(device),
        natoms=torch.tensor([n_atoms], dtype=torch.int64, device=device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=device),
    )
    return TGBatch.from_data_list([data])


def get_model_ckpt(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name
    return model


def profile_forward_pass(model, batch, output_dir="./results_profile"):
    """Profile a forward pass."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Profiling forward pass...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            ener, force, out = model.forward(batch, otf_graph=True, hessian=False)

    # Export to Chrome trace format
    trace_file = output_dir / "forward_pass_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"Chrome trace saved to: {trace_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Forward Pass Profiling Summary")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Save detailed table to file
    table_file = output_dir / "forward_pass_table.txt"
    with open(table_file, "w") as f:
        f.write("Forward Pass Profiling Summary\n")
        f.write("=" * 80 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total"))
    print(f"\nDetailed table saved to: {table_file}")

    return prof


def profile_forward_plus_vjps(model, batch, n_vjps=2, output_dir="./results_profile"):
    """Profile a forward pass + n_vjps VJPs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\nProfiling forward pass + {n_vjps} VJPs...")

    batch.pos.requires_grad_()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        ener, force, out = model.forward(batch, otf_graph=True, hessian=False)

        # Compute n_vjps VJPs
        for i in range(n_vjps):
            retain_graph = i < n_vjps - 1
            compute_vjp(batch.pos, force, retain_graph=retain_graph)

    # Export to Chrome trace format
    trace_file = output_dir / f"forward_plus_{n_vjps}_vjps_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"Chrome trace saved to: {trace_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"Forward Pass + {n_vjps} VJPs Profiling Summary")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Save detailed table to file
    table_file = output_dir / f"forward_plus_{n_vjps}_vjps_table.txt"
    with open(table_file, "w") as f:
        f.write(f"Forward Pass + {n_vjps} VJPs Profiling Summary\n")
        f.write("=" * 80 + "\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total"))
    print(f"\nDetailed table saved to: {table_file}")

    return prof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile forward pass and forward pass + VJPs"
    )

    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/hip_v2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=100.0,
        help="Cutoff radius for message passing in angstroms.",
    )
    parser.add_argument(
        "--cutoff_hessian",
        type=float,
        default=100.0,
        help="Cutoff radius for Hessian computation in angstroms.",
    )
    parser.add_argument(
        "--n_atoms",
        type=int,
        default=5,
        help="Number of atoms (default: 5)",
    )
    parser.add_argument(
        "--n_vjps",
        type=int,
        default=2,
        help="Number of VJPs to compute (default: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_profile",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for batch generation",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model = get_model_ckpt(args.ckpt_path, device="cuda")
    model.cutoff = args.cutoff
    model.cutoff_hessian = args.cutoff_hessian
    print("Model loaded.\n")

    # Warm up
    print("Warming up model...")
    warmup_batch = create_random_batch(args.n_atoms, device="cuda", seed=0)
    for _ in range(5):
        with torch.no_grad():
            model.forward(warmup_batch, otf_graph=True, hessian=False)
    torch.cuda.synchronize()
    print("Warmup complete.\n")

    # Create batch for profiling
    batch = create_random_batch(args.n_atoms, device="cuda", seed=args.seed)

    # Profile forward pass
    prof_forward = profile_forward_pass(model, batch, output_dir=args.output_dir)

    # Profile forward pass + VJPs
    batch_vjp = create_random_batch(args.n_atoms, device="cuda", seed=args.seed)
    prof_vjp = profile_forward_plus_vjps(
        model, batch_vjp, n_vjps=args.n_vjps, output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nTo view Chrome traces:")
    print("  1. Open Chrome browser")
    print("  2. Navigate to chrome://tracing")
    print("  3. Load the trace JSON files from the output directory")
