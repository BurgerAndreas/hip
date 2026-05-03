import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as TGDataLoader

from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from hip.training_module import PotentialModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile HIP Equiformer forward calls with torch.profiler."
    )
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/hip_v2.ckpt",
        help="Path to a Lightning checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset path or known dataset file name.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Number of samples per profiled batch.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="First dataset index to profile.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Forward calls to run before profiling.",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=10,
        help="Forward calls to capture in the trace.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="profile_forward",
        help="Directory for profiler trace and summary table.",
    )
    parser.add_argument(
        "--trace_name",
        type=str,
        default=None,
        help="Base name for profiler outputs. Defaults to checkpoint/dataset names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--otf_graph",
        action="store_true",
        help="Force Hessian graph construction inside every forward call.",
    )
    parser.add_argument(
        "--no_hessian",
        action="store_true",
        help="Profile energy/force forward without direct Hessian prediction.",
    )
    parser.add_argument(
        "--hessian_fully_connected",
        action="store_true",
        help="Build the Hessian graph as fully connected within each sample.",
    )
    parser.add_argument(
        "--reuse_batch",
        action="store_true",
        help="Profile the same device batch repeatedly instead of distinct batches.",
    )
    parser.add_argument(
        "--row_limit",
        type=int,
        default=40,
        help="Number of profiler table rows to print and save.",
    )
    parser.add_argument(
        "--with_stack",
        action="store_true",
        help="Capture Python stack traces. Useful but can make traces much larger.",
    )
    parser.add_argument(
        "--with_flops",
        action="store_true",
        help="Ask torch.profiler to estimate FLOPs where supported.",
    )
    parser.add_argument(
        "--no_record_shapes",
        action="store_true",
        help="Disable input shape recording.",
    )
    parser.add_argument(
        "--no_profile_memory",
        action="store_true",
        help="Disable profiler memory tracking.",
    )
    return parser.parse_args()


def load_device_batches(args, device):
    dataset = LmdbDataset(fix_dataset_path(args.dataset))
    num_batches = 1 if args.reuse_batch else args.warmup_steps + args.profile_steps
    num_items = max(1, num_batches * args.batch_size)
    indices = [(args.sample_index + i) % len(dataset) for i in range(num_items)]

    loader = TGDataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
    )
    batches = [batch.to(device) for batch in loader]
    if not batches:
        raise RuntimeError(f"No batches loaded from dataset {args.dataset}")
    return batches


def forward_once(model, batch, args):
    with record_function("hip_equiformer_forward"):
        energy, forces, outputs = model.forward(
            batch,
            hessian=not args.no_hessian,
            otf_graph=args.otf_graph,
        )

    # Touch the outputs so the profiled call mirrors inference use.
    if not args.no_hessian and "hessian" not in outputs:
        raise RuntimeError("Model forward did not return outputs['hessian']")
    return energy, forces, outputs


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    ckpt_path = Path(args.ckpt_path)
    dataset_name = Path(args.dataset).stem
    trace_name = args.trace_name or f"{ckpt_path.stem}_{dataset_name}_forward"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PotentialModule.load_from_checkpoint(
        args.ckpt_path,
        map_location=device,
        strict=False,
        weights_only=False,
    ).potential.to(device)
    if args.hessian_fully_connected:
        model.fully_connected_hessian = True
    model.eval()

    batches = load_device_batches(args, device)
    print(
        f"Loaded {len(batches)} device batch(es); "
        f"batch_size={args.batch_size}, device={device}"
    )

    for step in range(args.warmup_steps):
        batch = batches[0] if args.reuse_batch else batches[step % len(batches)]
        forward_once(model, batch, args)
    synchronize_if_cuda(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)

    with profile(
        activities=activities,
        record_shapes=not args.no_record_shapes,
        profile_memory=not args.no_profile_memory,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
    ) as prof:
        for step in range(args.profile_steps):
            batch_index = 0 if args.reuse_batch else (args.warmup_steps + step)
            batch = batches[batch_index % len(batches)]
            with record_function(f"profile_step_{step}"):
                forward_once(model, batch, args)
            prof.step()

    synchronize_if_cuda(device)

    trace_path = output_dir / f"{trace_name}.json"
    table_path = output_dir / f"{trace_name}.txt"
    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit)

    prof.export_chrome_trace(str(trace_path))
    table_path.write_text(table)

    print(table)
    print(f"\nWrote Chrome trace: {trace_path}")
    print(f"Wrote key averages: {table_path}")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"Peak CUDA memory allocated during profiled steps: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
