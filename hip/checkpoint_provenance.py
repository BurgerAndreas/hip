"""Record compact, reproducible training provenance in Lightning checkpoints."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import threading
from urllib.parse import urlsplit, urlunsplit

from lightning.pytorch import Callback
import torch

from hip.path_config import fix_dataset_path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_value(root, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def safe_remote(value):
    if not value or "://" not in value:
        return value
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def dataset_manifest(path):
    """Hash each dataset file once per run, without storing its contents."""
    if isinstance(path, (list, tuple)):
        return [dataset_manifest(item) for item in path]
    try:
        resolved = Path(fix_dataset_path(str(path))).resolve()
        files = sorted(p for p in resolved.rglob("*") if p.is_file()) if resolved.is_dir() else [resolved]
        entries = []
        for file in files:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries.append({"path": str(file), "size_bytes": file.stat().st_size, "sha256": digest.hexdigest()})
        return {"requested_path": str(path), "resolved_path": str(resolved), "files": entries}
    except (OSError, ValueError, RuntimeError) as exc:
        return {"requested_path": str(path), "error": str(exc)}


class CheckpointProvenance(Callback):
    def __init__(self, config, root):
        super().__init__()
        self.config = config
        self.root = Path(root)
        self.started_at = None
        self.datasets = {}
        self.hash_thread = None
        self.git = None

    def on_fit_start(self, trainer, pl_module):
        self.started_at = utc_now()
        if not trainer.is_global_zero:
            return
        status = git_value(self.root, "status", "--porcelain", "--untracked-files=normal")
        self.git = {"commit": git_value(self.root, "rev-parse", "HEAD"),
                    "branch": git_value(self.root, "symbolic-ref", "--quiet", "--short", "HEAD"),
                    "dirty": None if status is None else bool(status),
                    "remote": safe_remote(git_value(self.root, "remote", "get-url", "origin"))}
        paths = {key: self.config.get("training", {}).get(key) for key in ("trn_path", "val_path")}

        def hash_datasets():
            for key, path in paths.items():
                if path:
                    self.datasets[key] = dataset_manifest(path)

        self.hash_thread = threading.Thread(target=hash_datasets, daemon=True)
        self.hash_thread.start()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not trainer.is_global_zero:
            return
        if self.hash_thread is not None:
            self.hash_thread.join()
        packages = {}
        for name in ("torch", "lightning", "pytorch-lightning", "torch-geometric", "torch-cluster", "numpy", "omegaconf", "hydra-core"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        model = pl_module.potential
        graph = {key: getattr(model, key, None) for key in
                 ("max_neighbors", "cutoff", "cutoff_hessian", "fully_connected_hessian", "otf_graph", "use_pbc")}
        graph["backbone_builder"] = "radius_graph_pbc" if graph["use_pbc"] else "torch_geometric.nn.radius_graph"
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        batch_size = self.config.get("training", {}).get("bz")
        accumulation = trainer.accumulate_grad_batches
        effective_batch = (batch_size * trainer.world_size * accumulation
                           if isinstance(batch_size, int) and isinstance(accumulation, int) else None)
        checkpoint["provenance"] = {
            "schema_version": 1, "saved_at_utc": utc_now(), "fit_started_at_utc": self.started_at,
            "git": self.git,
            "resolved_config": self.config, "datasets": self.datasets, "graph": graph,
            "software": {"python": platform.python_version(), "packages": packages,
                         "cuda": torch.version.cuda},
            "run": {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "wandb_run_id": getattr(pl_module, "wandb_run_id", None),
                    "model_checkpoint": self.config.get("ckpt_model_path"),
                    "trainer_checkpoint": self.config.get("ckpt_trainer_path"),
                    "seed": os.environ.get("PL_GLOBAL_SEED"),
                    "torch_initial_seed": torch.initial_seed(),
                    "world_size": trainer.world_size,
                    "accumulate_grad_batches": trainer.accumulate_grad_batches,
                    "batch_size_per_device": batch_size, "effective_batch_size": effective_batch,
                    "gpu_names_visible_to_rank_zero": devices},
        }
