"""Pydantic model for default configuration and validation."""

import subprocess
import json
import torch
from typing import Literal, Dict, Optional
from pydantic_settings import BaseSettings

try:
    VERSION = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    VERSION = "NA"
    pass


class TrainConfig(BaseSettings):
    """Hyperparameter schema for training."""

    epochs: int = 1000
    batch_size: int = 32
    vt_batch_size: int = 32
    lr: float = 0.0005
    lr_decay_factor: float = 0.5
    lr_decay_step_size: int = 150
    weight_decay: float = 0
    save_dir: str = ""
    log_dir: str = ""
    num_workers: int = 8
    accumulation_steps: int = 1
    disable_tqdm: bool = False
    scheduler: str = "steplr"
    norm_label: bool = False
    device: str = "cuda"
    force: bool = False
    stress: bool = False
    energy_loss: str = "mae"
    force_loss: str = "mae"
    stress_loss: str = "mae"
    energy_metric: str = "mae"
    force_metric: str = "mae"
    stress_metric: str = "mae"
    energy_coef: float = 1.0
    force_coef: float = 0.0
    stress_coef: float = 0.0
    eval_steps: int = 1


class DataConfig(BaseSettings):
    """Hyperparameter schema for dataset."""

    root: str = "dataset/"
    dataset_name: str = "qm9"
    target: str = "U0"
    train_size: Optional[int] = None
    valid_size: Optional[int] = None
    test_size: Optional[int] = None
    train_dataset: Optional[str] = None
    valid_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    seed: int = 42


class AlphaConfig(BaseSettings):
    """Hyperparameter schema for AlphaNet."""

    name: Literal["Alphanet"] = "Alphanet"
    num_layers: int = 3
    num_targets: int = 1
    output_dim: int = 1
    readout: str = "sum"
    norm: bool = False
    use_pbc: bool = True
    compute_forces: bool = False
    compute_stress: bool = False
    eps: float = 1e-10
    hidden_channels: int = 128
    cutoff: float = 5.0
    pos_require_grad: bool = False
    num_radial: int = 96
    use_sigmoid: bool = False
    head: int = 16
    a: float = 1
    b: float = 0
    main_chi1: int = 24
    mp_chi1: int = 24
    chi2: int = 6
    hidden_channels_chi: int = 96
    has_dropout_flag: bool = True
    has_norm_before_flag: bool = True
    has_norm_after_flag: bool = False
    reduce_mode: str = "sum"
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )


class All_Config:
    def __init__(self, data=None, model=None, train=None, loss_func=None):
        self.data = DataConfig(**data) if data else DataConfig()
        self.model = AlphaConfig(**model) if model else AlphaConfig()
        self.train = TrainConfig(**train) if train else TrainConfig()

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
