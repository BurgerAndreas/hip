"""
PyTorch Lightning module for training AlphaNet
"""

from typing import Dict, List, Optional, Tuple, Any, Mapping
from collections.abc import Iterable
from omegaconf import ListConfig, OmegaConf
import os
import time
from pathlib import Path
import wandb
import numpy as np

import torch
from torch import nn
from torch.utils.data import ConcatDataset

from torch_geometric.loader import DataLoader as TGDataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    # CosineAnnealingLR,
)
from hip.lrscheduler import StepLR, CosineAnnealingLR

try:
    from pytorch_lightning.utilities import grad_norm as pl_grad_norm
    from pytorch_lightning import LightningModule
except ImportError:
    from lightning.pytorch.utilities import grad_norm as pl_grad_norm
    from lightning import LightningModule
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from ocpmodels.hessian_graph_transform import (
    HessianGraphTransform,
)

from hip.ff_lmdb import LmdbDataset
from hip.utils import average_over_batch_metrics

# import hip.utils as diff_utils
import yaml
from hip.path_config import find_project_root, fix_dataset_path
from hip.loss_functions import (
    # compute_loss_blockdiagonal_hessian,
    get_hessian_eigen_loss_fn,
    get_eigval_eigvec_metrics,
    # BatchHessianLoss,
    # L1HessianLoss,
    # L2HessianLoss,
)

LR_SCHEDULER = {
    "cosine": CosineAnnealingLR,
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
    "plateau": ReduceLROnPlateau,
}
GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])


# from ocpmodels
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            name.endswith(".bias")
            or name.endswith(".affine_weight")
            or name.endswith(".affine_bias")
            or name.endswith(".mean_shift")
            or "bias." in name
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class SchemaUniformDataset:
    """Wrapper that ensures all datasets have the same attributes.

    RGD1 lacks:
    ae: <class 'torch.Tensor'> torch.Size([]) torch.float32 -> same as energy
    rxn: <class 'torch.Tensor'> torch.Size([]) torch.int64 -> add -1 to all

    All other (T1x based) datasets lack:
    freq: <class 'torch.Tensor'> torch.Size([N*3])
    eig_values: <class 'torch.Tensor'> torch.Size([N*3])
    force_constant: <class 'torch.Tensor'> torch.Size([N*3])
    -> remove these attributes from the dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Add missing attributes
        if not hasattr(data, "ae"):
            data.ae = torch.tensor(data.energy.item(), dtype=data.energy.dtype)
        if not hasattr(data, "rxn"):
            data.rxn = torch.tensor(-1, dtype=torch.int64)

        # Remove extra attributes
        if hasattr(data, "freq"):
            delattr(data, "freq")
        if hasattr(data, "eig_values"):
            delattr(data, "eig_values")
        if hasattr(data, "force_constant"):
            delattr(data, "force_constant")
        return data


class AlphaConfig:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)


class PotentialModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
    ) -> None:
        super().__init__()

        # Freeze all parameters except the specified heads
        self.heads_to_train = [
            "hessian_layers",
            "hessian_head",
            "hessian_edge_message_proj",
            "hessian_node_proj",
        ]

        training_config = self.fix_paths(training_config)
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config

        if self.model_config["name"] == "EquiformerV2":
            root_dir = find_project_root()
            config_path = os.path.join(root_dir, "configs/equiformer_v2.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(root_dir, "equiformer_v2.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            model_config = config["model"]
            model_config.update(self.model_config)
            self.potential = EquiformerV2_OC20(**model_config)
            self.pos_require_grad = False
        elif self.model_config["name"] == "AlphaNet":
            from alphanet.models.alphanet import AlphaNet

            self.potential = AlphaNet(AlphaConfig(model_config)).float()
            self.pos_require_grad = True
        elif (
            self.model_config["name"] == "LEFTNet"
            or self.model_config["name"] == "LEFTNet-df"
        ):
            from leftnet.potential import Potential
            from leftnet.model import LEFTNet

            self.pos_require_grad = True
            leftnet_config = dict(
                pos_require_grad=True,
                cutoff=10.0,
                num_layers=6,
                hidden_channels=196,
                num_radial=96,
                in_hidden_channels=8,
                reflect_equiv=True,
                legacy=True,
                update=True,
                pos_grad=False,
                single_layer_output=True,
            )
            node_nfs: List[int] = [9] * 1  # 3 (pos) + 5 (cat) + 1 (charge)
            edge_nf: int = 0  # edge type
            condition_nf: int = 1
            fragment_names: List[str] = ["structure"]
            pos_dim: int = 3
            edge_cutoff: Optional[float] = None
            self.potential = Potential(
                model_config=leftnet_config,
                node_nfs=node_nfs,  # 3 (pos) + 5 (cat) + 1 (charge),
                edge_nf=edge_nf,
                condition_nf=condition_nf,
                fragment_names=fragment_names,
                pos_dim=pos_dim,
                edge_cutoff=edge_cutoff,
                model=LEFTNet,
                enforce_same_encoding=None,
                source=None,
                timesteps=5000,
                condition_time=False,
            )
        else:
            print(
                "Please Check your model name (choose from 'EquiformerV2', 'AlphaNet', 'LEFTNet', 'LEFTNet-df')"
            )

        self.val_step_outputs = []

        self.wandb_run_id = None
        self.num_muon_params = None
        self.grad_norm_history = []

        # For Lightning
        # Allow non-strict checkpoint loading for transfer learning
        self.strict_loading = False

        # energy and force loss
        self.loss_fn = torch.nn.L1Loss()
        self.loss_fn_val = torch.nn.L1Loss()

        # Hessian loss
        self.do_hessian = self.training_config.get("hessian_loss_weight", 0.0) > 0.0
        if self.do_hessian:
            hessian_loss_type = self.training_config.get("hessian_loss_type", "mae")
            if hessian_loss_type == "mse":
                self.loss_fn_hessian = torch.nn.MSELoss()
            elif hessian_loss_type == "mae":
                self.loss_fn_hessian = torch.nn.L1Loss()
            else:
                raise ValueError(f"Invalid Hessian loss type: {hessian_loss_type}")

        self.do_eigen_loss = False
        if "eigen_loss" in self.training_config:
            self.loss_fn_eigen = get_hessian_eigen_loss_fn(
                **training_config["eigen_loss"]
            )
            _alpha = self.training_config["eigen_loss"]["alpha"]
            if isinstance(_alpha, Iterable) or (
                isinstance(_alpha, float) and _alpha > 0.0
            ):
                self.do_eigen_loss = True

        # Save arguments to hparams attribute for checkpointing
        self.save_hyperparameters(logger=False)

        self.use_hessian_graph_transform = True
        if (
            "otfgraph_in_model" in self.training_config
            and self.training_config["otfgraph_in_model"]
        ):
            # no need because we will compute graph during forward pass
            self.use_hessian_graph_transform = False

    def set_wandb_run_id(self, run_id: str) -> None:
        """Set the WandB run ID for checkpoint continuation."""
        self.wandb_run_id = run_id

    def get_wandb_run_id(self) -> Optional[str]:
        """Get the WandB run ID for checkpoint continuation."""
        return self.wandb_run_id

    def fix_paths(self, training_config):
        """
        Fix paths in the training config to be relative to the project root.
        """
        try:
            training_config["trn_path"] = fix_dataset_path(training_config["trn_path"])
            training_config["val_path"] = fix_dataset_path(training_config["val_path"])
        except Exception as e:
            pass
        return training_config

    def _freeze_except_heads(self, heads_to_train: List[str]) -> None:
        """
        Freeze all model parameters except the specified heads.

        Args:
            heads_to_train: List of head names to keep trainable
        """
        # First, freeze all parameters
        for param in self.potential.parameters():
            param.requires_grad = False

        # Then unfreeze only the specified heads
        for head_name in heads_to_train:
            if hasattr(self.potential, head_name):
                head_module = getattr(self.potential, head_name)
                if head_module is not None:
                    for param in head_module.parameters():
                        param.requires_grad = True
                    print(f"Unfroze parameters for {head_name}")
                else:
                    print(
                        f"Warning: {head_name} is None - head not created during model initialization"
                    )
            else:
                print(f"Warning: {head_name} not found in model")

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.potential.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.potential.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)"
        )

    def configure_optimizers(self):
        print("Configuring optimizer")
        # Only optimize parameters that require gradients (unfrozen heads)
        if self.training_config["train_hessian_only"]:
            print(
                f"Updating heads_to_train from \n {self.heads_to_train} to \n {self.potential.hessian_module_list}"
            )
            self.heads_to_train = self.potential.hessian_module_list
            self._freeze_except_heads(self.heads_to_train)
        trainable_params = [p for p in self.potential.parameters() if p.requires_grad]

        optimizer_config = self.optimizer_config
        optim_type = optimizer_config.pop("optimizer", "AdamW")
        optimizer_config.pop("beta1", None)
        optimizer_config.pop("beta2", None)

        # if weight_decay and filter_bias_and_bn:
        #     skip = {}
        #     if hasattr(model, 'no_weight_decay'):
        #         skip = model.no_weight_decay()
        #     parameters = add_weight_decay(model, weight_decay, skip)
        #     weight_decay = 0.
        # else:
        #     parameters = model.parameters()
        if optim_type.lower() == "adamw":
            optimizer_config = {
                k: v for k, v in self.optimizer_config.items() if "muon" not in k
            }
            optimizer = torch.optim.AdamW(trainable_params, **optimizer_config)
        elif optim_type.lower() == "muon":
            assert not self.training_config.get("train_hessian_only", False)
            # Muon is an optimizer for the hidden weights of a neural network.
            # Other parameters, such as embeddings, classifier heads, and hidden gains/biases
            # should be optimized using standard AdamW.
            from muon import MuonWithAuxAdam

            # hidden_weights = [p for p in self.potential.body.parameters() if p.ndim >= 2]
            # hidden_gains_biases = [p for p in self.potential.body.parameters() if p.ndim < 2]
            # nonhidden_params = [*self.potential.head.parameters(), *self.potential.embed.parameters()]
            # muon_params = hidden_weights
            # adam_params = hidden_gains_biases + nonhidden_params
            muon_params, adam_params = self.potential.get_muon_param_groups(
                **self.optimizer_config
            )
            # "params", "lr", "betas", "eps", "weight_decay", "use_muon"
            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.optimizer_config.get("lr_muon", 0.02),
                    weight_decay=self.optimizer_config.get("weight_decay_muon", 0.01),
                ),
                # Adam
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=self.optimizer_config.get("lr", 0.0005),
                    betas=self.optimizer_config.get("betas", (0.9, 0.999)),
                    eps=self.optimizer_config.get("eps", 1e-12),
                    weight_decay=self.optimizer_config.get("weight_decay", 0.01),
                    # **self.optimizer_config
                ),
            ]
            self.num_muon_params = np.sum([_p.numel() for _p in muon_params])
            self.num_adam_params = np.sum([_p.numel() for _p in adam_params])
            print(f"Number of muon parameters: {self.num_muon_params}")
            print(f"Number of adam parameters: {self.num_adam_params}")
            print(
                f"Percentage of muon parameters: {self.num_muon_params / (self.num_muon_params + self.num_adam_params) * 100:.2f}%"
            )
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {optim_type}")

        if self.training_config["lr_schedule_type"] is not None:
            if self.training_config["lr_schedule_type"] == "plateau":
                lr_schedule_config = self.training_config["lr_schedule_config"]
                lr_schedule_config = OmegaConf.to_container(lr_schedule_config)
                monitor = lr_schedule_config.pop("monitor")
                frequency = lr_schedule_config.pop("frequency", 1)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, **lr_schedule_config),
                        "monitor": monitor,
                        "frequency": frequency,
                    },
                }
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        print("Setting up dataset")
        if stage == "fit":
            # train dataset
            print(f"Loading training dataset from {self.training_config['trn_path']}")
            if (
                isinstance(self.training_config["trn_path"], list)
                or isinstance(self.training_config["trn_path"], tuple)
                or isinstance(self.training_config["trn_path"], ListConfig)
            ):
                datasets = []
                for path in self.training_config["trn_path"]:
                    transform = None
                    if self.use_hessian_graph_transform:
                        transform = HessianGraphTransform(
                            cutoff=self.potential.cutoff,
                            cutoff_hessian=self.potential.cutoff_hessian,
                            max_neighbors=self.potential.max_neighbors,
                            use_pbc=self.potential.use_pbc,
                        )
                    base_dataset = LmdbDataset(
                        Path(path),
                        transform=transform,
                        **self.training_config,
                    )
                    wrapped_dataset = SchemaUniformDataset(base_dataset)
                    datasets.append(wrapped_dataset)
                    print(
                        f"Loaded dataset from {path} with {len(wrapped_dataset)} samples (after split)"
                    )

                # Combine all datasets into a single concatenated dataset
                if ("data_weight" in self.training_config) and (
                    self.training_config["data_weight"] is not None
                ):
                    _datasets = []
                    for dataset, weight in zip(
                        datasets, self.training_config["data_weight"]
                    ):
                        for _ in range(weight):
                            _datasets.append(dataset)
                    datasets = _datasets
                self.train_dataset = ConcatDataset(datasets)
                print(
                    f"Combined {len(datasets)} datasets into one with {len(self.train_dataset)} total samples"
                )
            else:
                transform = None
                if self.use_hessian_graph_transform:
                    transform = HessianGraphTransform(
                        cutoff=self.potential.cutoff,
                        cutoff_hessian=self.potential.cutoff_hessian,
                        max_neighbors=self.potential.max_neighbors,
                        use_pbc=self.potential.use_pbc,
                    )
                self.train_dataset = SchemaUniformDataset(
                    LmdbDataset(
                        Path(self.training_config["trn_path"]),
                        transform=transform,
                        **self.training_config,
                    )
                )
            # val dataset
            transform = None
            if self.use_hessian_graph_transform:
                transform = HessianGraphTransform(
                    cutoff=self.potential.cutoff,
                    cutoff_hessian=self.potential.cutoff_hessian,
                    max_neighbors=self.potential.max_neighbors,
                    use_pbc=self.potential.use_pbc,
                )
            self.val_dataset = SchemaUniformDataset(
                LmdbDataset(
                    Path(self.training_config["val_path"]),
                    transform=transform,
                    **self.training_config,
                )
            )
            print("Number of training samples: ", len(self.train_dataset))
            print("Number of validation samples: ", len(self.val_dataset))
            num_train_batches = len(self.train_dataset) // self.training_config["bz"]
            num_val_batches = len(self.val_dataset) // self.training_config["bz_val"]
            print(f"Number of training batches: {num_train_batches}")
            print(f"Number of validation batches: {num_val_batches}")
            if self.training_config["drop_last"]:
                assert num_train_batches >= 1, (
                    f"Training set will be empty with drop_last {len(self.train_dataset)} / {self.training_config['bz']}"
                )
                assert num_val_batches >= 1, (
                    f"Validation set will be empty with drop_last {len(self.val_dataset)} / {self.training_config['bz_val']}"
                )
            print(f"self.do_eigen_loss: {self.do_eigen_loss}")

        else:
            raise NotImplementedError
        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.potential.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.potential.parameters())
        try:
            wandb.log(
                {
                    "train-trainable_params": trainable_params,
                    "train-total_params": total_params,
                    "num_train_samples": len(self.train_dataset),
                    "num_val_samples": len(self.val_dataset),
                    "num_train_batches": num_train_batches,
                    "num_val_batches": num_val_batches,
                    "num_muon_params": self.num_muon_params,
                }
            )
        except Exception as e:
            print(f"Error logging trainable parameters: {e}")
        return

    def train_dataloader(self):
        """Override to use custom collate function for Hessian batch offsetting"""
        return TGDataLoader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
            follow_batch=self.training_config["follow_batch"],
            drop_last=self.training_config["drop_last"],
        )

    def val_dataloader(self):
        """Override to use custom collate function for Hessian batch offsetting"""
        return TGDataLoader(
            self.val_dataset,
            batch_size=self.training_config["bz_val"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            follow_batch=self.training_config["follow_batch"],
            drop_last=self.training_config["drop_last"],
        )

    # not used
    def test_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.test_dataset,
            batch_size=self.training_config["bz_val"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            follow_batch=self.training_config["follow_batch"],
            drop_last=self.training_config["drop_last"],
        )

    @torch.enable_grad()
    def compute_loss(self, batch, return_efh=False):
        loss = 0.0
        info = {}
        # batch.pos.requires_grad_()

        hat_ae, hat_forces, outputs = self.potential.forward(
            batch.to(self.device),
            hessian=self.do_hessian,
            otf_graph=self.training_config["otfgraph_in_model"],
        )

        if self.do_hessian:
            hessian_pred = outputs["hessian"].to(self.device)
            hessian_true = batch.hessian.to(self.device)

            hessian_loss = self.loss_fn_hessian(hessian_pred, hessian_true)
            loss += hessian_loss * self.training_config["hessian_loss_weight"]
            info["Loss Hessian"] = hessian_loss.detach().item()

            if self.do_eigen_loss:
                eigen_loss = self.loss_fn_eigen(
                    pred=hessian_pred,
                    target=hessian_true,
                    data=batch,
                )
                loss += eigen_loss
                info["Loss Eigen"] = eigen_loss.detach().item()

        if not self.training_config["train_hessian_only"]:
            # energy
            hat_ae = hat_ae.squeeze().to(self.device)
            ae = batch.ae.to(self.device)
            # ae = batch.energy.to(self.device)
            eloss = self.loss_fn(ae, hat_ae)
            loss += eloss * self.training_config["energy_loss_weight"]
            info["Loss E"] = eloss.detach().item()

            # forces
            hat_forces = hat_forces.to(self.device)
            forces = batch.forces.to(self.device)
            floss = self.loss_fn(forces, hat_forces)
            loss += floss * self.training_config["force_loss_weight"]
            info["Loss F"] = floss.detach().item()

        if return_efh:
            return loss, info, (hat_ae, hat_forces, outputs)
        # loss = floss * 100 + eloss * 4 + hessian_loss * 4
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)

        # self.log("train-totloss", loss, rank_zero_only=True)
        loss_dict = {"train-totloss": loss}

        for k, v in info.items():
            # self.log(f"train-{k}", v, rank_zero_only=True)
            loss_dict[f"train-{k}"] = v
        self.log_dict(loss_dict, rank_zero_only=True)
        del info
        return loss

    def compute_eval_loss(self, batch, prefix, efh):
        """Compute comprehensive evaluation metrics for eigenvalues and eigenvectors."""
        hat_ae, hat_forces, outputs = efh
        eval_metrics = {}

        eval_metrics["MAE E"] = (
            self.loss_fn_val(hat_ae, batch.ae).abs().mean().detach().item()
        )
        eval_metrics["MAE F"] = (
            self.loss_fn_val(hat_forces, batch.forces).abs().mean().detach().item()
        )

        if self.do_hessian:
            hessian_true = batch.hessian
            hessian_pred = outputs["hessian"].detach()

            # MSE Hessian
            eval_metrics["MSE Hessian"] = (
                (hessian_pred.squeeze() - hessian_true.squeeze()).pow(2).mean().item()
            )
            eval_metrics["MAE Hessian"] = (
                (hessian_pred.squeeze() - hessian_true.squeeze()).abs().mean().item()
            )

            # Eigenvalue, Eigenvector metrics
            eig_metrics = get_eigval_eigvec_metrics(
                hessian_true.to("cpu"),
                hessian_pred.to("cpu"),
                batch.to("cpu"),
                prefix=f"{prefix}-step{self.global_step}-epoch{self.current_epoch}",
            )
            eval_metrics.update(eig_metrics)

        return eval_metrics

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        # compute training loss on eval set
        with torch.no_grad():
            loss, info, efh = self.compute_loss(batch, return_efh=True)
        detached_loss = loss.detach()
        info["trainloss"] = detached_loss.item()

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v

        # compute eval metrics on eval set
        eval_info = self.compute_eval_loss(batch, prefix=prefix, efh=efh)
        for k, v in eval_info.items():
            info_prefix[f"{prefix}-{k}"] = v

        info_prefix[f"{prefix}-totloss"] = eval_info["MAE E"] + eval_info["MAE F"]

        if self.do_hessian:
            info_prefix[f"{prefix}-totloss"] += eval_info["MAE Hessian"]
            # info_prefix[f"{prefix}-totloss"] += eval_info["MAE Eigvals Eckart"]
            # info_prefix[f"{prefix}-totloss"] += (-1 * eval_info["Abs Cosine Sim v1 Eckart"] / 20)

        # del info
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        val_epoch_metrics = average_over_batch_metrics(self.val_step_outputs)
        # if self.trainer.is_global_zero:
        #     # tqdm.write(f"Val epoch {self.current_epoch} completed")
        #     pretty_print(
        #         self.current_epoch,
        #         {"val-totloss": val_epoch_metrics["val-totloss"]},
        #         prefix="\nVal"
        #     )

        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True, prog_bar=False)
        if hasattr(self, "val_start_time"):
            self.log(
                "val-val_duration_seconds",
                time.time() - self.val_start_time,
                prog_bar=False,
                rank_zero_only=True,
                sync_dist=True,
            )

        self.val_step_outputs.clear()

    def on_train_epoch_start(self):
        """Record the start time of the training epoch."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        """Calculate and log the time taken for the training epoch."""
        if hasattr(self, "epoch_start_time"):
            epoch_duration = time.time() - self.epoch_start_time
            self.log(
                "train-epoch_duration_seconds",
                epoch_duration,
                rank_zero_only=True,
                sync_dist=True,
            )
            # if self.trainer.is_global_zero:
            #     print(f"Epoch {self.current_epoch} completed in {epoch_duration:.2f} seconds")

    def on_validation_epoch_start(self):
        """Reset the validation dataloader at the start of every epoch."""
        self.val_start_time = time.time()
        super().on_validation_epoch_start()

    def on_train_start(self):
        """Log the starting epoch to Weights & Biases."""
        # Route through Lightning logger so it reaches WandB and respects rank_zero_only
        self.log(
            "start_epoch",
            int(self.current_epoch),
            rank_zero_only=True,
            prog_bar=False,
            sync_dist=True,
        )
        super().on_train_start()

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms: The dictionary of p-norms of each parameter's gradient and
        # a special entry for the total p-norm of the gradients viewed as a single vector
        norms = pl_grad_norm(module=self.potential, norm_type=2)
        self.grad_norm_history.append(norms["grad_2.0_norm_total"])  # float
        if (self.global_step % 100 == 0) and self.global_step > 350:
            norms["grad_2.0_norm_std"] = np.std(self.grad_norm_history)
        self.log_dict(norms)
        # super().on_before_optimizer_step(optimizer)
