"""
PyTorch Lightning module for training AlphaNet
"""

from typing import Dict, List, Optional, Tuple

from pathlib import Path
import torch
from torch import nn

from torch_geometric.loader import DataLoader as TGDataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    StepLR,
    CosineAnnealingLR,
)

try:
    from pytorch_lightning import LightningModule
except ImportError:
    from lightning import LightningModule
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    CosineSimilarity,
)

from alphanet.ff_lmdb import LmdbDataset
from alphanet.utils import average_over_batch_metrics, pretty_print
import alphanet.utils as diff_utils
from alphanet.models.alphanet import AlphaNet

LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}


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
        self.potential = AlphaNet(AlphaConfig(model_config)).float()

        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.pos_require_grad = model_config["pos_require_grad"]

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = diff_utils.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

        self.loss_fn = nn.L1Loss()
        self.MAEEval = MeanAbsoluteError()
        self.MAPEEval = MeanAbsolutePercentageError()
        self.cosineEval = CosineSimilarity(reduction="mean")
        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.potential.parameters(), **self.optimizer_config
        )
        # scheduler = CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-6)
        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        return optimizer

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_train.lmdb"),
                **self.training_config,
            )
            self.val_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_valid.lmdb"),
                **self.training_config,
            )
            print("# of training data: ", len(self.train_dataset))
            print("# of validation data: ", len(self.val_dataset))
        elif stage == "test":
            self.test_dataset = LmdbDataset(
                Path(self.training_config["datadir"], f"ff_test.lmdb"),
                **self.training_config,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.train_dataset,
            batch_size=self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
        )

    def val_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.val_dataset,
            batch_size=self.training_config["bz"] * 2,
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    def test_dataloader(self) -> TGDataLoader:
        return TGDataLoader(
            self.test_dataset,
            batch_size=self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
        )

    @torch.enable_grad()
    def compute_loss(self, batch):
        batch.pos.requires_grad_()
        hat_ae, hat_forces = self.potential.forward(
            batch.to(self.device),
        )
        hat_ae = hat_ae.squeeze().to(self.device)
        hat_forces = hat_forces.to(self.device)
        ae = batch.ae.to(self.device)
        forces = batch.forces.to(self.device)

        eloss = self.loss_fn(ae, hat_ae)
        floss = self.loss_fn(forces, hat_forces)
        info = {
            "MAE_E": self.MAEEval(hat_ae, ae).item(),
            "MAE_F": self.MAEEval(hat_forces, forces).item(),
            "MAPE_E": self.MAPEEval(hat_ae, ae).item(),
            "MAPE_F": self.MAPEEval(hat_forces, forces).item(),
            "MAE_Fcos": 1
            - self.cosineEval(hat_forces.detach().cpu(), forces.detach().cpu()),
            "Loss_E": eloss.item(),
            "Loss_F": floss.item(),
        }
        self.MAEEval.reset()
        self.MAPEEval.reset()
        self.cosineEval.reset()

        loss = floss * 400 + eloss * 4
        return loss, info

    def training_step(self, batch, batch_idx):
        loss, info = self.compute_loss(batch)

        self.log("train-totloss", loss, rank_zero_only=True)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)
        del info
        return loss

    def __shared_eval(self, batch, batch_idx, prefix, *args):
        with torch.enable_grad():
            loss, info = self.compute_loss(batch)
            info["totloss"] = loss.item()

            info_prefix = {}
            for k, v in info.items():
                key = f"{prefix}-{k}"
                if isinstance(v, torch.Tensor):
                    v = v.detach()
                    if v.is_cuda:
                        v = v.cpu()
                    if v.numel() == 1:
                        info_prefix[key] = v.item()
                    else:
                        info_prefix[key] = v.numpy()
                else:
                    info_prefix[key] = v
                self.log(
                    key, v, on_step=True, on_epoch=True, prog_bar=True, logger=True
                )
            # self.log(f"{prefix}_totloss", info_prefix[f"{prefix}-totloss"],
            #        on_step=True, on_epoch=True, prog_bar=True, logger=True)
            del info
        return info_prefix

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        loss, info = self.compute_loss(batch)
        detached_loss = loss.detach()
        info["totloss"] = detached_loss.item()
        # info["totloss"] = loss.item()

        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        del info

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # 收集每个 batch 的输出
        self.val_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        val_epoch_metrics = average_over_batch_metrics(self.val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")

        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

        self.val_step_outputs.clear()

    # def _validation_epoch_end(self, val_step_outputs):
    #     val_epoch_metrics = average_over_batch_metrics(val_step_outputs)
    #     if self.trainer.is_global_zero:
    #         pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
    #     val_epoch_metrics.update({"epoch": self.current_epoch})
    #     for k, v in val_epoch_metrics.items():
    #         self.log(k, v, sync_dist=True)

    def _configure_gradient_clipping(
        self,
        optimizer,
        # optimizer_idx,
        gradient_clip_val,
        gradient_clip_algorithm,
    ):
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        max_grad_norm = 2 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = diff_utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
