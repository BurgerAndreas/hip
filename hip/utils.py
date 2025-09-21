"""
Utility functions for training and evaluation.
"""

from typing import Dict, List, Union, Iterable
import numpy as np
import torch


def average_over_batch_metrics(batch_metrics: List[Dict], allowed: List = []):
    epoch_metrics = {}
    effective_batch = {}
    for ii, out in enumerate(batch_metrics):
        for k, v in out.items():
            if not (k in allowed or len(allowed) == 0):
                continue
            if ii == 0:
                epoch_metrics[k] = v
                effective_batch[k] = 1
            else:
                if not np.isnan(v):
                    epoch_metrics[k] += v
                    effective_batch[k] += 1
    for k in epoch_metrics:
        epoch_metrics[k] /= effective_batch[k]
    return epoch_metrics


def pretty_print(epoch, metric_dict, prefix="Train"):
    out = f"{prefix} epoch {epoch} "
    for k, v in metric_dict.items():
        out += f"{k} {v:.2f} "
    print(out)


class Queue:
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def get_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]], norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )

    return total_norm
