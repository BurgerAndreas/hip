import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Literal,
    Optional,
    SupportsFloat,
    TypedDict,
    Union,
)
from weakref import ref

from torch import inf, Tensor
import os
import torch
import matplotlib.pyplot as plt

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _warn_get_lr_called_within_step


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.
    The only difference from the PyTorch StepLR is that it supports a minimum learning rate.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler.last_epoch specifies the index of the previous epoch.
    When last_epoch=-1, sets initial lr as lr.

    Steps and epochs are used synonymously here and mean training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of the previous epoch. Default: -1.
        min_lr (float): Minimum learning rate. Default: None.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        min_lr: Optional[float] = None,
        warmup_epochs: int = 0,
        warmup_init_lr: float = 0.0,
        last_epoch: int = -1,
    ):  # noqa: D107
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        self.warmup_epochs = warmup_epochs
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        # Warmup phase (linear ramp from warmup_init_lr to base_lr)
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            start_lr = (
                0.0 if self.warmup_init_lr is None else float(self.warmup_init_lr)
            )
            progress = float(self.last_epoch + 1) / float(self.warmup_epochs)
            lrs = [
                start_lr + (base_lr - start_lr) * progress for base_lr in self.base_lrs
            ]
        else:
            # Standard step schedule
            if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
                lrs = [group["lr"] for group in self.optimizer.param_groups]
            else:
                lrs = [
                    group["lr"] * self.gamma for group in self.optimizer.param_groups
                ]
        if self.min_lr is not None:
            lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs

    def _get_closed_form_lr(self):
        # Closed form with warmup support
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            start_lr = self.warmup_init_lr
            progress = float(self.last_epoch + 1) / float(self.warmup_epochs)
            lrs = [
                start_lr + (base_lr - start_lr) * progress for base_lr in self.base_lrs
            ]
        else:
            lrs = [
                base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs
            ]
        if self.min_lr is not None:
            lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs


class CosineAnnealingLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing schedule.

    The :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        min_lr: Optional[float] = None,  # alias for eta_min
        warmup_epochs: int = 0,
        warmup_init_lr: float = 0.0,
        hold_min_after_tmax: bool = False,
        last_epoch: int = -1,
        **kwargs,
    ):  # noqa: D107
        self.T_max = T_max
        if min_lr is not None:
            eta_min = min_lr
        self.eta_min = eta_min
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        self.warmup_epochs = warmup_epochs
        self.warmup_init_lr = warmup_init_lr
        self.hold_min_after_tmax = bool(hold_min_after_tmax)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Retrieve the learning rate of each parameter group.
        η(t) = η_min + (base_lr - η_min) · (1 + cos(π·t/T_max)) / 2
        """
        # Upstream PyTorch's get_lr has recursive cases to accommodate LR being modified elsewhere.
        # This uses the closed form from base_lrs for simplicity and determinism.
        # This is correct as long as nothing else mutates the optimizer LR outside the scheduler.
        _warn_get_lr_called_within_step(self)

        # Warmup phase (linear ramp from warmup_init_lr to base_lr)
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            start_lr = float(self.warmup_init_lr)
            progress = float(self.last_epoch + 1) / float(self.warmup_epochs)
            return [
                start_lr + (base_lr - start_lr) * progress for base_lr in self.base_lrs
            ]

        # Cosine phase after warmup
        t = max(0, self.last_epoch - self.warmup_epochs)
        if self.hold_min_after_tmax and t >= self.T_max and self.eta_min > 0.0:
            return [self.eta_min for _ in self.base_lrs]
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
        # before:
        # if self.last_epoch == 0:
        #     return [group["lr"] for group in self.optimizer.param_groups]
        # elif self._step_count == 1 and self.last_epoch > 0:
        #     return [
        #         self.eta_min
        #         + (base_lr - self.eta_min)
        #         * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
        #         / 2
        #         for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        #     ]
        # elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
        #     return [
        #         group["lr"]
        #         + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
        #         for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        #     ]
        # return [
        #     (1 + math.cos(math.pi * self.last_epoch / self.T_max))
        #     / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
        #     * (group["lr"] - self.eta_min)
        #     + self.eta_min
        #     for group in self.optimizer.param_groups
        # ]

    def _get_closed_form_lr(self):
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            start_lr = float(self.warmup_init_lr)
            progress = float(self.last_epoch + 1) / float(self.warmup_epochs)
            return [
                start_lr + (base_lr - start_lr) * progress for base_lr in self.base_lrs
            ]
        t = max(0, self.last_epoch - self.warmup_epochs)
        if self.hold_min_after_tmax and t >= self.T_max and self.eta_min > 0.0:
            return [self.eta_min for _ in self.base_lrs]
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":
    # plot learning rate schedule
    # Create a tiny model and two independent optimizers
    model1 = torch.nn.Linear(1, 1)
    model2 = torch.nn.Linear(1, 1)
    model3 = torch.nn.Linear(1, 1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.05)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.05)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.05)
    # When resuming (last_epoch >= 0), PyTorch expects 'initial_lr' in param groups
    for opt in [optimizer1, optimizer2, optimizer3]:
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])  # base LR before any scheduling

    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.9)
    scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.9, last_epoch=10)
    scheduler3 = StepLR(
        optimizer3,
        step_size=2,
        gamma=0.9,
        min_lr=0.02,
        warmup_epochs=5,
        warmup_init_lr=0.0,
    )

    num_epochs = 40
    lrs1 = []
    lrs2 = []
    lrs3 = []
    epochs = list(range(num_epochs))
    for _ in epochs:
        lrs1.append(optimizer1.param_groups[0]["lr"])
        lrs2.append(optimizer2.param_groups[0]["lr"])
        lrs3.append(optimizer3.param_groups[0]["lr"])
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

    plt.plot(epochs, lrs1, label="scheduler1 (last_epoch=-1)")
    plt.plot(epochs, lrs2, label="scheduler2 (last_epoch=10)")
    plt.plot(epochs, lrs3, label=f"scheduler3 (last_epoch=-1, lrmin=0.02)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("StepLR schedule comparison")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    fname = "playground/plots/step_lr_schedule_comparison.png"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname)
    print(f"Saved plot to {fname}")
