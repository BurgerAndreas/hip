import torch
from typing import Optional


# https://github.com/deepprinciple/HORM/blob/eval/eval.py
def _get_derivatives_not_none(
    x: torch.Tensor,
    y: torch.Tensor,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> torch.Tensor:
    """Helper function to compute derivatives"""
    ret = torch.autograd.grad(
        [y.sum()],
        [x],
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )[0]
    return ret


def compute_hessian(
    coords, energy, forces=None, create_graph=False, allow_unused=False
):
    """Compute Hessian matrix using autograd."""
    # compute force if not given (first-order derivative)
    if forces is None:
        forces = -_get_derivatives_not_none(coords, energy, create_graph=True)
    # get number of element (n_atoms * 3)
    _forc = forces.reshape(-1)
    n_comp = _forc.shape[0]
    # Initialize hessian
    hess = []
    for f in _forc:
        # compute second-order derivative for each element
        hess_row = _get_derivatives_not_none(
            coords,
            -f,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )
        hess.append(hess_row)
    # stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)
