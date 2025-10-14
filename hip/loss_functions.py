import torch
from collections.abc import Iterable

from hip.frequency_analysis import eckart_projection_notmw_torch

# from hip.masses import MASS_DICT
from nets.prediction_utils import Z_TO_ATOM_SYMBOL


def tensor_info(t):
    if not isinstance(t, torch.Tensor):
        return f"{type(t)}"
    return f"{list(t.shape)} {int(t.numel())} {t.dtype}"


##############################################################################
# predicting the full Hessian matrix


def get_hessian_loss_fn(loss_name: str, **kwargs):
    if loss_name == "eigenspectrum":
        return BatchHessianLoss(eigenspectrum_loss, **kwargs)
    elif loss_name.lower() in ["mse", "l2"]:
        return torch.nn.MSELoss()  # F.mse_loss
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


##############################################################################
# vector losses


def cosine_similarity(a, b):
    """Cosine similarity: cos(a, b) = <a, b> / (|a|*|b|) =  < a/|a|, b/|b| >"""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def abs_cosine_similarity(a, b):
    """Sign-invariant absolute cosine similarity: |cos(a, b)|"""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.abs(torch.dot(a, b) / (torch.norm(a) * torch.norm(b)))


def dot_similarity(a, b):
    """Dot product similarity: <a, b>"""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.dot(a, b)


def abs_dot_similarity(a, b):
    """Sign-invariant absolute dot product similarity: |<a, b>|"""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.abs(torch.dot(a, b))


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sign-invariant cosine similarity loss: 1 - |cos(pred, target)|"""
    return 1.0 - abs_cosine_similarity(pred, target)


def cosine_squared_loss(
    v_pred: torch.Tensor,
    v_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine-squared loss, sign-invariant: 1 - |cos(pred, target)|^2"""
    return 1.0 - (abs_cosine_similarity(v_pred, v_true) ** 2)


##############################################################################
# Eigenspectrum losses that don't require to backprop through .eigh


def batch_hessian_loss(
    hessian_pred,
    hessian_true,
    data,
    lossfn,
    debugstr="",
    **lossfn_kwargs,
):
    """We can't normalize concatenated vectors, so we process each vector separately.
    Returns a scalar of similarity averaged over batches.
    lossfn should return a scalar, otherwise it will be averaged over all entries returned.
    data should be a torch_geometric.data.Batch object.
    """
    # N=15, B=64 ~ 130_000 entries
    natoms = data.natoms
    B = data.batch.max() + 1
    # ptr = data.ptr
    # check
    numels = data.natoms.pow(2).mul(9)
    ptr_hessian = torch.cat([torch.tensor([0], device=numels.device), numels], dim=0)
    ptr_hessian = torch.cumsum(ptr_hessian, dim=0)
    hessian_pred = hessian_pred.view(-1)
    hessian_true = hessian_true.view(-1)
    losses = []
    for _b in range(B):
        _start = ptr_hessian[_b].item()
        ND = natoms[_b] * 3
        _numel = ND**2
        _end = _numel + _start
        hessian_pred_b = hessian_pred[_start:_end]
        hessian_true_b = hessian_true[_start:_end]
        loss_b = lossfn(
            hessian_pred=hessian_pred_b,
            hessian_true=hessian_true_b,
            N=natoms[_b].item(),
            **lossfn_kwargs,
        )
        losses.append(loss_b)
    return torch.stack(losses).mean()


class BatchHessianLoss(torch.nn.Module):
    """Wrapper to batch a loss function over hessian matrices."""

    def __init__(self, loss_fn, **kwargs):
        super(BatchHessianLoss, self).__init__()
        self.loss_fn = loss_fn
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

    def forward(self, pred, target, data, **kwargs):
        _kwargs = self.kwargs.copy()
        if kwargs is not None:
            _kwargs.update(kwargs)
        return batch_hessian_loss(pred, target, data, self.loss_fn, **_kwargs)


def _reduce_diff(diff, dist):
    dist = str(dist).lower()
    if dist == "mse":
        return diff.pow(2).mean()
    elif dist == "mae":
        return diff.abs().mean()
    elif dist == "frosq":
        return torch.linalg.norm(diff, ord="fro") ** 2
    elif dist == "fro":
        return torch.linalg.norm(diff, ord="fro")
    elif dist == "1":
        return torch.linalg.norm(diff, ord=1)
    elif dist == "2":
        return torch.linalg.norm(diff, ord=2)
    else:
        raise ValueError(f"Invalid distance: {dist}")


def eigenspectrum_loss(
    hessian_pred,
    hessian_true,
    N,
    k=None,
    alpha=1.0,
    loss_type="eigen",
    dist="mse",  # MAE, MSE, frosq
    **kwargs,
):
    """Compute the eigenspectrum loss for a single Hessian matrix.

    Inspired by wavefunction alignment loss from
    Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems
    See formula (2) and (3) and appendix
    https://openreview.net/pdf/1bcd6e438fe04dffca1ba36654fe64ed6c042d79.pdf#page=38.10
    To get the original loss, use loss_type='wa' instead of 'eigen'.

    To get loss over all eigenvalues/vectors, do:
    loss = eigenspectrum_loss(hessian_pred, hessian_true, N)

    To get additional loss over the subspace, do:
    loss = eigenspectrum_loss(hessian_pred, hessian_true, N, k=[None, k], alpha=[1.0, alpha])

    To only get loss over the subspace, do:
    loss = eigenspectrum_loss(hessian_pred, hessian_true, N, k=k, alpha=alpha)

    k: list of integers, or None to use all eigenvalues/vectors
    alpha: list of floats, or None to use all eigenvalues/vectors
    """
    hessian_true = hessian_true.reshape(N * 3, N * 3)
    hessian_pred = hessian_pred.reshape(N * 3, N * 3)
    if not isinstance(k, Iterable):
        k = [k]
    if not isinstance(alpha, Iterable):
        alpha = [alpha] * len(k)
    loss = 0.0
    for k_i, alpha_i in zip(k, alpha):
        evals_true, evecs_true = torch.linalg.eigh(hessian_true)
        if k_i is None:
            # easy case: use all eigenvalues/vectors
            diff = (evecs_true.T @ (hessian_pred @ evecs_true)) - torch.diag(evals_true)
            loss += alpha_i * _reduce_diff(diff, dist)
        else:
            # use a subspace (subset) of the smallest eigenvalues/vectors
            evecs_true_k = evecs_true[:, :k_i]  # (N*3, k)
            evals_true_k = evals_true[:k_i]
            if loss_type == "wa":  # wavefunction alignment loss
                for i in range(k_i):
                    evec_true_i = evecs_true_k[:, i].reshape(N * 3, 1)  # (N*3, 1)
                    eval_true_i = evals_true_k[i]
                    # (1, N*3) @ (N*3, N*3) @ (N*3, 1) = (1, 1)
                    diff = (
                        evec_true_i.T @ (hessian_pred @ evec_true_i)
                    ) - eval_true_i  # (1)
                    # TODO: how was wa loss in the original paper computed?
                    loss += alpha_i * diff.squeeze().squeeze() ** 2
                    # loss += alpha_i * _reduce_diff(diff, dist)
            elif loss_type == "eigen":  # same as luca's loss
                diff = (evecs_true_k.T @ (hessian_pred @ evecs_true_k)) - torch.diag(
                    evals_true_k
                )  # (3N, 3N)
                loss += alpha_i * _reduce_diff(diff, dist)
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


def get_eigval_eigvec_metrics(hessian_true, hessian_pred, data, prefix=""):
    """We can't normalize concatenated vectors, so we process each vector separately.
    Returns a scalar of similarity averaged over batches.
    data should be a torch_geometric.data.Batch object.
    Warning: detach() is used to avoid memory leaks. Do not use for training!
    """
    # N=15, B=64 ~ 130_000 entries
    natoms = data.natoms
    B = data.batch.max() + 1
    # ptr = data.ptr
    # check
    numels = data.natoms.pow(2).mul(9)
    ptr_hessian = torch.cat([torch.tensor([0], device=numels.device), numels], dim=0)
    ptr_hessian = torch.cumsum(ptr_hessian, dim=0)
    hessian_pred = hessian_pred.view(-1)
    hessian_true = hessian_true.view(-1)
    metrics = {
        "Abs Cosine Sim v1 Eckart": [],
        "Abs Cosine Sim v2 Eckart": [],
        # L1
        "MAE Val1 Eckart": [],
        "MAE Val2 Eckart": [],
        "MAE Eigvals Eckart": [],
    }
    for _b in range(B):
        _start = ptr_hessian[_b].item()
        _numel = ((natoms[_b]) * 3) ** 2
        _end = _numel + _start
        hessian_pred_b = hessian_pred[_start:_end]
        hessian_true_b = hessian_true[_start:_end]

        hessian_pred_b = hessian_pred_b.reshape(natoms[_b] * 3, natoms[_b] * 3)
        hessian_true_b = hessian_true_b.reshape(natoms[_b] * 3, natoms[_b] * 3)

        # mass weight and Eckart project
        cart_coords = data.pos[_b].reshape(-1, 3).to(hessian_pred_b.device)
        atomsymbols = [Z_TO_ATOM_SYMBOL[z] for z in data.z[_b].tolist()]
        hessian_pred_b = eckart_projection_notmw_torch(
            hessian_pred_b, cart_coords, atomsymbols
        )
        hessian_true_b = eckart_projection_notmw_torch(
            hessian_true_b, cart_coords, atomsymbols
        )

        eigvals_true_b, eigvecs_true_b = torch.linalg.eigh(hessian_true_b)
        eigvals_pred_b, eigvecs_pred_b = torch.linalg.eigh(hessian_pred_b)

        # eigenvalues, scalars
        e1_true = eigvals_true_b[0]
        e1_pred = eigvals_pred_b[0]
        e2_true = eigvals_true_b[1]
        e2_pred = eigvals_pred_b[1]
        metrics["MAE Val1 Eckart"].append((e1_true - e1_pred).abs().mean())
        metrics["MAE Val2 Eckart"].append((e2_true - e2_pred).abs().mean())
        # over all eigenvalues
        metrics["MAE Eigvals Eckart"].append(
            torch.mean(torch.abs(eigvals_true_b - eigvals_pred_b))
        )
        # eigenvectors
        v1_true = eigvecs_true_b[:, 0].reshape(-1)
        v1_pred = eigvecs_pred_b[:, 0].reshape(-1)
        v2_true = eigvecs_true_b[:, 1].reshape(-1)
        v2_pred = eigvecs_pred_b[:, 1].reshape(-1)
        metrics["Abs Cosine Sim v1 Eckart"].append(
            torch.abs(torch.dot(v1_true, v1_pred))
        )
        metrics["Abs Cosine Sim v2 Eckart"].append(
            torch.abs(torch.dot(v2_true, v2_pred))
        )

    # average over batches
    for key in metrics:
        metrics[key] = torch.stack(metrics[key]).mean().detach().cpu().item()
    return metrics
