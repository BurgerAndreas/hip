import torch
import torch.nn.functional as F
from functools import partial
from collections.abc import Iterable


def tensor_info(t):
    if not isinstance(t, torch.Tensor):
        return f"{type(t)}"
    return f"{list(t.shape)} {int(t.numel())} {t.dtype}"


##############################################################################
# predicting the full Hessian matrix


def compute_loss_blockdiagonal_hessian(pred_hessian, true_hessian, loss_fn, data):
    """
    pred_hessian: (B*N, 3, B*N, 3)
    true_hessian: (B*N*3*N*3)
    """
    B = data.batch.max().item() + 1
    N = data.natoms.sum().item()
    pred_hessian = pred_hessian.reshape(N, 3, N, 3)
    loss = 0
    # compare by extracting blocks from full hessian
    atom_offset = 0
    past_entries = 0
    test_mask = torch.zeros_like(pred_hessian)
    for b in range(B):
        n_atoms_batch = data.natoms[b].item()
        assert atom_offset == data.ptr[b].item(), (
            f"Atom offset {atom_offset} does not match batch {b} ptr {data.ptr[b].item()}"
        )
        # Extract the block from the full hessian
        block_start = atom_offset
        block_end = atom_offset + n_atoms_batch
        pred_block = pred_hessian[block_start:block_end, :, block_start:block_end, :]
        test_mask[block_start:block_end, :, block_start:block_end, :] = 1
        # from the flat block
        n_entries_batch = (n_atoms_batch * 3) ** 2
        assert n_entries_batch == pred_block.numel(), (
            f"Number of entries in batch {b} does not match"
        )
        true_block = true_hessian[past_entries : past_entries + n_entries_batch]
        # Compare the blocks
        pred_block = pred_block.reshape(true_block.shape)
        loss += loss_fn(pred_block, true_block)
        atom_offset += n_atoms_batch
        past_entries += n_entries_batch
    # invert the mask
    test_mask = 1 - test_mask
    # all other (non-visited) entries should be zero
    assert torch.allclose(pred_hessian * test_mask, torch.zeros_like(pred_hessian))
    return loss


##############################################################################
# predicting eigenvalues and eigenvectors


def get_vector_loss_fn(loss_name: str):
    if loss_name == "cosine_squared":
        return BatchVectorLoss(cosine_squared_loss)
    elif loss_name == "angle":
        return BatchVectorLoss(L_ang_loss)
    elif loss_name == "cosine":
        return BatchVectorLoss(cosine_loss)
    elif loss_name == "min_l2":
        return BatchVectorLoss(min_l2_loss)
    elif loss_name == "min_l1":
        return BatchVectorLoss(min_l1_loss)
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


def get_vector_similarity_fn(loss_name: str):
    if loss_name == "cosine":
        return BatchVectorLoss(cosine_similarity)
    elif loss_name == "abs_cosine":
        return BatchVectorLoss(abs_cosine_similarity)
    elif loss_name == "dot":
        return BatchVectorLoss(dot_similarity)
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


def get_hessian_loss_fn(loss_name: str, **kwargs):
    if loss_name == "eigenspectrum":
        return BatchHessianLoss(eigenspectrum_loss, **kwargs)
    elif loss_name.lower() in ["mse", "l2"]:
        return torch.nn.MSELoss()  # F.mse_loss
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


def get_scalar_loss_fn(loss_name: str, **kwargs):
    if loss_name == "log_mse":
        return log_mse_loss
    elif loss_name == "huber":
        return HuberLoss(**kwargs)
    elif loss_name.lower() in ["mae", "l1"]:
        return F.l1_loss
    elif loss_name.lower() in ["mse", "l2"]:
        return F.mse_loss
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")


##############################################################################
# vector losses


def batch_similarity(a, b, data, lossfn):
    """We can't normalize concatenated vectors, so we process each vector separately.
    Returns a scalar of similarity averaged over batches.
    lossfn should return a scalar, otherwise it will be averaged over all entries returned.
    data should be a torch_geometric.data.Batch object.
    """
    B = data.batch.max() + 1
    ptr = data.ptr
    natoms = data.natoms
    sim = []
    for _b in range(B):
        _start = ptr[_b] * 3
        _end = (ptr[_b] + natoms[_b]) * 3
        a_b = a[_start:_end]
        b_b = b[_start:_end]
        sim_b = lossfn(a_b, b_b)
        sim.append(sim_b)
    return torch.stack(sim).mean()


class BatchVectorLoss(torch.nn.Module):
    """Wrapper to batch a loss function over vectors."""

    def __init__(self, loss_fn):
        super(BatchVectorLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, data):
        return batch_similarity(pred, target, data, self.loss_fn)


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


def min_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimum L2 (MSE) loss between pred vs target and pred vs -target"""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    # normalize to unit vectors
    pred = pred / torch.norm(pred)
    target = target / torch.norm(target)
    loss_pos = torch.mean((pred - target) ** 2)
    loss_neg = torch.mean((pred + target) ** 2)
    return torch.min(loss_pos, loss_neg)


def min_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimum L1 (MAE) loss between pred vs target and pred vs -target"""
    # eigenvectors are in N*3 space
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    # normalize to unit vectors
    pred = pred / torch.norm(pred)
    target = target / torch.norm(target)
    loss_pos = torch.mean(torch.abs(pred - target))
    loss_neg = torch.mean(torch.abs(pred + target))
    return torch.min(loss_pos, loss_neg)


def L_ang_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Squared angle loss, sign-invariant:
    L = arccos(|pred/|pred| · target/|target||)^2
    """
    # eigenvectors are in N*3 space
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    # normalize to unit vectors
    pred = pred / torch.norm(pred)
    target = target / torch.norm(target)
    # dot product
    dots = torch.sum(pred * target).abs()
    # clamp for numeric stability
    dots = dots.clamp(-1.0 + eps, 1.0 - eps)
    # squared arccosine
    ang = torch.acos(dots)
    return ang.pow(2)


##############################################################################
# scalar losses
def log_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the log-mean-squared-error:
        mean( (log(|pred| + ε) - log(|target| + ε))^2 )

    Supports pred/target of shape (B,), (B,1) or scalar ().
    """
    # squeeze any singleton trailing dimension
    if pred.dim() > 1 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    if target.dim() > 1 and target.size(-1) == 1:
        target = target.squeeze(-1)

    # now pred and target should be same shape: either (B,) or ()
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}")

    lp = torch.log(torch.abs(pred) + epsilon)
    lt = torch.log(torch.abs(target) + epsilon)
    return torch.mean((lp - lt) ** 2)


class HuberLoss(torch.nn.Module):
    """
    Huber Loss implementation for PyTorch.

    Combines MSE for small errors and MAE for large errors.
    Loss = 0.5 * (y_true - y_pred)^2                     if |y_true - y_pred| <= delta
    Loss = delta * |y_true - y_pred| - 0.5 * delta^2    if |y_true - y_pred| > delta

    Args:
        delta (float): Threshold where loss transitions from quadratic to linear
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, delta=1.0, reduction="mean"):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predictions of shape (B,) or (B, 1)
            y_true: Ground truth of shape (B,) or (B, 1)

        Returns:
            loss: Scalar loss value (if reduction != 'none')
        """
        # Ensure both tensors have same shape
        if y_pred.dim() != y_true.dim():
            if y_pred.dim() == 2 and y_pred.size(1) == 1:
                y_pred = y_pred.squeeze(1)
            if y_true.dim() == 2 and y_true.size(1) == 1:
                y_true = y_true.squeeze(1)

        # Calculate absolute error
        abs_error = torch.abs(y_true - y_pred)

        # Huber loss calculation
        quadratic = torch.min(
            abs_error, torch.tensor(self.delta, device=abs_error.device)
        )
        linear = abs_error - quadratic

        loss = 0.5 * quadratic.pow(2) + self.delta * linear

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


##############################################################################
# Eigenspectrum losses that don't require to backprop through .eigh


def batch_hessian_loss(
    hessian_pred,
    hessian_true,
    data,
    lossfn,
    mask_hessian=False,
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
    total_numel = sum(numels)
    # if hessian_pred.numel() != total_numel:
    #     print(
    #         f"{debugstr}\n hessian_pred numel {hessian_pred.numel()} != total_numel {total_numel}"
    #     )
    #     print(" numels:", numels)
    #     print(" natoms:", natoms)
    #     print(" hessian_pred:", tensor_info(hessian_pred))
    #     print(" hessian_true:", tensor_info(hessian_true))
    #     # return torch.tensor(0.0)
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
        # if hessian_pred_b.numel() != _numel:
        #     print(f"Skipping!! {debugstr}")
        #     print(" hessian_pred:", tensor_info(hessian_pred))
        #     print(" hessian_true:", tensor_info(hessian_true))
        #     print(" start, end", _start.item(), _end.item())
        #     print(" numel", _numel)
        #     print(" N", natoms[_b].item())
        #     print(" B", B.item(), set(data.batch.tolist()))
        #     continue
        if mask_hessian:
            # only regress the upper triangular part of the Hessian, including the diagonal
            mask = (
                torch.ones(
                    (ND, ND),
                    device=hessian_pred_b.device,
                    dtype=torch.long,
                )
                .triu(diagonal=0)
                .reshape_as(hessian_pred_b)
            )
            hessian_pred_b = hessian_pred_b[mask]
            hessian_true_b = hessian_true_b[mask]
        loss_b = lossfn(
            hessian_pred=hessian_pred_b,
            hessian_true=hessian_true_b,
            N=natoms[_b].item(),
            **lossfn_kwargs,
        )
        losses.append(loss_b)
    # if _end != hessian_pred.numel():
    #     print(f"Missed or overshot entries: {debugstr}")
    #     print(" hessian_pred:", tensor_info(hessian_pred))
    #     print(" hessian_true:", tensor_info(hessian_true))
    #     print(" start, end", _start.item(), _end.item())
    #     print(" numel", _numel)
    #     print(" N", natoms[_b].item())
    return torch.stack(losses).mean()


class BatchHessianLoss(torch.nn.Module):
    """Wrapper to batch a loss function over hessian matrices."""

    def __init__(self, loss_fn, mask_hessian=False, **kwargs):
        super(BatchHessianLoss, self).__init__()
        self.loss_fn = loss_fn
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.mask_hessian = mask_hessian

    def forward(self, pred, target, data, **kwargs):
        _kwargs = self.kwargs.copy()
        if kwargs is not None:
            _kwargs.update(kwargs)
        return batch_hessian_loss(
            pred, target, data, self.loss_fn, mask_hessian=self.mask_hessian, **_kwargs
        )


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
    dof_filter_fn=None,
    loss_type="eigen",
    dist="mse",  # MAE, MSE, frosq
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
    dof_filter_fn: function that takes a list of eigenvalues and returns a list of booleans
    to filter out the eigenvalues/vectors to use.
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


class L1HessianLoss(torch.nn.Module):
    def forward(self, hessian_pred, hessian_true, N=None, **kwargs):
        return torch.mean(torch.abs(hessian_pred - hessian_true))


class L2HessianLoss(torch.nn.Module):
    def forward(self, hessian_pred, hessian_true, N=None, **kwargs):
        return torch.mean((hessian_pred - hessian_true) ** 2)


# by HORM
def hess2eigenvalues(hess):
    """Convert Hessian to eigenvalues with proper unit conversion"""
    hartree_to_ev = 27.2114
    bohr_to_angstrom = 0.529177
    ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev

    hess = hess * ev_angstrom_2_to_hartree_bohr_2
    eigen_values, _ = torch.linalg.eigh(hess)
    return eigen_values


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
    total_numel = sum(numels)
    hessian_pred = hessian_pred.view(-1)
    hessian_true = hessian_true.view(-1)
    metrics = {
        "Abs Cosine Sim v1": [],
        "Abs Cosine Sim v2": [],
        # L2
        "MSE Vec1": [],
        "MSE Vec2": [],
        "MSE Val1": [],
        "MSE Val2": [],
        # L1
        "MAE Vec1": [],
        "MAE Vec2": [],
        "MAE Val1": [],
        "MAE Val2": [],
        "MAE Eigvals": [],
    }
    for _b in range(B):
        _start = ptr_hessian[_b].item()
        _numel = ((natoms[_b]) * 3) ** 2
        _end = _numel + _start
        hessian_pred_b = hessian_pred[_start:_end]
        hessian_true_b = hessian_true[_start:_end]

        hessian_pred_b = hessian_pred_b.reshape(natoms[_b] * 3, natoms[_b] * 3)
        hessian_true_b = hessian_true_b.reshape(natoms[_b] * 3, natoms[_b] * 3)

        eigvals_true_b, eigvecs_true_b = torch.linalg.eigh(hessian_true_b)
        eigvals_pred_b, eigvecs_pred_b = torch.linalg.eigh(hessian_pred_b)

        # eigenvalues, scalars
        e1_true = eigvals_true_b[0]
        e1_pred = eigvals_pred_b[0]
        e2_true = eigvals_true_b[1]
        e2_pred = eigvals_pred_b[1]
        metrics["MSE Val1"].append((e1_true - e1_pred).pow(2).mean())
        metrics["MSE Val2"].append((e2_true - e2_pred).pow(2).mean())
        metrics["MAE Val1"].append((e1_true - e1_pred).abs().mean())
        metrics["MAE Val2"].append((e2_true - e2_pred).abs().mean())
        # over all eigenvalues
        metrics["MAE Eigvals"].append(
            torch.mean(torch.abs(eigvals_true_b - eigvals_pred_b))
        )
        # eigenvectors
        v1_true = eigvecs_true_b[:, 0].reshape(-1)
        v1_pred = eigvecs_pred_b[:, 0].reshape(-1)
        v2_true = eigvecs_true_b[:, 1].reshape(-1)
        v2_pred = eigvecs_pred_b[:, 1].reshape(-1)
        metrics["Abs Cosine Sim v1"].append(torch.abs(torch.dot(v1_true, v1_pred)))
        metrics["Abs Cosine Sim v2"].append(torch.abs(torch.dot(v2_true, v2_pred)))
        metrics["MSE Vec1"].append(torch.abs(v1_true - v1_pred).pow(2).mean())
        metrics["MSE Vec2"].append(torch.abs(v2_true - v2_pred).pow(2).mean())
        metrics["MAE Vec1"].append(torch.abs(v1_true - v1_pred).mean())
        metrics["MAE Vec2"].append(torch.abs(v2_true - v2_pred).mean())

    # average over batches
    for key in metrics:
        metrics[key] = torch.stack(metrics[key]).mean().detach().cpu().item()
    return metrics


##############################################################################
if __name__ == "__main__":
    B, N = 4, 5
    v_pred = torch.randn(B, N, 3)
    v_true = torch.randn(B, N, 3)

    print(f"\nShould be the same:")
    loss1 = cosine_squared_loss(v_pred, v_true)
    print(f"cosine_squared_loss:                {loss1}")
    loss1_flipped = cosine_squared_loss(v_pred, -v_true)
    print(f"cosine_squared_loss (sign flipped): {loss1_flipped}")
    print(
        f"cosine_squared_loss (shape B, N*3): {cosine_squared_loss(v_pred.reshape(B, N * 3), v_true.reshape(B, N * 3))}"
    )

    print(f"\nShould be the same:")
    loss2 = L_ang_loss(v_pred, v_true)
    print(f"L_ang_loss:                {loss2}")
    loss2_flipped = L_ang_loss(v_pred, -v_true)
    print(f"L_ang_loss (sign flipped): {loss2_flipped}")
    print(
        f"L_ang_loss (shape B, N*3): {L_ang_loss(v_pred.reshape(B, N * 3), v_true.reshape(B, N * 3))}"
    )

    print(f"\nShould be the same:")
    loss3 = cosine_loss(v_pred, v_true)
    print(f"_cosine_loss:                {loss3}")
    loss3_flipped = cosine_loss(v_pred, -v_true)
    print(f"_cosine_loss (sign flipped): {loss3_flipped}")
    print(
        f"_cosine_loss (shape B, N*3): {cosine_loss(v_pred.reshape(B, N * 3), v_true.reshape(B, N * 3))}"
    )

    print(f"\nShould be the same:")
    loss4 = min_l2_loss(v_pred, v_true)
    print(f"_min_l2_loss:                {loss4}")
    loss4_flipped = min_l2_loss(v_pred, -v_true)
    print(f"_min_l2_loss (sign flipped): {loss4_flipped}")
    print(
        f"_min_l2_loss (shape B, N*3): {min_l2_loss(v_pred.reshape(B, N * 3), v_true.reshape(B, N * 3))}"
    )

    print(f"\nShould be the same:")
    loss5 = min_l1_loss(v_pred, v_true)
    print(f"_min_l1_loss:                {loss5}")
    loss5_flipped = min_l1_loss(v_pred, -v_true)
    print(f"_min_l1_loss (sign flipped): {loss5_flipped}")
    print(
        f"_min_l1_loss (shape B, N*3): {min_l1_loss(v_pred.reshape(B, N * 3), v_true.reshape(B, N * 3))}"
    )

    ######################################################################
    # scalar losses

    # Example usage
    batch_size = 32

    # Test with different input shapes
    y_true_1d = torch.randn(batch_size)  # Shape: (B,)
    y_pred_1d = torch.randn(batch_size)  # Shape: (B,)

    y_true_2d = y_true_1d.unsqueeze(1)  # Shape: (B, 1)
    y_pred_2d = y_pred_1d.unsqueeze(1)  # Shape: (B, 1)

    # Initialize Huber loss
    huber_loss_fn = HuberLoss(delta=1.0, reduction="mean")

    print(f"\nComparison of scalar losses:")
    mse_loss_1d = F.mse_loss(y_pred_1d, y_true_1d)
    mse_loss_2d = F.mse_loss(y_pred_2d, y_true_2d)
    print(f"MSE Loss (1D): {mse_loss_1d.item():.4f}")
    print(f"MSE Loss (2D): {mse_loss_2d.item():.4f}")
    mae_loss_1d = F.l1_loss(y_pred_1d, y_true_1d)
    mae_loss_2d = F.l1_loss(y_pred_2d, y_true_2d)
    print(f"MAE Loss (1D): {mae_loss_1d.item():.4f}")
    print(f"MAE Loss (2D): {mae_loss_2d.item():.4f}")
    loss_1d = huber_loss_fn(y_pred_1d, y_true_1d)
    loss_2d = huber_loss_fn(y_pred_2d, y_true_2d)
    print(f"Huber Loss (δ=1.0) (1D): {loss_1d.item():.4f}")
    print(f"Huber Loss (δ=1.0) (2D): {loss_2d.item():.4f}")

    loss6 = log_mse_loss(y_pred_1d, y_true_1d)
    print(f"log_mse_loss (1D): {loss6.item():.4f}")
    loss6_flipped = log_mse_loss(y_pred_2d, y_true_2d)
    print(f"log_mse_loss (2D): {loss6_flipped.item():.4f}")
