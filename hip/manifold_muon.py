import math

import torch

# from https://github.com/sdbuch/thinky-manifolds
# https://sdbuchanan.com/blog/manifold-muon/

ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# safety factor for numerical stability (but exclude last polynomial)
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


@torch.no_grad()
def msign(G: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Polar Express algorithm for the matrix sign function:
    https://arxiv.org/abs/2505.16932
    """
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G.bfloat16()
    if should_transpose:
        x = x.mT

    x /= x.norm(dim=(-2, -1), keepdim=True) * 1.01
    for step in range(steps):
        a, b, c = (
            ABC_LIST_STABLE[step]
            if step < len(ABC_LIST_STABLE)
            else ABC_LIST_STABLE[-1]
        )
        s = x @ x.mT
        # goal is to compute x = a x + b S x + c S^2 x
        # we can break this up into: x = (a I + (b I + c S) S) x
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    x = torch.nan_to_num(x)
    return x.float()


@torch.no_grad()
def manifold_muon(W, G, eta=0.1, alpha=0.01, steps=100, tol=1e-6):
    """
    Note that this actually implements GD on || G + W @ (L + L.mT) ||_*,
    whereas the blog discusses the parameterization with an extra factor of 2 on L
    It exploits the property that if L is initialized symmetric, it stays symmetric
    """
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    # Ascend on the dual problem to find the update direction A
    for step in range(steps):
        # Update the candidate direction A
        A = msign(G + 2 * W @ Lambda)
        # Measure deviation of A from the tangent space:
        H = W.T @ A + A.T @ W
        # Check the stopping criterion
        if torch.norm(H) / math.sqrt(H.numel()) < tol:
            break
        # Update the dual variable
        Lambda -= alpha * (1 - step / steps) * H
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W


@torch.no_grad()
def manifold_muon_admm(W, G, eta=0.1, steps=10, rho=4.0):
    """Implements GD on || G + W @ (L + L.mT) ||_* (c.f. the blog)"""
    # Ensure that W and G are both tall matrices
    should_tranpose = W.shape[0] < W.shape[1]
    if should_tranpose:
        W = W.T
        G = G.T
    # Initialize the lagrangian, slack, and dual variable
    Lambda = -0.25 * (W.T @ G + G.T @ W)
    X = G + 2 * W @ Lambda
    Omega = torch.zeros_like(X)
    # Solve the dual problem with ADMM to find the update direction A
    for step in range(steps):
        # Update for Lambda (orthonormal least-squares solve)
        P = W.mT @ (1 / rho * Omega + X - G)
        Lambda_upd = 0.25 * (P + P.mT)
        # Update for X (singular value thresholding)
        B = G + 2 * W @ Lambda_upd - 1 / rho * Omega
        eye = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
        P_pos = 0.5 * (eye + msign(B.mT @ B - 1 / rho**2 * eye))
        X_upd = (B - 1 / rho * msign(B)) @ P_pos
        # Update for Omega (dual ascent)
        Omega_upd = Omega + rho * (X_upd - 2 * W @ Lambda_upd - G)
        Lambda, X, Omega = Lambda_upd, X_upd, Omega_upd
    # Calculate A from final ADMM solution
    # (at convergence, G + 2 * W @ Lambda \approx X)
    A = msign(G + 2 * W @ Lambda)
    # Descend on the primal problem
    new_W = W - eta * A
    # Retract to the manifold
    new_W = msign(new_W)
    # Restore the shape of the solution and return
    return new_W.T if should_tranpose else new_W


class ManifoldMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Optimizer that applies `manifold_muon` to matrix-shaped parameters (ndim >= 2)
    and AdamW to all other parameters in separate param groups.

    Expected param_groups format:
      - dict(params=[...], use_manifold_muon=True, lr=float, steps=int, alpha=float, tol=float)
      - dict(params=[...], use_manifold_muon=False, lr=float, betas=tuple, eps=float, weight_decay=float)
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_manifold_muon" in group
            if group["use_manifold_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["steps"] = group.get("steps", 10)
                group["alpha"] = group.get("alpha", 0.01)
                group["tol"] = group.get("tol", 1e-6)
                assert set(group.keys()) == set(
                    ["params", "lr", "steps", "alpha", "tol", "use_manifold_muon"]
                )
            else:
                group["lr"] = group.get("lr", 5e-4)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-12)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "betas",
                        "eps",
                        "weight_decay",
                        "use_manifold_muon",
                    ]
                )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()

        for group in self.param_groups:
            if group["use_manifold_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.ndim >= 2:
                        p.data = manifold_muon(
                            p.data,
                            p.grad,
                            eta=group["lr"],
                            alpha=group["alpha"],
                            steps=group["steps"],
                            tol=group["tol"],
                        )
                    else:
                        p.add_(p.grad, alpha=-group["lr"])  # fallback for vectors
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    b1, b2 = group["betas"]

                    exp_avg.mul_(b1).add_(p.grad, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)

                    bias_c1 = 1 - b1 ** state["step"]
                    bias_c2 = 1 - b2 ** state["step"]
                    denom = (exp_avg_sq / bias_c2).sqrt().add_(group["eps"])
                    step_update = (exp_avg / bias_c1) / denom

                    # Decoupled weight decay (AdamW)
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])

                    p.add_(step_update, alpha=-group["lr"])
        return None
