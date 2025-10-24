import torch

from hip.masses import MASS_DICT
from nets.prediction_utils import Z_TO_ATOM_SYMBOL
from hip.frequency_analysis import eckart_projection_notmw_torch as projection_not_differentiable

# ---- helpers --------------------------------------------------------------

def _to_torch_double(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def mass_weigh_hessian_torch(hessian, masses3d):
    """M^{-1/2} H M^{-1/2} (all torch-autodiff friendly)."""
    h_t = _to_torch_double(hessian, device=hessian.device)
    m_t = _to_torch_double(masses3d, device=hessian.device)
    mm_sqrt_inv = torch.diag(1.0 / torch.sqrt(m_t))
    return mm_sqrt_inv @ h_t @ mm_sqrt_inv


def _center_of_mass(coords3d, masses):
    total_mass = torch.sum(masses)
    return (coords3d * masses[:, None]).sum(dim=0) / total_mass


# ---- differentiable Eckart generators & projector ------------------------

def eckart_B_massweighted_torch(cart_coords, masses, eps=1e-12):
    """
    Build the 6 Eckart generators (3 translations, 3 rotations) in mass-weighted space:
      B \in R^{3N x 6}, columns are orthogonal to vibrations under the Euclidean metric.
    No eigen-decompositions, no QR -> smooth grads.
    """
    coords = _to_torch_double(cart_coords)
    masses = _to_torch_double(masses, device=coords.device)

    # shapes & mass factors
    xyz = coords.reshape(-1, 3)                                # (N, 3)
    N = xyz.shape[0]
    sqrt_m = torch.sqrt(masses)                                # (N,)
    sqrt_m3 = sqrt_m.repeat_interleave(3)                      # (3N,)

    # center geometry at mass-weighted COM (still differentiable)
    com = _center_of_mass(xyz, masses)                         # (3,)
    r = xyz - com[None, :]                                     # (N, 3)

    # --- 3 translations (mass-weighted unit directions) ---
    # For axis a, component-wise displacement is constant 1. After mass-weighting multiply by sqrt(m_i).
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=coords.device)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=coords.device)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=coords.device)
    Tcols = []
    for e in (ex, ey, ez):
        tiled = e.repeat(N)                                    # (3N,)
        col = sqrt_m3 * tiled                                  # mass-weighted translation
        # normalize to avoid scale pathologies (keeps smooth grads)
        col = col / (col.norm() + eps)
        Tcols.append(col)

    # --- 3 rotations (infinitesimal) ---
    # For axis ω = ex/ey/ez: δr_i = ω × r_i; then mass-weight by sqrt(m_i) per component.
    # ex × r = (0, -r_z, r_y); ey × r = (r_z, 0, -r_x); ez × r = (-r_y, r_x, 0)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    R_ex = torch.stack([torch.zeros_like(rx), -rz, ry], dim=1)  # (N,3)
    R_ey = torch.stack([rz, torch.zeros_like(ry), -rx], dim=1)
    R_ez = torch.stack([-ry, rx, torch.zeros_like(rz)], dim=1)
    Rcols = []
    for Raxis in (R_ex, R_ey, R_ez):
        col = (Raxis * sqrt_m[:, None]).reshape(-1)             # mass-weighted
        # normalize each rotational generator for numerical stability
        col = col / (col.norm() + eps)
        Rcols.append(col)

    # Stack to B: (3N, 6)
    B = torch.stack(Tcols + Rcols, dim=1)
    return B  # columns span translations+rotations in MW metric


def eckart_projector_massweighted_torch(cart_coords, masses, eps=1e-10):
    """
    Return the vibrational projector P \in R^{3N x 3N} in the *mass-weighted* space.
    P = I - B (B^T B + eps I)^{-1} B^T
    """
    B = eckart_B_massweighted_torch(cart_coords, masses, eps=eps)   # (3N, 6)
    G = B.transpose(0, 1) @ B                                       # (6, 6)
    # stabilized inverse; cholesky preferred when G is SPD (generic non-linear molecules)
    try:
        L = torch.linalg.cholesky(G + eps * torch.eye(6, dtype=G.dtype, device=G.device))
        Ginvt_Bt = torch.cholesky_solve(B.transpose(0, 1), L)       # (6, 3N)
    except RuntimeError:
        Ginvt_Bt = torch.linalg.solve(G + eps * torch.eye(6, dtype=G.dtype, device=G.device),
                                      B.transpose(0, 1))
    P = torch.eye(B.shape[0], dtype=B.dtype, device=B.device) - B @ Ginvt_Bt
    # make numerically symmetric/idempotent
    P = 0.5 * (P + P.transpose(0, 1))
    return P


# ---- your top-level function (differentiable) ----------------------------

def differentiable_eckart_projection_notmw_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,   # kept for API compat; unused here
):
    """
    Eckart projection for a *non*-mass-weighted Hessian:
      1) mass-weight H
      2) build MW projector P
      3) project: P H_mw P
    Everything is differentiable w.r.t. coords, Hessian elements, and (if you wish) masses.
    """
    device = hessian.device
    dtype = torch.float64

    masses_t = torch.tensor([MASS_DICT[atom.lower()] for atom in atomsymbols],
                            dtype=dtype, device=device)
    masses3d_t = masses_t.repeat_interleave(3)

    # 1) mass-weight the Hessian
    H_mw = mass_weigh_hessian_torch(hessian, masses3d_t)             # (3N,3N)

    # 2) projector in the MW space
    P = eckart_projector_massweighted_torch(cart_coords, masses_t)   # (3N,3N)

    # 3) project and symmetrize
    H_proj = P @ H_mw @ P
    H_proj = 0.5 * (H_proj + H_proj.transpose(0, 1))
    return H_proj


# ---- compare with non-differentiable version ---------------------------------------------------

def compare_truncated_vs_projected(
    H_trunc: torch.Tensor,      # (3N-6, 3N-6) reduced (basis-truncated) Hessian
    H_proj: torch.Tensor,       # (3N, 3N) projected Hessian (with 6 near-zeros)
    V: torch.Tensor | None = None, # optional (3N, 3N-6) basis used for truncation
    zero_tol: float = 1e-8,     # threshold to drop rigid-body zeros in H_proj
):
    """
    Compute eigensystems and compare.
    Returns a dict with eigenvalue diffs and (if V provided) subspace/eigenvector alignment.
    """
    dtype = torch.float64
    H_trunc = H_trunc.to(dtype)
    H_proj  = H_proj.to(dtype)
    H_trunc = 0.5 * (H_trunc + H_trunc.T)
    H_proj  = 0.5 * (H_proj  + H_proj.T)

    # --- eigensystems ---
    # Reduced (vibrational) eigensystem
    eval_red, evec_red = torch.linalg.eigh(H_trunc)  # shapes: (K,), (K,K), K=3N-6

    # Projected: take vibrational (non-zero) spectrum only
    eval_full, evec_full = torch.linalg.eigh(H_proj)  # (3N,), (3N,3N)
    keep = torch.abs(eval_full) > zero_tol
    eval_vib = eval_full[keep]              # (K,)
    evec_vib = evec_full[:, keep]           # (3N, K)

    # Sort both vibrational spectra
    idx_red  = torch.argsort(eval_red)
    idx_vib  = torch.argsort(eval_vib)
    eval_red = eval_red[idx_red]
    evec_red = evec_red[:, idx_red]
    eval_vib = eval_vib[idx_vib]
    evec_vib = evec_vib[:, idx_vib]

    # --- eigenvalue comparison ---
    if eval_red.numel() != eval_vib.numel():
        raise ValueError(f"Vibrational eigenvalue count mismatch: "
                         f"{eval_red.numel()} vs {eval_vib.numel()} (check zero_tol)")

    ev_diff = eval_vib - eval_red
    ev_abs_err_max = torch.max(torch.abs(ev_diff)).item()
    ev_abs_err_mean = torch.mean(torch.abs(ev_diff)).item()
    ev_rel_err_mean = (torch.mean(torch.abs(ev_diff) / (torch.abs(eval_red) + 1e-30))).item()

    result = {
        "eigenvalues_reduced": eval_red,
        "eigenvalues_projected_vibrational": eval_vib,
        "eigenvalue_abs_err_max": ev_abs_err_max,
        "eigenvalue_abs_err_mean": ev_abs_err_mean,
        "eigenvalue_rel_err_mean": ev_rel_err_mean,
    }

    # --- eigenvector / subspace comparison (requires V) ---
    if V is not None:
        V = V.to(dtype)
        # Lift reduced eigenvectors to full space: U_red_full = V @ evec_red
        U_red_full = V @ evec_red              # (3N, K)
        # Orthonormalize numerically (good hygiene)
        U_red_full, _ = torch.linalg.qr(U_red_full, mode="reduced")  # (3N, K)

        # Orthonormalize evec_vib too (it should already be)
        U_vib = evec_vib
        # Subspace principal angles via SVD of cross-Gram
        C = U_vib.T @ U_red_full               # (K, K)
        svals = torch.linalg.svdvals(C)        # cosines of principal angles
        svals = torch.clamp(svals, 0.0, 1.0)
        principal_angles = torch.arccos(svals) # radians

        # Report permutation/degeneracy-invariant metrics
        result.update({
            "principal_angles_rad": principal_angles,
            "principal_angle_max_deg": torch.rad2deg(torch.max(principal_angles)).item(),
            "principal_angle_mean_deg": torch.rad2deg(torch.mean(principal_angles)).item(),
            "mean_vector_overlap": torch.mean(svals).item(),  # = mean cos(angle)
        })

        # Optional: per-mode overlaps after a cheap greedy matching (not assignment-optimal,
        # but quick and informative). Comment out if you prefer only subspace metrics.
        M = torch.abs(U_vib.T @ (V @ evec_red))  # (K,K) absolute overlaps
        # greedy: match each vib vector to the best reduced one
        perm = torch.argmax(M, dim=1)
        per_mode_overlap = M[torch.arange(M.shape[0]), perm]
        result.update({
            "per_mode_overlap_greedy": per_mode_overlap,
            "per_mode_overlap_mean_greedy": torch.mean(per_mode_overlap).item(),
            "per_mode_overlap_min_greedy": torch.min(per_mode_overlap).item(),
        })
    else:
        result["note"] = (
            "Eigenvector comparison requires the truncation basis V (shape 3N×(3N-6)). "
            "Eigenvalues are compared; subspace angles skipped."
        )

    return result

if __name__ == "__main__":
    
    from torch_geometric.data import Batch as TGBatch
    from torch_geometric.data import Data as TGData
    from hip.inference_utils import get_model_from_checkpoint
    from nets.prediction_utils import Z_TO_ATOM_SYMBOL
    import os
    
    # Load model with the specified checkpoint
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/hip_v2.ckpt")
    
    # Load model directly
    model = get_model_from_checkpoint(checkpoint_path, device="cuda")
    
    # Example molecular system
    atoms = ["H", "O", "H", "C"]
    atomic_nums = [1, 8, 1, 6]  # H, O, H, C
    nat = len(atoms)
    coords3d = torch.randn(nat, 3)  # Random coordinates in Angstrom
    
    # Convert to torch tensors with gradients enabled
    coords_torch = torch.tensor(coords3d, dtype=torch.float64, requires_grad=True)
    
    # Ensure coordinates are a leaf on the model device with gradients enabled
    coords = torch.tensor(coords3d, dtype=torch.float32, device=model.device, requires_grad=True)
    # Create batch using the same code as in EquiformerTorchCalculator
    data = TGData(
        pos=coords, 
        z=torch.as_tensor(atomic_nums, dtype=torch.int64, device=model.device),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64, device=model.device),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64, device=model.device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=model.device),
    )
    batch = TGBatch.from_data_list([data])
    
    # Get hessian from model using autograd
    with torch.enable_grad():
        # batch.pos.requires_grad = True
        energy, forces, out = model.forward(
            batch,
            otf_graph=True,
        )
        hessian = out["hessian"]
        
        # Convert atomic numbers to symbols
        atomsymbols = [Z_TO_ATOM_SYMBOL[z] for z in atomic_nums]
        hessian = hessian.reshape(coords.numel(), coords.numel())
        hessian_proj = differentiable_eckart_projection_notmw_torch(hessian, coords.reshape(-1, 3), atomsymbols)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(hessian_proj)
        
        # Compute product of the two smallest eigenvalues
        product = eigvals[0] * eigvals[1]
    
    # Compute gradient
    grad = torch.autograd.grad(
        product, 
        # batch.pos.reshape(-1, 3), 
        coords,
        retain_graph=True,
        create_graph=True
    )[0]
    
    # neg_inds = eigvals < ev_thresh
    # neg_eigvals = eigvals[neg_inds]
    # neg_num = torch.sum(neg_inds)
    
    print(f"\nProduct: {product}")
    print(f"Gradient: {grad}")
    print(f"Gradient norm: {torch.norm(grad):.6f}")
    
    # compare with non-differentiable version
    print("coords.shape: ", coords.shape)
    print("hessian.shape: ", hessian.shape)
    print("hessian_proj.shape: ", hessian_proj.shape)
    hessian_trunc, V = projection_not_differentiable(hessian.detach(), coords.detach(), atomsymbols, return_basis=True)
    print("hessian_trunc.shape: ", hessian_trunc.shape)
    
    # H_trunc: (3N-6,3N-6)   from basis truncation
    # H_proj:  (3N,3N)       from projector method
    # V:       (3N,3N-6)     the vibrational basis used to make H_trunc (optional but needed for vector tests)

    stats = compare_truncated_vs_projected(hessian_trunc, hessian_proj, V=V, zero_tol=1e-8)
    print("max |Δλ| =", stats["eigenvalue_abs_err_max"])
    if "principal_angle_max_deg" in stats:
        print("max principal angle (deg) =", stats["principal_angle_max_deg"])