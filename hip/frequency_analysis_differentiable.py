# Adapted from pysisyphus

import numpy as np
import scipy.constants as spc
import torch

from hip.masses import MASS_DICT
from nets.prediction_utils import Z_TO_ATOM_SYMBOL

# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# Å -> Bohr conversion factor
ANG2BOHR = 1 / BOHR2ANG
# Hartree to J
AU2J = spc.value("Hartree energy")
# Speed of light in m/s
C = spc.c
NA = spc.Avogadro


def _to_torch_double(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def inertia_tensor_torch(coords3d, masses):
    """Inertia tensor using torch."""
    coords3d_t = _to_torch_double(coords3d)
    masses_t = _to_torch_double(masses)
    x, y, z = coords3d_t.T
    squares = torch.sum(coords3d_t**2 * masses_t[:, None], dim=0)
    I_xx = squares[1] + squares[2]
    I_yy = squares[0] + squares[2]
    I_zz = squares[0] + squares[1]
    I_xy = -torch.sum(masses_t * x * y)
    I_xz = -torch.sum(masses_t * x * z)
    I_yz = -torch.sum(masses_t * y * z)
    return torch.stack(
        [
            torch.stack([I_xx, I_xy, I_xz]),
            torch.stack([I_xy, I_yy, I_yz]),
            torch.stack([I_xz, I_yz, I_zz]),
        ]
    )


def get_trans_rot_vectors_torch(cart_coords, masses, rot_thresh=1e-6):
    """Torch version of get_trans_rot_vectors."""
    cart_coords_t = _to_torch_double(cart_coords)
    masses_t = _to_torch_double(masses)

    coords3d = cart_coords_t.reshape(-1, 3)
    total_mass = torch.sum(masses_t)
    com = (coords3d * masses_t[:, None]).sum(dim=0) / total_mass
    coords3d_centered = coords3d - com[None, :]

    _, Iv = torch.linalg.eigh(inertia_tensor_torch(coords3d, masses_t))
    Iv = Iv.T  # rows are eigenvectors

    masses_rep = masses_t.repeat_interleave(3)
    sqrt_masses = torch.sqrt(masses_rep)
    num = masses_t.numel()

    # Translation vectors (mass-weighted unit vectors along axes)
    trans_vecs = []  # (3, 3N)
    device = cart_coords_t.device
    for vec in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        tiled = _to_torch_double(vec, device=device).repeat(num)
        v = sqrt_masses * tiled
        trans_vecs.append(v / torch.linalg.norm(v))  # (3N,)

    # Rotation vectors
    rot_vecs = torch.zeros(
        (3, cart_coords_t.numel()), dtype=torch.float64, device=device
    )
    for i in range(masses_t.size(0)):
        p_vec = Iv @ coords3d_centered[i]
        for ix in range(3):
            rot_vecs[0, 3 * i + ix] = Iv[2, ix] * p_vec[1] - Iv[1, ix] * p_vec[2]
            rot_vecs[1, 3 * i + ix] = Iv[2, ix] * p_vec[0] - Iv[0, ix] * p_vec[2]
            rot_vecs[2, 3 * i + ix] = Iv[0, ix] * p_vec[1] - Iv[1, ix] * p_vec[0]
    rot_vecs = rot_vecs * sqrt_masses[None, :]  # (3, 3N)

    # Drop vectors with vanishing norms
    norms = torch.linalg.norm(rot_vecs, dim=1)  # (3)
    keep = norms > rot_thresh
    rot_vecs = rot_vecs[keep]  # (3, 3N)

    trans_vecs = torch.stack(trans_vecs)  # (3, 3N)
    tr_vecs = torch.cat([trans_vecs, rot_vecs], dim=0)  # (6, 3N)
    Q, _ = torch.linalg.qr(tr_vecs.T)
    return Q.T  # (6, 3N)


def get_trans_rot_projector_torch(cart_coords, masses, full=False):
    tr_vecs = get_trans_rot_vectors_torch(cart_coords, masses=masses)
    if full:
        n = tr_vecs.size(1)
        P = torch.eye(n, dtype=tr_vecs.dtype, device=tr_vecs.device)
        for tr_vec in tr_vecs:
            P = P - torch.outer(tr_vec, tr_vec)
        return P
    else:
        U, S, _ = torch.linalg.svd(tr_vecs.T, full_matrices=True)
        P = U[:, S.numel() :].T
        return P


def mass_weigh_hessian_torch(hessian, masses3d):
    """mass-weighted hessian M^(-1/2) H M^(-1/2) using torch."""
    h_t = _to_torch_double(hessian, device=hessian.device)
    m_t = _to_torch_double(masses3d, device=hessian.device)
    mm_sqrt_inv = torch.diag(
        1.0 / torch.sqrt(m_t),
    )
    return mm_sqrt_inv @ h_t @ mm_sqrt_inv


def unweight_mw_hessian_torch(mw_hessian, masses3d):
    h_t = _to_torch_double(mw_hessian, device=mw_hessian.device)
    m_t = _to_torch_double(masses3d, device=mw_hessian.device)
    mm_sqrt = torch.diag(
        torch.sqrt(m_t),
    )
    return mm_sqrt @ h_t @ mm_sqrt


def eckart_projection_notmw_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
):
    """Eckart projection starting from not-mass-weighted Hessian (torch).

    hessian: torch.Tensor (N*3, N*3)
    cart_coords: torch.Tensor (N*3)
    atomsymbols: list[str] (N)
    """
    masses_t = torch.tensor(
        [MASS_DICT[atom.lower()] for atom in atomsymbols],
        dtype=torch.float64,
        device=hessian.device,
    )
    masses3d_t = masses_t.repeat_interleave(3)

    mw_hessian_t = mass_weigh_hessian_torch(hessian, masses3d_t)
    P_t = get_trans_rot_projector_torch(cart_coords, masses=masses_t, full=False)
    proj_hessian_t = P_t @ mw_hessian_t @ P_t.T
    proj_hessian_t = (proj_hessian_t + proj_hessian_t.T) / 2.0
    return proj_hessian_t


def analyze_frequencies_torch(
    hessian: torch.Tensor,  # eV/Angstrom^2
    cart_coords: torch.Tensor,  # Angstrom
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
):
    cart_coords = cart_coords.reshape(-1, 3).to(hessian.device)
    hessian = hessian.reshape(cart_coords.numel(), cart_coords.numel())

    if isinstance(atomsymbols[0], torch.Tensor):
        atomsymbols = atomsymbols.tolist()
    if not isinstance(atomsymbols[0], str):
        # atomic numbers were passed instead of symbols
        atomsymbols = [Z_TO_ATOM_SYMBOL[z] for z in atomsymbols]

    proj_hessian = eckart_projection_notmw_torch(hessian, cart_coords, atomsymbols)
    eigvals, eigvecs = torch.linalg.eigh(proj_hessian)

    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = sum(neg_inds)
    # # eigval_str = np.array2string(eigvals[:10], precision=4)
    # if neg_num > 0:
    #     wavenumbers = eigval_to_wavenumber(neg_eigvals)
    #     # wavenum_str = np.array2string(wavenumbers, precision=2)
    # else:
    #     wavenumbers = None
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        # "wavenumbers": wavenumbers,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(atomsymbols),
    }


def compute_smallest_eigenvalues_product_gradient(
    hessian: torch.Tensor,  # (N*3, N*3) - requires grad
    cart_coords: torch.Tensor,  # (N*3,) - requires grad  
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
    project: bool = True,
):
    """
    Compute the gradient of the product of the two smallest eigenvalues 
    with respect to input positions.
    
    Args:
        hessian: Hessian matrix with requires_grad=True
        cart_coords: Cartesian coordinates with requires_grad=True
        atomsymbols: List of atom symbols
        ev_thresh: Threshold for negative eigenvalues
        
    Returns:
        dict: Contains eigenvalues, eigenvectors, and gradient of product
    """
    cart_coords = cart_coords.reshape(-1, 3).to(hessian.device)
    hessian = hessian.reshape(cart_coords.numel(), cart_coords.numel())

    if isinstance(atomsymbols[0], torch.Tensor):
        atomsymbols = atomsymbols.tolist()
    if not isinstance(atomsymbols[0], str):
        # atomic numbers were passed instead of symbols
        atomsymbols = [Z_TO_ATOM_SYMBOL[z] for z in atomsymbols]

    # Perform Eckart projection
    if project:
        proj_hessian = eckart_projection_notmw_torch(hessian, cart_coords, atomsymbols)
    else:
        proj_hessian = hessian
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = torch.linalg.eigh(proj_hessian)
    
    # Get the two smallest eigenvalues
    smallest_two = torch.sort(eigvals)[0][:2]
    
    # Compute product of the two smallest eigenvalues
    product = smallest_two[0] * smallest_two[1]
    
    # Compute gradient
    grad = torch.autograd.grad(
        product, 
        cart_coords, 
        retain_graph=True,
        create_graph=True
    )[0]
    
    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = torch.sum(neg_inds)
    
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "smallest_two": smallest_two,
        "product": product,
        "gradient": grad,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(atomsymbols),
    }


def compute_eigenvalues_product_gradient_with_model(
    model,
    coords: torch.Tensor,  # (N, 3) - requires grad
    atomic_nums: list[int],
    ev_thresh: float = -1e-6,
):
    """
    Compute the gradient of the product of the two smallest eigenvalues 
    using a trained model to compute the Hessian.
    
    Args:
        model: Trained model instance
        coords: Cartesian coordinates with requires_grad=True
        atomic_nums: List of atomic numbers
        ev_thresh: Threshold for negative eigenvalues
        
    Returns:
        dict: Contains eigenvalues, eigenvectors, and gradient of product
    """
    from torch_geometric.data import Batch as TGBatch
    from torch_geometric.data import Data as TGData
    
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
        
        # Compute gradient of product of smallest eigenvalues
        hessian = hessian.reshape(coords.numel(), coords.numel())

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(hessian)
        
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
    
    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = torch.sum(neg_inds)
    
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "smallest_two": [eigvals[0], eigvals[1]],
        "product": product,
        "gradient": grad,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(atomsymbols),
    }


if __name__ == "__main__":
    # Usage example: frequency analysis with PyTorch
    rng = np.random.default_rng(0)
    
    # Example molecular system
    atoms = ["H", "O", "H", "C"]
    nat = len(atoms)
    coords3d = rng.normal(size=(nat, 3))  # Random coordinates in Angstrom
    n3 = 3 * nat
    A = rng.normal(size=(n3, n3))
    hessian = (A + A.T) / 2.0  # Symmetric Hessian matrix
    
    print(f"System: {atoms}")
    print(f"Coordinates shape: {coords3d.shape}")
    print(f"Hessian shape: {hessian.shape}")
    
    # Convert to torch tensors with gradients enabled
    hessian_torch = torch.tensor(hessian, dtype=torch.float64, requires_grad=True)
    coords_torch = torch.tensor(coords3d, dtype=torch.float64, requires_grad=True)
    
    # Perform frequency analysis
    result = analyze_frequencies_torch(hessian_torch, coords_torch, atoms)
    
    print(f"\nFrequency analysis results:")
    print(f"Number of atoms: {result['natoms']}")
    print(f"Number of negative eigenvalues: {result['neg_num']}")
    print(f"Eigenvalues shape: {result['eigvals'].shape}")
    print(f"Eigenvectors shape: {result['eigvecs'].shape}")
    
    if result['neg_num'] > 0:
        print(f"Negative eigenvalues: {result['neg_eigvals']}")
    
    # Example: Compute gradient of product of two smallest eigenvalues
    print(f"\nGradient computation example:")
    grad_result = compute_smallest_eigenvalues_product_gradient(
        hessian_torch, coords_torch, atoms
    )
    
    print(f"Two smallest eigenvalues: {grad_result['smallest_two']}")
    print(f"Product of smallest eigenvalues: {grad_result['product']}")
    print(f"Gradient shape: {grad_result['gradient'].shape}")
    print(f"Gradient: {grad_result['gradient']}")
    print(f"Gradient norm: {torch.norm(grad_result['gradient']):.6f}")
    
    # Example with Equiformer model
    print(f"\n" + "="*50)
    print("Hessian from Equiformer model example:")
    print("="*50)
    
    from hip.inference_utils import get_model_from_checkpoint
    import os
    
    # Load model with the specified checkpoint
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/hip_v2.ckpt")
    
    # Load model directly
    model = get_model_from_checkpoint(checkpoint_path, device="cuda")
    
    # Create coordinates with atomic numbers
    atomic_nums = [1, 8, 1, 6]  # H, O, H, C
    coords = torch.tensor(coords3d, dtype=torch.float32, device=model.device, requires_grad=True)

    # Small differentiability test: does model Hessian depend on coords?
    from torch_geometric.data import Batch as TGBatch
    from torch_geometric.data import Data as TGData
    data = TGData(
        pos=coords,
        z=torch.as_tensor(atomic_nums, dtype=torch.int64, device=model.device),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64, device=model.device),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64, device=model.device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=model.device),
    )
    batch = TGBatch.from_data_list([data])
    with torch.enable_grad():
        _, _, out = model.forward(batch, otf_graph=True)
        h_small = out["hessian"]
        s = h_small.sum()
        g = torch.autograd.grad(s, coords, retain_graph=True, create_graph=True)[0]
        print(f"Hessian requires_grad: {h_small.requires_grad}")
        print(f"d(sum(H))/d(coords) is None: {g is None}")
        print(f"d(sum(H))/d(coords) norm: {g.norm().item():.6f}")
    
    # Example with Equiformer model
    print(f"\n" + "="*50)
    print("Product of smallest eigenvalues from Equiformer model example:")
    print("="*50)
    
    
    # Compute gradient using the model
    grad_result = compute_eigenvalues_product_gradient_with_model(
        model, coords3d, atomic_nums
    )
    
    print(f"Model-based gradient computation:")
    print(f"Two smallest eigenvalues: {grad_result['smallest_two']}")
    print(f"Product: {grad_result['product']}")
    print(f"Gradient norm: {torch.norm(grad_result['gradient']):.6f}")
    print(f"Number of negative eigenvalues: {grad_result['neg_num']}")
        
