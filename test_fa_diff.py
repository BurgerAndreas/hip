# Adapted from pysisyphus

import numpy as np
import torch

from nets.prediction_utils import Z_TO_ATOM_SYMBOL

from hip.frequency_analysis import eckart_projection_notmw_torch, analyze_frequencies_torch

####################################################################################################################

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
    print(f"\n" + "="*50)
    print("Product of smallest eigenvalues from Equiformer model example:")
    print("="*50)
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
        
