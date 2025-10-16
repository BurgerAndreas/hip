import torch
from typing import Optional


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


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


###################################################################################################
# postselect way to compute Hessian, but no lists and fewer loops
###################################################################################################


# TODO: what is the difference between this and the one in training_module.py?
def compute_full_hessian(coords, energy, forces=None, with_gradients=False):
    """Compute Hessian matrix using autograd.
    This hessian has unnecessary elements! Should be block diagonal, but is not.
    """
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, retain_graph=True, create_graph=True)

    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]

    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        # if not with_gradients: # very slow, but less memory usage
        #     hess_row = hess_row.detach()
        hess.append(hess_row)

    # Stack hessian
    hessian = torch.stack(hess).detach()  # [N*3, N*3]
    return hessian.reshape(n_comp, -1)


def get_smallest_eigen_from_full_hessian(
    batch, hessian, n_smallest=2
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return n_smallest eigenvalues and eigenvectors per batch.
    Does not have gradients, cannot be used for training.
    Returns:
        lists of tensors of shape [B, n_smallest] and [B, n_smallest, n_atoms*3]
    """

    # Compute Hessian and eigenspectrum for each batch separately
    B = batch.batch.max().item() + 1
    smallest_eigenvals = []
    smallest_eigenvecs = []
    atom_start = 0

    # assert B == 1, "Batch size must be 1 for this function"

    # loop over batches
    for mol_idx in range(B):
        # current batch start index
        n_atoms = batch.natoms[mol_idx].item()
        atom_end = atom_start + n_atoms
        coord_start = atom_start * 3
        coord_end = atom_end * 3

        hessian_this_batch = hessian[
            coord_start:coord_end, coord_start:coord_end
        ]  # [n_atoms*3, n_atoms*3]

        # Get eigenvalues [n_atoms*3] and eigenvectors [n_atoms*3, n_atoms*3]
        # The eigenvalues are returned in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian_this_batch)
        smallest_eigenvals.append(eigenvalues[:n_smallest])
        smallest_eigenvecs.append(eigenvectors[:, :n_smallest])

        atom_start = atom_end

    assert len(smallest_eigenvals) == B, (
        f"Number of eigenvalues does not match number of batches: {len(smallest_eigenvals)} != {B}"
    )

    return smallest_eigenvals, smallest_eigenvecs


###################################################################################################
# batchwise way to compute Hessian, but requires lists
###################################################################################################


def compute_hessian_batches(batch, coords, energy, forces=None) -> list[torch.Tensor]:
    """Compute Hessian matrix using autograd."""
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, retain_graph=True, create_graph=True)

    B = batch.batch.max().item() + 1
    hessians = []  # one Hessian per batch

    # Initialize hessian
    hessian_this_batch = []
    mol_idx = 0
    atom_idx = 0
    atoms_in_batch = batch.natoms[mol_idx].item()
    atoms_start = 0
    atoms_end = atoms_start + atoms_in_batch
    coord_start = 0
    coord_end = atoms_in_batch * 3
    for f in forces.reshape(-1):  # one atom at a time
        # Compute second-order derivative
        # postselect, since we only need the forces and coords for the current batch
        # but we cannot index the forces and coords without breaking the autograd graph
        hess_row = _get_derivatives(coords, -f, retain_graph=True, create_graph=False)
        # hess_row = hess_row.reshape(-1) # [n_atoms_total, 3] -> [n_atoms_total*3]
        hessian_this_batch.append(
            hess_row[atoms_start:atoms_end]
        )  # [n_atoms_in_batch, 3]
        atom_idx += 1
        if atom_idx == atoms_in_batch * 3:
            # new batch
            # [n_atoms_in_batch*3, n_atoms_in_batch, 3]
            hessian_this_batch = torch.stack(hessian_this_batch)
            hessian_this_batch = hessian_this_batch.reshape(
                atoms_in_batch * 3, atoms_in_batch * 3
            )
            hessians.append(hessian_this_batch.detach())
            # reset
            hessian_this_batch = []
            atom_idx = 0
            mol_idx += 1
            if mol_idx == B:
                break
            atoms_in_batch = batch.natoms[mol_idx].item()
            atoms_start = atoms_end
            atoms_end = atoms_start + atoms_in_batch
            coord_start = coord_end
            coord_end = coord_start + atoms_in_batch * 3

    assert len(hessians) == B, (
        f"Number of hessians does not match number of batches: {len(hessians)} != {B}"
    )

    # Stack hessian
    return hessians


def get_smallest_eigen_from_batched_hessians(
    batch, hessians, n_smallest=2
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Get the smallest eigenvalues and eigenvectors from a list of batched hessians."""
    # B = batch.batch.max().item() + 1
    smallest_eigenvals = []
    smallest_eigenvecs = []
    if isinstance(hessians, torch.Tensor):
        hessians = [hessians]
    for hessian in hessians:
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        smallest_eigenvals.append(eigenvalues[:n_smallest])
        smallest_eigenvecs.append(eigenvectors[:, :n_smallest])
    return smallest_eigenvals, smallest_eigenvecs


###################################################################################################
# Compute a single Hessian in parallel using vmap
###################################################################################################


# https://github.com/ACEsuit/mace/blob/d39cc6b5f0f416dbc5eb3462f67544592130076e/mace/modules/utils.py#L111
@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
    chunk_size=None,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            outputs=-1 * forces_flatten,
            inputs=positions,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        gradient = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(
            I_N
        )[0].detach()
    except RuntimeError as e:
        print(f"compute_hessians_vmap: {e}")
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        # hess_row = hess_row.detach()  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian.detach()


def compute_hessian_single_batch(batch, coords, energy, forces) -> torch.Tensor:
    """Compute Hessian matrix using autograd.
    For the Hessian (second derivatives), we need to compute
    the gradient of each element of forces with respect to each element of coords
    """

    # B = batch.batch.max().item() + 1
    # assert B == 1, "Batch size must be 1 for this function"

    # 3D coordinates -> 3N^2 Hessian elements
    N = coords.shape[0]

    forces = forces.reshape(-1)
    # hessian = compute_hessians_vmap(forces, coords) # [N*3, N, 3]

    num_elements = forces.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            outputs=-1 * forces,
            inputs=coords,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements, device=forces.device)
    hessian = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=None)(I_N)[0]
    hessian = hessian.view(N * 3, N * 3)

    return hessian


def predict_eigen_from_batch(
    batch, model, n_smallest=2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict eigenvalues and eigenvectors from a batch.
    Same as model.potential.forward, compute_hessian_single_batch, get_smallest_eigen_from_batched_hessians.
    """
    energy, forces, out = model.potential.forward(batch)

    forces = forces.reshape(-1)
    num_elements = forces.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            outputs=-1 * forces,
            inputs=batch.pos,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements, device=forces.device)
    hessian = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=None)(I_N)[0]

    # 3D coordinates -> 3N^2 Hessian elements
    N = batch.pos.shape[0]
    hessian = hessian.view(N * 3, N * 3)

    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    smallest_eigenvals = eigenvalues[:n_smallest]
    smallest_eigenvecs = eigenvectors[:, :n_smallest]
    return smallest_eigenvals, smallest_eigenvecs


###################################################################################################
# Test
###################################################################################################
def test_hessian_utils():
    import os
    import time
    from torch_geometric.loader import DataLoader as TGDataLoader
    from hip.path_config import find_project_root
    from hip.training_module import PotentialModule
    from hip.ff_lmdb import LmdbDataset

    batch_size = 2

    # Paths
    root_dir = find_project_root()
    print(f"Root directory: {root_dir}")
    checkpoint_path = os.path.join(root_dir, "ckpt/eqv2.ckpt")

    # Try different dataset paths in order of preference
    dataset_paths = [
        "data/sample_100.lmdb",  # Local small dataset for testing
        os.path.expanduser(
            "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x-val.lmdb"
        ),
        os.path.expanduser(
            "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/RGD1.lmdb"
        ),
    ]

    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        print("No dataset found! Please check dataset paths.")
        return

    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint to get model info
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    print(f"Model type: {model_name}")

    # Load the full model
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")

    print(f"Loading dataset from: {dataset_path}")
    dataset = LmdbDataset(dataset_path)
    dataloader = TGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} samples")

    print("\n" + "=" * 50)
    print(f"Comparing Predictions (batch_size={batch_size})")
    print("=" * 50)

    # Predict on a few batches
    processed_batches = 0
    for i, batch in enumerate(dataloader):
        # keys: ['pos', 'charges', 'hessian', 'batch', 'forces', 'natoms', 'one_hot', 'ptr', 'rxn', 'energy', 'ae']
        if batch.pos.shape[0] > 25:
            # so that we don't run out of memory on my RTX 3060
            continue
        # Only process first 3 batches
        processed_batches += 1
        if processed_batches > 3:
            break

        # Move batch to device
        batch = batch.to(device)

        #############################################################
        # compute Hessian eigenvalues and eigenvectors in postselect way

        batch.pos.requires_grad_(True)

        # Forward pass to get energy and forces
        model_name = model.model_config["name"]

        if model_name == "LEFTNet":
            energy, forces = model.potential.forward_autograd(batch)
        else:
            energy, forces, out = model.potential.forward(batch)

        full_hessian = compute_full_hessian(
            coords=batch.pos, energy=energy, forces=forces
        )
        smallest_eigenvals_full, smallest_eigenvecs_full = (
            get_smallest_eigen_from_full_hessian(batch, full_hessian, n_smallest=2)
        )

        #############################################################
        # compute Hessian eigenvalues and eigenvectors in batchwise way

        hessians = compute_hessian_batches(batch, batch.pos, energy, forces)
        smallest_eigenvals_batched, smallest_eigenvecs_batched = (
            get_smallest_eigen_from_batched_hessians(batch, hessians, n_smallest=2)
        )

        #############################################################
        # compare results
        print(f"\nBatch {i}")
        for b in range(batch_size):
            print(f"Element {b} (full vs batched)")
            max_diff_eigenval = torch.max(
                torch.abs(smallest_eigenvals_full[b] - smallest_eigenvals_batched[b])
            )
            max_diff_eigenvec = torch.max(
                torch.abs(smallest_eigenvecs_full[b] - smallest_eigenvecs_batched[b])
            )
            print(f" Max diff in eigenvalues: {max_diff_eigenval}")
            print(f" Max diff in eigenvectors: {max_diff_eigenvec}")

    #############################################################################################
    # Compare with batch_size=1
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} samples")

    print("\n" + "=" * 50)
    print(f"Comparing Predictions (batch_size=1)")
    print("=" * 50)

    # Predict on a few batches
    processed_batches = 0
    for i, batch in enumerate(dataloader):
        # keys: ['pos', 'charges', 'hessian', 'batch', 'forces', 'natoms', 'one_hot', 'ptr', 'rxn', 'energy', 'ae']
        if batch.pos.shape[0] > 25:
            # so that we don't run out of memory on my RTX 3060
            continue
        # Only process first 3 batches
        processed_batches += 1
        if processed_batches > 3:
            break

        # Move batch to device
        batch = batch.to(device)
        

        #############################################################
        # compute Hessian eigenvalues and eigenvectors in postselect way

        batch.pos.requires_grad_(True)

        # Forward pass to get energy and forces
        model_name = model.model_config["name"]

        if model_name == "LEFTNet":
            energy, forces = model.potential.forward_autograd(batch)
        else:
            energy, forces, out = model.potential.forward(batch)

        full_hessian = compute_full_hessian(
            coords=batch.pos, energy=energy, forces=forces
        )
        smallest_eigenvals_full, smallest_eigenvecs_full = (
            get_smallest_eigen_from_full_hessian(batch, full_hessian, n_smallest=2)
        )

        #############################################################
        # compute Hessian eigenvalues and eigenvectors in batchwise way

        hessians = compute_hessian_batches(batch, batch.pos, energy, forces)
        smallest_eigenvals_batched, smallest_eigenvecs_batched = (
            get_smallest_eigen_from_batched_hessians(batch, hessians, n_smallest=2)
        )

        #############################################################
        # Compute Hessian in parallel and compare

        hessian_single = compute_hessian_single_batch(batch, batch.pos, energy, forces)
        smallest_eigenvals_single, smallest_eigenvecs_single = (
            get_smallest_eigen_from_batched_hessians(
                batch, hessian_single, n_smallest=2
            )
        )

        # Compare with the first element from the batched computation
        max_diff_eigenval = torch.max(
            torch.abs(smallest_eigenvals_single[0] - smallest_eigenvals_batched[0])
        )
        max_diff_eigenvec = torch.max(
            torch.abs(smallest_eigenvecs_single[0] - smallest_eigenvecs_batched[0])
        )
        print(f"Single vs batched")
        print(f" Max diff in eigenvalues: {max_diff_eigenval}")
        print(f" Max diff in eigenvectors: {max_diff_eigenvec}")

    #############################################################################################
    # now time the methods using 100 batches
    print("\n" + "=" * 50)
    print("Timing the methods")
    print("=" * 50)

    max_batches = 20

    # dataset = dataset[:100]
    dataloader = TGDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i > max_batches:
            break
        batch = batch.to(device)
        
        # batch.pos.requires_grad_(True)
        energy, forces, out = model.potential.forward(batch)
    t1 = time.time()
    print(f"Time taken for just energy/force prediction: {t1 - t0:.2f} seconds")

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i > max_batches:
            break
        batch = batch.to(device)
        
        # batch.pos.requires_grad_(True)
        energy, forces, out = model.potential.forward(batch)
        # compute Hessian in postselect way
        full_hessian = compute_full_hessian(
            coords=batch.pos, energy=energy, forces=forces
        )
        seigvals, seigvecs = get_smallest_eigen_from_full_hessian(
            batch, full_hessian, n_smallest=2
        )
    t1 = time.time()
    print(f"Time taken for postselect method: {t1 - t0:.2f} seconds")

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i > max_batches:
            break
        batch = batch.to(device)
        
        # batch.pos.requires_grad_(True)
        energy, forces, out = model.potential.forward(batch)
        # compute Hessian in batchwise way
        hessians = compute_hessian_batches(batch, batch.pos, energy, forces)
        seigvals, seigvecs = get_smallest_eigen_from_batched_hessians(
            batch, hessians, n_smallest=2
        )
    t1 = time.time()
    print(f"Time taken for batchwise method: {t1 - t0:.2f} seconds")

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i > max_batches:
            break
        batch = batch.to(device)
        
        # batch.pos.requires_grad_(True)
        energy, forces, out = model.potential.forward(batch)
        # compute Hessian in parallel
        hessian_single = compute_hessian_single_batch(batch, batch.pos, energy, forces)
        seigvals, seigvecs = get_smallest_eigen_from_batched_hessians(
            batch, hessian_single, n_smallest=2
        )
    t1 = time.time()
    print(f"Time taken for parallel method: {t1 - t0:.2f} seconds")

    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i > max_batches:
            break
        batch = batch.to(device)
        
        seigvals, seigvecs = predict_eigen_from_batch(batch, model)
    t1 = time.time()
    print(f"Time taken for predict_eigen_from_batch: {t1 - t0:.2f} seconds")

    batch = next(iter(dataloader))
    batch = batch.to(device)
    
    # batch.pos.requires_grad_(True)
    energy, forces, out = model.potential.forward(batch)
    # compute Hessian in parallel
    hessian_single = compute_hessian_single_batch(batch, batch.pos, energy, forces)
    t0 = time.time()
    for i, batch in enumerate(dataloader):
        if i > max_batches:
            break
        seigvals, seigvecs = get_smallest_eigen_from_batched_hessians(
            batch, hessian_single, n_smallest=2
        )
    t1 = time.time()
    print(f"Time taken for just eigen computation: {t1 - t0:.2f} seconds")

    #############################################################################################
    # Test the maximum batch size that can be processed without running out of memory
    print("\n" + "=" * 50)
    print("Testing maximum batch size")
    print("=" * 50)

    max_batch_size = None
    for batch_size in range(1, 100):
        try:
            dataloader = TGDataLoader(dataset, batch_size=batch_size, shuffle=False)
            for i, batch in enumerate(dataloader):
                batch = batch.to(device)
                
                # batch.pos.requires_grad_(True)
                energy, forces, out = model.potential.forward(batch)
                # Try to compute Hessian in the batchwise way
                hessians = compute_hessian_batches(batch, batch.pos, energy, forces)
                break  # Only need to test one batch per batch_size
            max_batch_size = batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory at batch size {batch_size}")
                break
            else:
                raise
    if max_batch_size is not None:
        print(
            f"Maximum batch size that can be processed using batchwise method: {max_batch_size}"
        )
    else:
        print("Could not process any batch size without running out of memory.")

    # Test the maximum batch size that can be processed without running out of memory (postselect method)
    max_batch_size_postselect = None
    for batch_size in range(1, 100):
        try:
            dataloader = TGDataLoader(dataset, batch_size=batch_size, shuffle=False)
            for i, batch in enumerate(dataloader):
                batch = batch.to(device)
                
                energy, forces, out = model.potential.forward(batch)
                # Try to compute Hessian in the postselect way
                full_hessian = compute_full_hessian(
                    coords=batch.pos, energy=energy, forces=forces
                )
                if i > 100:
                    break
                break  # Only need to test one batch per batch_size
            max_batch_size_postselect = batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory at batch size {batch_size} (postselect method)")
                break
            else:
                raise
    if max_batch_size_postselect is not None:
        print(
            f"Maximum batch size that can be processed using postselect method: {max_batch_size_postselect}"
        )
    else:
        print(
            "Could not process any batch size without running out of memory (postselect method)."
        )
        max_batch_size_postselect = 1


if __name__ == "__main__":
    test_hessian_utils()
