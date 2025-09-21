import torch


def reshape_to_batch_with_padding(energy, forces, hessian, batch, device):
    # Reshape to batch format
    B = batch.batch.max().item() + 1  # Number of molecules in batch
    max_atoms = max(batch.natoms)  # Max atoms in any molecule in this batch

    # Energy: reshape to [B, 1]
    energy_batched = energy.view(B, 1)

    # Forces: reshape from [total_atoms, 3] to [B, max_atoms, 3] with padding
    forces_batched = torch.zeros(B, max_atoms, 3, device=device, dtype=forces.dtype)

    # Hessian: reshape from [total_atoms*3, total_atoms*3] to [B, max_atoms, 3, max_atoms, 3] with padding
    hessian_batched = torch.zeros(
        B, max_atoms, 3, max_atoms, 3, device=device, dtype=hessian.dtype
    )

    # Atom masks: [B, max_atoms] indicating which atoms are real vs padding
    atom_masks = torch.zeros(B, max_atoms, device=device, dtype=torch.bool)

    # Fill in the batched tensors
    atom_start = 0
    for mol_idx in range(B):
        n_atoms = batch.natoms[mol_idx].item()
        atom_end = atom_start + n_atoms

        # Fill forces for this molecule
        forces_batched[mol_idx, :n_atoms] = forces[atom_start:atom_end]

        # Fill hessian for this molecule
        coord_start = atom_start * 3
        coord_end = atom_end * 3
        hess_mol = hessian[
            coord_start:coord_end, coord_start:coord_end
        ]  # [n_atoms*3, n_atoms*3]
        hess_mol_reshaped = hess_mol.view(
            n_atoms, 3, n_atoms, 3
        )  # [n_atoms, 3, n_atoms, 3]
        hessian_batched[mol_idx, :n_atoms, :, :n_atoms, :] = hess_mol_reshaped

        # Fill atom mask for this molecule
        atom_masks[mol_idx, :n_atoms] = True

        atom_start = atom_end
    return energy_batched, forces_batched, hessian_batched, atom_masks
