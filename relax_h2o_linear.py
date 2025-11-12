"""
Script to relax H2O from a linear geometry and measure the bond angle.
Also relaxes from the correct H2O geometry for comparison.
"""

import os
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from hip.equiformer_ase_calculator import EquiformerASECalculator


def calculate_bond_angle(atoms, center_idx, atom1_idx, atom2_idx):
    """
    Calculate the bond angle between three atoms.

    Args:
        atoms: ASE Atoms object
        center_idx: Index of the central atom (O in H2O)
        atom1_idx: Index of the first atom (H1)
        atom2_idx: Index of the second atom (H2)

    Returns:
        angle: Bond angle in degrees
    """
    pos = atoms.get_positions()

    # Vectors from center to each atom
    vec1 = pos[atom1_idx] - pos[center_idx]
    vec2 = pos[atom2_idx] - pos[center_idx]

    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    # Calculate angle using dot product
    cos_angle = np.dot(vec1_norm, vec2_norm)
    # Clamp to [-1, 1] to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def create_correct_h2o_geometry():
    """
    Create H2O with correct experimental geometry:
    - O-H bond length: ~0.96 Å
    - H-O-H bond angle: ~104.5°
    """
    # O at origin
    # H atoms in xz plane with correct angle
    oh_length = 0.96  # Å
    bond_angle = 104.5  # degrees

    # Convert angle to radians
    half_angle_rad = np.radians(bond_angle / 2)

    # Place H atoms symmetrically around O in xz plane
    h1_x = oh_length * np.sin(half_angle_rad)
    h1_z = oh_length * np.cos(half_angle_rad)
    h2_x = -oh_length * np.sin(half_angle_rad)
    h2_z = oh_length * np.cos(half_angle_rad)

    atoms = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O (center)
            [h1_x, 0.0, h1_z],  # H1
            [h2_x, 0.0, h2_z],  # H2
        ],
    )

    return atoms


def relax_molecule(atoms, calculator, label, trajectory_name):
    """Relax a molecule and return results."""
    atoms.calc = calculator

    initial_angle = calculate_bond_angle(atoms, center_idx=0, atom1_idx=1, atom2_idx=2)
    initial_energy = calculator.get_potential_energy(atoms)

    print(f"\n{label}:")
    print(f"  Initial bond angle: {initial_angle:.2f}°")
    print(f"  Initial energy: {initial_energy:.6f} eV")

    # Relax the structure
    optimizer = BFGS(atoms, trajectory=trajectory_name, logfile=None)
    optimizer.run(fmax=0.01, steps=2000)

    final_energy = calculator.get_potential_energy(atoms)
    final_forces = calculator.get_forces(atoms)
    final_angle = calculate_bond_angle(atoms, center_idx=0, atom1_idx=1, atom2_idx=2)
    oh1_dist = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
    oh2_dist = np.linalg.norm(atoms.positions[2] - atoms.positions[0])

    print(f"  Relaxation converged in {optimizer.get_number_of_steps()} steps")
    print(f"  Final energy: {final_energy:.6f} eV")
    print(f"  Energy change: {final_energy - initial_energy:.6f} eV")
    print(f"  O-H1 bond length: {oh1_dist:.4f} Å")
    print(f"  O-H2 bond length: {oh2_dist:.4f} Å")
    print(f"  Final bond angle: {final_angle:.2f}°")
    print(f"  Angle change: {final_angle - initial_angle:.2f}°")
    print(f"  Max force: {np.max(np.abs(final_forces)):.6f} eV/Å")

    return {
        "initial_angle": initial_angle,
        "final_angle": final_angle,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "oh1_dist": oh1_dist,
        "oh2_dist": oh2_dist,
        "steps": optimizer.get_number_of_steps(),
    }


def main():
    # Initialize calculator
    project_root = os.path.dirname(__file__)
    checkpoint_path = os.path.join(project_root, "ckpt/hesspred_v1.ckpt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path")
        return

    calculator = EquiformerASECalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    print("=" * 60)
    print("H2O Relaxation Comparison")
    print("=" * 60)

    # 1. Relax from linear geometry
    atoms_linear = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O (center)
            [0.96, 0.0, 0.0],  # H1 (one side, typical O-H bond length ~0.96 Å)
            [-0.96, 0.0, 0.0],  # H2 (opposite side)
        ],
    )

    # Add small random noise to break perfect symmetry
    noise = np.random.normal(0, 0.02, size=atoms_linear.positions.shape)
    atoms_linear.positions += noise

    results_linear = relax_molecule(
        atoms_linear,
        calculator,
        "Relaxation from linear geometry (180°)",
        "h2o_relaxation_linear.traj",
    )

    # 2. Relax from correct geometry
    atoms_correct = create_correct_h2o_geometry()
    # Add small noise to the correct geometry too
    noise = np.random.normal(0, 0.02, size=atoms_correct.positions.shape)
    atoms_correct.positions += noise

    results_correct = relax_molecule(
        atoms_correct,
        calculator,
        "Relaxation from correct geometry (~104.5°)",
        "h2o_relaxation_correct.traj",
    )

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"Linear start -> Final angle: {results_linear['final_angle']:.2f}°")
    print(f"Correct start -> Final angle: {results_correct['final_angle']:.2f}°")
    print(
        f"Angle difference: {abs(results_linear['final_angle'] - results_correct['final_angle']):.2f}°"
    )
    print()
    print(f"Linear start -> Final energy: {results_linear['final_energy']:.6f} eV")
    print(f"Correct start -> Final energy: {results_correct['final_energy']:.6f} eV")
    print(
        f"Energy difference: {abs(results_linear['final_energy'] - results_correct['final_energy']):.6f} eV"
    )
    print()
    print(f"Linear start -> Steps: {results_linear['steps']}")
    print(f"Correct start -> Steps: {results_correct['steps']}")


if __name__ == "__main__":
    main()
