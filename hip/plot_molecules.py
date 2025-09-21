import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDetermineBonds

# from rdkit.Chem.Draw.IPythonConsole import drawMol3D  # draws a single molecule in 3D using py3Dmol
from rdkit.Chem.rdmolops import GetFormalCharge


import io
import base64
from IPython.display import Image
import os
import py3Dmol
import webbrowser
import plotly.io as pio
import seaborn as sns

# pio.renderers.default = "browser"

from hip.ff_lmdb import LmdbDataset
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.align_unordered_mols import compute_rmsd
from ocpmodels.units import ELEMENT_TO_ATOMIC_NUMBER, ATOMIC_NUMBER_TO_ELEMENT

# # Set global font size for all plots
# plt.rcParams.update({
#     'font.size': 6,
#     'axes.titlesize': 8,
#     'axes.labelsize': 6,
#     'xtick.labelsize': 5,
#     'ytick.labelsize': 5,
#     'legend.fontsize': 6,
#     'figure.titlesize': 10
# })


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def clean_filename(filename, prefix=None, suffix=None):
    filename = "".join(
        [c if c.isalnum() or c == "_" else "_" for c in filename.lower()]
    )
    if prefix is not None:
        filename = f"{prefix}_{filename}"
    if suffix is not None:
        filename = f"{filename}.{suffix}"
    for _ in range(5):
        filename = filename.replace("__", "_")
    return filename


# https://github.com/BurgerAndreas/meta-sampler/blob/a64d9a3d44604f04ebafe20ebd0756f4a222f5d5/alanine_dipeptide/data/alanine-dipeptide-nowater.pdb#L4
def save_to_xyz(coords, atomic_numbers, plotfolder, filename, title="molecule"):
    """
    Save molecular coordinates to XYZ format file.

    Parameters:
    -----------
    coords : array-like
        Atomic coordinates with shape (n_atoms, 3)
    atomic_numbers : array-like
        Atomic numbers for each atom
    filename : str
        Output filename
    title : str
        Comment line for the XYZ file
    """
    # Convert to numpy arrays if needed
    coords = to_numpy(coords)
    atomic_numbers = to_numpy(atomic_numbers)

    # Basic validation
    if len(coords) != len(atomic_numbers):
        raise ValueError(
            f"Mismatch: {len(coords)} coordinates vs {len(atomic_numbers)} atomic numbers"
        )

    # Create XYZ block
    xyz_lines = [str(len(coords)), str(title)]

    for i, (x, y, z) in enumerate(coords):
        atomic_num = int(atomic_numbers[i])
        symbol = ATOMIC_NUMBER_TO_ELEMENT.get(atomic_num, f"X{atomic_num}")
        xyz_lines.append(f"{symbol} {float(x):.6f} {float(y):.6f} {float(z):.6f}")

    xyz_block = "\n".join(xyz_lines)

    filename = clean_filename(filename, "xyz", "xyz")
    filename = os.path.join(plotfolder, filename)
    with open(filename, "w") as f:
        f.write(xyz_block)
    print(f"Saved molecule to {filename}")


def save_trajectory_to_xyz(
    coords_traj, atomic_numbers, plotfolder, filename, title_prefix="frame"
):
    """
    Save molecular trajectory to multi-frame XYZ format file.

    Parameters:
    -----------
    coords_traj : array-like
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    atomic_numbers : array-like
        Atomic numbers for each atom with shape (n_atoms,)
    filename : str
        Output filename
    title_prefix : str
        Prefix for frame titles (will be: "title_prefix_0", "title_prefix_1", etc.)
    """
    # Convert to numpy arrays if needed
    coords_traj = to_numpy(coords_traj)
    atomic_numbers = to_numpy(atomic_numbers)

    # Basic validation
    n_frames, n_atoms = coords_traj.shape[:2]
    if len(atomic_numbers) != n_atoms:
        raise ValueError(
            f"Mismatch: {n_atoms} atoms vs {len(atomic_numbers)} atomic numbers"
        )

    filename = clean_filename(filename, "xyztraj", "xyz")
    filename = os.path.join(plotfolder, filename)
    with open(filename, "w") as f:
        for frame_idx, coords in enumerate(coords_traj):
            # Create title for this frame
            frame_title = f"{title_prefix}_{frame_idx}"

            # Write XYZ block for this frame
            f.write(f"{n_atoms}\n")
            f.write(f"{frame_title}\n")

            for i, (x, y, z) in enumerate(coords):
                atomic_num = int(atomic_numbers[i])
                symbol = ATOMIC_NUMBER_TO_ELEMENT.get(atomic_num, f"X{atomic_num}")
                f.write(f"{symbol} {float(x):.6f} {float(y):.6f} {float(z):.6f}\n")

    print(f"Saved trajectory to {filename}")


def plot_molecule_mpl(
    coords,
    atomic_numbers,
    title,
    plot_dir,
    bonds=None,
    forces=None,
    save=False,
    filename=None,
    fig=None,
):
    """
    Plot a 3D molecule from atomic coordinates.

    Parameters:
    -----------
    coords : torch.Tensor or np.ndarray
        Atomic positions with shape (n_atoms, 3)
    title : str
        Title for the plot and filename
    plot_dir : str
        Directory to save the plot
    atomic_numbers : list or np.ndarray, optional
        Atomic numbers for each atom (for coloring)
    bonds : list of tuples, optional
        List of (i, j) tuples representing bonds between atoms
    forces : torch.Tensor or np.ndarray, optional
        Forces on each atom with shape (n_atoms, 3)
    """
    # Convert to numpy if torch tensor
    coords = to_numpy(coords)

    # Create 3D plot with two subplots
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(20, 8), subplot_kw={"projection": "3d"}
        )
    else:
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

    # Define atomic colors using seaborn pastel palette
    pastel_palette = sns.color_palette("pastel", 10)
    atomic_colors = {
        1: pastel_palette[0],  # H - light blue
        6: pastel_palette[1],  # C - light orange
        7: pastel_palette[2],  # N - light green
        8: pastel_palette[3],  # O - light red
        9: pastel_palette[4],  # F - light purple
        15: pastel_palette[5],  # P - light brown
        16: pastel_palette[6],  # S - light pink
        17: pastel_palette[7],  # Cl - light gray
        35: pastel_palette[8],  # Br - light olive
        53: pastel_palette[9],  # I - light cyan
    }

    # Set colors and sizes based on atomic numbers
    if torch.is_tensor(atomic_numbers):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()
    colors = [atomic_colors.get(int(z), pastel_palette[9]) for z in atomic_numbers]
    # Ensure colors is always a 2D array/list, even for a single atom
    if len(colors) == 1:
        colors = [colors[0]]
    # Bigger circles - size based on atomic number (larger for heavier atoms)
    sizes = [max(300, int(z) * 30) for z in atomic_numbers]

    # Function to plot on a single axis
    def plot_on_axis(ax, view_azim):
        # Plot atoms
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=colors,
            s=sizes,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )

        # Add bonds if provided
        if bonds is not None:
            for i, j in bonds:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    [coords[i, 2], coords[j, 2]],
                    "k-",
                    alpha=0.6,
                    linewidth=1.5,
                    zorder=1,
                )
        else:
            # Infer bonds using RDKit if atomic numbers are provided
            try:
                # Create RDKit molecule from atomic coordinates and numbers
                mol = Chem.RWMol()

                # Add atoms to molecule
                for i, atomic_num in enumerate(atomic_numbers):
                    atom = Chem.Atom(int(atomic_num))
                    mol.AddAtom(atom)

                # Add conformer with 3D coordinates
                conf = Chem.Conformer(len(coords))
                for i, (x, y, z) in enumerate(coords):
                    conf.SetAtomPosition(i, (float(x), float(y), float(z)))
                mol.AddConformer(conf)

                # Use RDKit's bond perception
                rdDetermineBonds.DetermineBonds(mol, charge=0)

                # Extract bonds and plot them
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    ax.plot(
                        [coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        "k-",
                        alpha=0.6,
                        linewidth=1.5,
                        zorder=1,
                    )

            except Exception as e:
                # print(
                #     f"Warning: RDKit bond inference failed ({e}), falling back to distance-based method"
                # )
                # Fallback to distance-based method
                n_atoms = len(coords)
                bond_threshold = 2.0  # Angstroms

                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        dist = np.linalg.norm(coords[i] - coords[j])
                        if dist < bond_threshold:
                            ax.plot(
                                [coords[i, 0], coords[j, 0]],
                                [coords[i, 1], coords[j, 1]],
                                [coords[i, 2], coords[j, 2]],
                                "k-",
                                alpha=0.4,
                                linewidth=1.0,
                                zorder=1,
                            )

        # Add atom labels
        for i, (x, y, z) in enumerate(coords):
            if atomic_numbers is not None:
                element_symbol = ATOMIC_NUMBER_TO_ELEMENT[int(atomic_numbers[i])]
                ax.text(
                    x,
                    y,
                    z,
                    f"{element_symbol}{i}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=10,
                )
            else:
                ax.text(
                    x, y, z, f"{i}", fontsize=8, ha="center", va="center", zorder=10
                )

        # Add force vectors as arrows if provided
        if forces is not None:
            forces_np = to_numpy(forces)
            for i, (coord, force) in enumerate(zip(coords, forces_np)):
                start_pos = coord
                end_pos = coord + force
                ax.quiver(
                    start_pos[0],
                    start_pos[1],
                    start_pos[2],
                    end_pos[0],
                    end_pos[1],
                    end_pos[2],
                    length=0.1,
                    # normalize=True,
                    # arrow_length_ratio=0.1,
                    color="gray",
                    zorder=8,
                )

        # Set labels
        fontsize = 9
        ax.set_xlabel("X (Å)", fontsize=fontsize)
        ax.set_ylabel("Y (Å)", fontsize=fontsize)
        ax.set_zlabel("Z (Å)", fontsize=fontsize)

        # Make axes equal
        max_range = (
            np.array(
                [
                    coords[:, 0].max() - coords[:, 0].min(),
                    coords[:, 1].max() - coords[:, 1].min(),
                    coords[:, 2].max() - coords[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
        mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
        mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Set the view angle
        ax.view_init(elev=30, azim=view_azim)

    # Plot on both axes with different view angles
    plot_on_axis(ax1, 45)  # Original view
    plot_on_axis(ax2, 135)  # 90 degrees rotated around z-axis

    # Add single global title
    fig.suptitle(f"{title}", fontsize=16, fontweight="bold", y=0.95)

    # Save the plot
    if save:
        plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce space between subplots
        plt.tight_layout(pad=0.0)  # Adjust for title
        if filename is None:
            filename = clean_filename(title, "mol", "png")
            filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()  # Close to free memory

        print(f"Saved structure to\n {filepath}")
    else:
        plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Reduce space between subplots
        plt.tight_layout(pad=0.0)  # Adjust for title
        return fig


def plot_traj_mpl(
    coords_traj,
    title,
    plot_dir,
    atomic_numbers,
    bonds_start=None,
    bonds_end=None,
    forces_traj=None,
    save=False,
    filename=None,
    fig=None,
    plot_forces_every=10,
    plot_dots_every=1,
    plot_atoms=True,
    plot_bond_traj=False,
    plot_atom_traj=True,
):
    """
    Plot a 3D trajectory from atomic coordinates.
    Plots the last frame of the trajectory.
    Every frame is plotted with small dots and a line connecting them.
    Every `plot_forces_every` frames the forces are plotted as arrows.

    Parameters:
    -----------
    coords_traj : torch.Tensor or np.ndarray
        Atomic positions with shape (T, n_atoms, 3)
    title : str
        Title for the plot and filename
    plot_dir : str
        Directory to save the plot
    atomic_numbers : list or np.ndarray, optional
        Atomic numbers for each atom (for coloring)
    bonds : list of tuples, optional
        List of (i, j) tuples representing bonds between atoms
    forces_traj : torch.Tensor or np.ndarray, optional
        Forces on each atom with shape (T, n_atoms, 3)
    """
    ########################################
    # plot final frame

    # Convert to numpy if torch tensor
    coords = to_numpy(coords_traj[-1])

    # Create 3D plot
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Define atomic colors using seaborn pastel palette
    pastel_palette = sns.color_palette("pastel", 10)
    atomic_colors = {
        1: pastel_palette[0],  # H - light blue
        6: pastel_palette[1],  # C - light orange
        7: pastel_palette[2],  # N - light green
        8: pastel_palette[3],  # O - light red
        9: pastel_palette[4],  # F - light purple
        15: pastel_palette[5],  # P - light brown
        16: pastel_palette[6],  # S - light pink
        17: pastel_palette[7],  # Cl - light gray
        35: pastel_palette[8],  # Br - light olive
        53: pastel_palette[9],  # I - light cyan
    }

    # Set colors and sizes based on atomic numbers
    if torch.is_tensor(atomic_numbers):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()
    colors = [atomic_colors.get(int(z), pastel_palette[9]) for z in atomic_numbers]
    # Ensure colors is always a 2D array/list, even for a single atom
    if len(colors) == 1:
        colors = [colors[0]]
    # Bigger circles - size based on atomic number (larger for heavier atoms)
    sizes = [max(300, int(z) * 30) for z in atomic_numbers]

    # Plot atoms
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        s=sizes,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )

    # Add bonds if provided
    bonds = bonds_end
    if bonds is not None:
        for i, j in bonds:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]],
                "k-",
                alpha=0.6,
                linewidth=1.5,
                zorder=1,
            )
    else:
        # Infer bonds using RDKit if atomic numbers are provided
        try:
            # Create RDKit molecule from atomic coordinates and numbers
            mol = Chem.RWMol()

            # Add atoms to molecule
            for i, atomic_num in enumerate(atomic_numbers):
                atom = Chem.Atom(int(atomic_num))
                mol.AddAtom(atom)

            # Add conformer with 3D coordinates
            conf = Chem.Conformer(len(coords))
            for i, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            mol.AddConformer(conf)

            # Use RDKit's bond perception
            rdDetermineBonds.DetermineBonds(mol, charge=0)

            bonds = mol.GetBonds()
            for bond in bonds:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    [coords[i, 2], coords[j, 2]],
                    "k-",
                    alpha=0.6,
                    linewidth=1.5,
                    zorder=1,
                )

        except Exception as e:
            # print(
            #     f"Warning: RDKit bond inference failed ({e}), falling back to distance-based method"
            # )
            # Fallback to distance-based method
            n_atoms = len(coords)
            bond_threshold = 2.0  # Angstroms

            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist < bond_threshold:
                        ax.plot(
                            [coords[i, 0], coords[j, 0]],
                            [coords[i, 1], coords[j, 1]],
                            [coords[i, 2], coords[j, 2]],
                            "k-",
                            alpha=0.4,
                            linewidth=1.0,
                            zorder=1,
                        )

    # Add atom labels
    if plot_atoms:
        for i, (x, y, z) in enumerate(coords):
            if atomic_numbers is not None:
                element_symbol = ATOMIC_NUMBER_TO_ELEMENT[int(atomic_numbers[i])]
                ax.text(
                    x,
                    y,
                    z,
                    f"{element_symbol}{i}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=10,
                )
            else:
                ax.text(
                    x, y, z, f"{i}", fontsize=8, ha="center", va="center", zorder=10
                )

    # color gradient over time
    colors_traj = sns.color_palette("viridis", len(coords_traj))

    # plot trajectories of atoms using small dots and lines
    smalldot_size = 10
    max_dots = 100
    plot_dots_every = min(plot_dots_every, len(coords_traj) // max_dots)
    plot_dots_every = max(plot_dots_every, 1)
    for i in range(len(coords_traj)):
        if (i % plot_dots_every) == 0 or (i == 0) or (i == len(coords_traj) - 1):
            coords = to_numpy(coords_traj[i])
            # Patch: ensure colors_traj[i] is a list, not a tuple
            color = colors_traj[i]
            if not isinstance(color, (list, tuple)) or (
                isinstance(color, tuple) and len(color) == 3
            ):
                color = [color]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=[colors_traj[i]] * coords.shape[0],
                s=smalldot_size,
                alpha=0.8,
                # edgecolors='black',
                linewidth=0.5,
                zorder=2,
            )
            if plot_bond_traj:
                # plots lines between atoms = bonds
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    c=colors_traj[i],
                    alpha=0.6,
                    linewidth=1.5,
                    zorder=1,
                )
            if plot_atom_traj:
                if i > 0:
                    # plots lines between atoms across timesteps
                    coord_prev = to_numpy(coords_traj[i - 1])
                    for _atom in range(len(coords)):
                        ax.plot(
                            [coord_prev[_atom, 0], coords[_atom, 0]],
                            [coord_prev[_atom, 1], coords[_atom, 1]],
                            [coord_prev[_atom, 2], coords[_atom, 2]],
                            c=colors_traj[i],
                            alpha=0.6,
                            linewidth=1.5,
                            zorder=1,
                        )

    # Add force vectors as arrows if provided
    if forces_traj is not None:
        max_forces = 5
        plot_forces_every = min(plot_forces_every, len(forces_traj) // max_forces)
        plot_forces_every = max(plot_forces_every, 1)
        for i in range(len(forces_traj)):
            if (i % plot_forces_every) == 0 or (i == 0) or (i == len(forces_traj) - 1):
                forces = to_numpy(forces_traj[i])
                coords = to_numpy(coords_traj[i])
                for _, (coord, force) in enumerate(zip(coords, forces)):
                    start_pos = coord
                    end_pos = coord + force
                    ax.quiver(
                        start_pos[0],
                        start_pos[1],
                        start_pos[2],
                        end_pos[0],
                        end_pos[1],
                        end_pos[2],
                        length=0.1,
                        # normalize=True,
                        # arrow_length_ratio=0.1,
                        color=colors_traj[i],
                        zorder=8,
                    )

    # Set labels and title
    fontsize = 9
    ax.set_xlabel("X (Å)", fontsize=fontsize)
    ax.set_ylabel("Y (Å)", fontsize=fontsize)
    ax.set_zlabel("Z (Å)", fontsize=fontsize)
    ax.set_title(f"{title}", fontsize=fontsize)

    # Make axes equal - calculate bounds over entire trajectory
    coords_traj_np = to_numpy(coords_traj)  # Convert entire trajectory to numpy

    # Get global min/max over all timesteps and atoms
    global_x_min, global_x_max = (
        coords_traj_np[:, :, 0].min(),
        coords_traj_np[:, :, 0].max(),
    )
    global_y_min, global_y_max = (
        coords_traj_np[:, :, 1].min(),
        coords_traj_np[:, :, 1].max(),
    )
    global_z_min, global_z_max = (
        coords_traj_np[:, :, 2].min(),
        coords_traj_np[:, :, 2].max(),
    )

    max_range = (
        np.array(
            [
                global_x_max - global_x_min,
                global_y_max - global_y_min,
                global_z_max - global_z_min,
            ]
        ).max()
        / 2.0
    )

    max_range = max_range * 1.1  # add some padding

    mid_x = (global_x_max + global_x_min) * 0.5
    mid_y = (global_y_max + global_y_min) * 0.5
    mid_z = (global_z_max + global_z_min) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Improve the view
    ax.view_init(elev=30, azim=45)

    # Save the plot
    if save:
        plt.tight_layout(pad=0.1)
        if filename is None:
            filename = clean_filename(title, "traj", "png")
            filepath = os.path.join(plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()  # Close to free memory

        print(f"Saved trajectory to\n {filepath}")
    else:
        return fig


def plot_molecule_py3dmol(
    coords,
    atomic_numbers=None,
    smiles: str = None,
    title: str = "Molecule",
    save_path: str = None,
    save_png: bool = False,
    save_html: bool = False,
    forces=None,
):
    """Render an interactive 3-D view of a molecule using py3Dmol.

    Parameters
    ----------
    coords : torch.Tensor | np.ndarray
        Cartesian coordinates with shape (N, 3) corresponding to the atom
        order in *smiles*.
    smiles : str
        SMILES string describing the molecular graph.
    title : str, optional
        Title that will appear in the viewer.
    atomic_numbers : torch.Tensor | np.ndarray, optional
        Atomic numbers for each atom. If provided, will be used instead of
        trying to parse from SMILES.

    Returns
    -------
    py3Dmol.view
        A configured viewer instance that can be shown inside a notebook or
        whose HTML representation can be written to disk.
    """

    # Ensure NumPy array input
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    if atomic_numbers is not None and isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.detach().cpu().numpy()

    # Map atomic numbers to element symbols
    atomic_number_to_symbol = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
    }
    # https://github.com/BurgerAndreas/meta-sampler/blob/a64d9a3d44604f04ebafe20ebd0756f4a222f5d5/alanine_dipeptide/data/alanine-dipeptide-nowater.pdb#L4
    # Create XYZ block
    xyz_lines = [str(len(coords)), title]

    for i, (x, y, z) in enumerate(coords):
        atomic_num = int(atomic_numbers[i])
        symbol = atomic_number_to_symbol.get(atomic_num, f"X{atomic_num}")
        xyz_lines.append(f"{symbol} {float(x):.6f} {float(y):.6f} {float(z):.6f}")

    xyz_block = "\n".join(xyz_lines)

    # Configure viewer with XYZ format
    view = py3Dmol.view(width=400, height=400)
    view.addModel(xyz_block, "xyz")
    view.setStyle({"stick": {}})
    view.setTitle(title)
    view.zoomTo()

    # Add force vectors as arrows if provided
    if forces is not None:
        if isinstance(forces, torch.Tensor):
            forces = forces.detach().cpu().numpy()

        # Scale factor for force visualization (adjust as needed)
        force_scale = 1.0

        for i, (coord, force) in enumerate(zip(coords, forces)):
            start_pos = coord
            end_pos = coord + force * force_scale

            # Add arrow from atom position in direction of force
            view.addArrow(
                {
                    "start": {
                        "x": float(start_pos[0]),
                        "y": float(start_pos[1]),
                        "z": float(start_pos[2]),
                    },
                    "end": {
                        "x": float(end_pos[0]),
                        "y": float(end_pos[1]),
                        "z": float(end_pos[2]),
                    },
                    "radius": 0.1,
                    "color": "red",
                }
            )

    # # Build RDKit molecule (no additional hydrogens - assume coords already
    # # contain all atoms that appear in the SMILES string).
    # mol = Chem.MolFromSmiles(smiles)
    # # Original approach when atom counts match
    # conf = Chem.Conformer(mol.GetNumAtoms())

    # for idx, (x, y, z) in enumerate(coords):
    #     conf.SetAtomPosition(idx, (float(x), float(y), float(z)))

    # mol.AddConformer(conf, assignId=True)

    # # Convert to PDB (3-D) format understood by 3Dmol.js
    # pdb_block = Chem.MolToPDBBlock(mol)

    # # Configure viewer
    # view = py3Dmol.view(width=400, height=400)
    # view.addModel(pdb_block, "pdb")
    # view.setStyle({"stick": {}})
    # view.setTitle(title)
    # view.zoomTo()

    if save_path is not None:
        if save_html:
            # Write an HTML file so it can be opened in any browser
            save_path = clean_filename(save_path, "py3dmol", "html")
            with open(save_path, "w") as fh:
                fh.write(view._make_html())
            print(f"Saved interactive py3Dmol viewer to {save_path}")
            # Open the HTML file in the default browser
            # webbrowser.open(save_path)

        if save_png:
            # save as png
            fname = clean_filename(save_path, "py3dmol", "png")
            view.saveImage(fname)
            print(f"Saved screenshot to {fname}")

    return view
