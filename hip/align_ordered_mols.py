import numpy as np
import torch


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Numpy array of shape (N,D) -- Point Cloud to Align (source)
        -    B: Numpy array of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = np.asarray([[1., 1.], [2., 2.], [1.5, 3.]])
        >>> R0 = np.asarray([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]])
        >>> B = (R0.dot(A.T)).T
        >>> t0 = np.array([3., 3.])
        >>> B += t0
        >>> B.shape
        (3, 2)
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.dot(A.T)).T + t
        >>> rmsd = np.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        2.5639502485114184e-16
        >>> B *= np.array([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.dot(A.T)).T + t
        >>> rmsd = np.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        2.5639502485114184e-16
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.dot(B_c)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    # Rotation matrix
    R = V.dot(U.T)
    # Ensure R is a proper rotation matrix
    if np.linalg.det(R) < 0:  # reflection
        V[:, -1] *= -1  # flip the sign of the last column of V
        R = V.dot(U.T)
    # Translation vector
    t = b_mean - R.dot(a_mean)
    return R, t


def get_rmsd(A, B):
    # A: (N, 3)
    # B: (N, 3)
    return np.sqrt((((A - B) ** 2).sum(axis=1)).mean())


def align_ordered_and_get_rmsd(A, B):
    """Get Root Mean Square Distance (RMSD) between two sets of coordinates.
    Only works if A and B have the atoms in the same order.
    Alignment is NOT mass weighted.
    A: (N, 3)
    B: (N, 3)
    """
    if A.shape != B.shape:
        # probably optimization diverged
        print(
            f"Error in get_rmsd: A and B must have the same shape, but got {A.shape} and {B.shape}"
        )
        return float("inf")
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R.dot(A.T)).T + t
    return float(get_rmsd(A_aligned, B))


if __name__ == "__main__":
    import doctest
    import argparse
    import pymol.cmd as cmd
    import os

    doctest.testmod()
    parser = argparse.ArgumentParser(
        description="Align 2 protein structures with same number of atoms"
    )
    parser.add_argument(
        "--pdb1", type=str, help="First (mobile) pdb file name", required=True
    )
    parser.add_argument(
        "--pdb2", type=str, help="Second (reference) pdb file name", required=True
    )
    parser.add_argument(
        "--select",
        type=str,
        help="Selection to perform the alignment on. Default: all atoms",
        required=False,
        default="all",
    )
    args = parser.parse_args()
    cmd.load(args.pdb1, "pdb1")
    cmd.load(args.pdb2, "pdb2")
    coords1 = cmd.get_coords(selection=f"pdb1 and {args.select}")
    coords2 = cmd.get_coords(selection=f"pdb2 and {args.select}")
    R, t = find_rigid_alignment(coords1, coords2)
    coords1 = cmd.get_coords(selection="pdb1")
    coords1_aligned = (R.dot(coords1.T)).T + t
    cmd.load_coords(coords1_aligned, "pdb1")
    rmsd = get_rmsd(
        cmd.get_coords(selection=f"pdb1 and {args.select}"),
        cmd.get_coords(selection=f"pdb2 and {args.select}"),
    )
    print("RMSD: %.4f" % rmsd)
    outname = os.path.splitext(args.pdb1)[0] + "_align.pdb"
    cmd.save(outname, selection="pdb1")
