#!/usr/bin/env python3
import argparse
import copy
import gzip
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Iterator, List, Optional, Protocol, Set, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial import distance_matrix  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

try:
    import qmllib  # type: ignore
    from qmllib.kernels import laplacian_kernel  # type: ignore
    from qmllib.representations import generate_fchl19  # type: ignore
except ImportError:
    qmllib = None


__intro__ = """
Calculate Root-mean-square deviation (RMSD) between structure A and B, in XYZ
or PDB format, using transformation and rotation.

From:
https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py

Please cite:
@misc{charnley_rmsd_1_6_4,
  author       = {Charnley, Daniel},
  title        = {Calculate Root-mean-square deviation (RMSD) of Two Molecules Using Rotation},
  year         = {2022},
  version      = {1.6.4},
  url          = {https://github.com/charnley/rmsd/tree/rmsd-1.6.4},
  note         = {GitHub repository, commit version 1.6.4}
}

Inertia hungarian reorder algorithm:
http://dx.doi.org/10.1109/TAES.2016.140952

Kabsch rotation algorithm:
http://dx.doi.org/10.1107/S0567739476001873

"""

__details__ = """

details:

  --rotation <method>
    Specifies the method for rotating molecules. Available options are:

    kabsch (Default): This is the default rotation method, used for aligning molecular structures using the Kabsch algorithm.

    quaternion: An alternative rotation method that uses quaternion mathematics for structure alignment.

  --reorder-method <method>
    Specifies the method for reordering atoms. Available options are:

    inertia-hungarian (Default): Use this method when structures are not aligned. The process involves: 1. Aligning the molecules based on their inertia moments. 2. Applying the Hungarian distance assignment (see below) for atom pairing.

    hungarian: Best used when the structures are already aligned. It assigns atom-atom pairs based on a linear-sum assignment of the distance combination.

    qml: Use this method when structures are not aligned and the inertia-hungarian method fails. It employs atomic structure descriptors and Hungarian cost-assignment to determine the best atom order.

    distance: This method assigns atom-atom pairs based on the sorted shortest distance between atoms.

    brute: A brute-force enumeration of all possible atom-atom pair combinations. This method is provided for reference only and has no practical use due to its computational cost.

For more information, usage, example and citation read more at
https://github.com/charnley/rmsd
"""

__version__ = "1.6.4"


METHOD_KABSCH = "kabsch"
METHOD_QUATERNION = "quaternion"
METHOD_NOROTATION = "none"
ROTATION_METHODS = [METHOD_KABSCH, METHOD_QUATERNION, METHOD_NOROTATION]

REORDER_NONE = "none"
REORDER_QML = "qml"
REORDER_HUNGARIAN = "hungarian"
REORDER_INERTIA_HUNGARIAN = "inertia-hungarian"
REORDER_BRUTE = "brute"
REORDER_DISTANCE = "distance"
REORDER_METHODS = [
    REORDER_NONE,
    REORDER_QML,
    REORDER_HUNGARIAN,
    REORDER_INERTIA_HUNGARIAN,
    REORDER_BRUTE,
    REORDER_DISTANCE,
]

FORMAT_XYZ = "xyz"
FORMAT_PDB = "pdb"
FORMATS = [FORMAT_XYZ, FORMAT_PDB]

AXIS_SWAPS = np.array(
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]
)

AXIS_REFLECTIONS = np.array(
    [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
    ]
)

ELEMENT_WEIGHTS = {
    1: 1.00797,
    2: 4.00260,
    3: 6.941,
    4: 9.01218,
    5: 10.81,
    6: 12.011,
    7: 14.0067,
    8: 15.9994,
    9: 18.998403,
    10: 20.179,
    11: 22.98977,
    12: 24.305,
    13: 26.98154,
    14: 28.0855,
    15: 30.97376,
    16: 32.06,
    17: 35.453,
    19: 39.0983,
    18: 39.948,
    20: 40.08,
    21: 44.9559,
    22: 47.90,
    23: 50.9415,
    24: 51.996,
    25: 54.9380,
    26: 55.847,
    28: 58.70,
    27: 58.9332,
    29: 63.546,
    30: 65.38,
    31: 69.72,
    32: 72.59,
    33: 74.9216,
    34: 78.96,
    35: 79.904,
    36: 83.80,
    37: 85.4678,
    38: 87.62,
    39: 88.9059,
    40: 91.22,
    41: 92.9064,
    42: 95.94,
    43: 98,
    44: 101.07,
    45: 102.9055,
    46: 106.4,
    47: 107.868,
    48: 112.41,
    49: 114.82,
    50: 118.69,
    51: 121.75,
    53: 126.9045,
    52: 127.60,
    54: 131.30,
    55: 132.9054,
    56: 137.33,
    57: 138.9055,
    58: 140.12,
    59: 140.9077,
    60: 144.24,
    61: 145,
    62: 150.4,
    63: 151.96,
    64: 157.25,
    65: 158.9254,
    66: 162.50,
    67: 164.9304,
    68: 167.26,
    69: 168.9342,
    70: 173.04,
    71: 174.967,
    72: 178.49,
    73: 180.9479,
    74: 183.85,
    75: 186.207,
    76: 190.2,
    77: 192.22,
    78: 195.09,
    79: 196.9665,
    80: 200.59,
    81: 204.37,
    82: 207.2,
    83: 208.9804,
    84: 209,
    85: 210,
    86: 222,
    87: 223,
    88: 226.0254,
    89: 227.0278,
    91: 231.0359,
    90: 232.0381,
    93: 237.0482,
    92: 238.029,
    94: 242,
    95: 243,
    97: 247,
    96: 247,
    102: 250,
    98: 251,
    99: 252,
    108: 255,
    109: 256,
    100: 257,
    101: 258,
    103: 260,
    104: 261,
    107: 262,
    105: 262,
    106: 263,
    110: 269,
    111: 272,
    112: 277,
}

ELEMENT_NAMES = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    114: "Uuq",
    116: "Uuh",
}

NAMES_ELEMENT = {value: key for key, value in ELEMENT_NAMES.items()}


class ReorderCallable(Protocol):
    def __call__(
        self,
        p_atoms: ndarray,
        q_atoms: ndarray,
        p_coord: ndarray,
        q_coord: ndarray,
        **kwargs: Any,
    ) -> ndarray:
        """
        Protocol for a reorder callable function

        Return:
            ndarray dtype=int  # Array of indices
        """
        ...  # pragma: no cover


class RmsdCallable(Protocol):
    def __call__(
        self,
        P: ndarray,
        Q: ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Protocol for a rotation callable function

        return:
            RMSD after rotation
        """
        ...  # pragma: no cover


def str_atom(atom: int) -> str:
    """
    Convert atom type from integer to string

    Parameters
    ----------
    atoms : string

    Returns
    -------
    atoms : integer

    """
    return ELEMENT_NAMES[atom]


def int_atom(atom: str) -> int:
    """
    Convert atom type from string to integer

    Parameters
    ----------
    atoms : string

    Returns
    -------
    atoms : integer
    """

    atom = atom.capitalize().strip()
    return NAMES_ELEMENT[atom]


def get_rotation_matrix(degrees: float) -> ndarray:
    """https://en.wikipedia.org/wiki/Rotation_matrix"""

    radians = degrees * np.pi / 180.0

    r11 = np.cos(radians)
    r12 = -np.sin(radians)
    r21 = np.sin(radians)
    r22 = np.cos(radians)

    R = np.array([[r11, r12], [r21, r22]])

    return R


def degree2radiant(degrees):
    return degrees * np.pi / 180.0


def rotate_coord(angle, coord, axis=[0, 1]):
    U = get_rotation_matrix(angle)
    _xy = np.dot(coord[:, axis], U)
    _coord = np.array(coord, copy=True)
    _coord[:, axis] = _xy
    return _coord


def compute_rmsd(P: ndarray, Q: ndarray, **kwargs) -> float:
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = P - Q
    return np.sqrt((diff * diff).sum() / P.shape[0])


def kabsch_rmsd(
    P: ndarray,
    Q: ndarray,
    W: Optional[ndarray] = None,
    translate: bool = False,
    **kwargs: Any,
) -> float:
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    An optional vector of weights W may be provided.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    translate : bool
        Use centroids to translate vector P and Q unto each other.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    if W is not None:
        return kabsch_weighted_rmsd(P, Q, W)

    P = kabsch_rotate(P, Q)
    return compute_rmsd(P, Q)


def kabsch_rotate(P: ndarray, Q: ndarray) -> ndarray:
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated

    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch_fit(P: ndarray, Q: ndarray, W: Optional[ndarray] = None) -> ndarray:
    """
    Rotate and translate matrix P unto matrix Q using Kabsch algorithm.
    An optional vector of weights W may be provided.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated and translated.

    """
    if W is not None:
        P, _ = kabsch_weighted_fit(P, Q, W, return_rmsd=False)
    else:
        QC = centroid(Q)
        Q = Q - QC
        P = P - centroid(P)
        P = kabsch_rotate(P, Q) + QC
    return P


def kabsch(P: ndarray, Q: ndarray) -> ndarray:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U: ndarray = np.dot(V, W)

    return U


def kabsch_weighted(
    P: ndarray, Q: ndarray, W: Optional[ndarray] = None
) -> Tuple[ndarray, ndarray, float]:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q.
    Each vector set is represented as an NxD matrix, where D is the
    dimension of the space.
    An optional vector of weights W may be provided.

    Note that this algorithm does not require that P and Q have already
    been overlayed by a centroid translation.

    The function returns the rotation matrix U, translation vector V,
    and RMS deviation between Q and P', where P' is:

        P' = P * U + V

    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.

    Returns
    -------
    U    : matrix
           Rotation matrix (D,D)
    V    : vector
           Translation vector (D)
    RMSD : float
           Root mean squared deviation between P and Q
    """
    # Computation of the weighted covariance matrix
    CMP = np.zeros(3)
    CMQ = np.zeros(3)
    C = np.zeros((3, 3))
    if W is None:
        W = np.ones(len(P)) / len(P)
    W = np.array([W, W, W]).T
    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw = 3.0 / W.sum()
    n = len(P)
    for i in range(3):
        for j in range(n):
            for k in range(3):
                C[i, k] += P[j, i] * Q[j, k] * W[j, i]
    CMP = (P * W).sum(axis=0)
    CMQ = (Q * W).sum(axis=0)
    PSQ = (P * P * W).sum() - (CMP * CMP).sum() * iw
    QSQ = (Q * Q * W).sum() - (CMQ * CMQ).sum() * iw
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U = np.dot(V, W)
    msd = (PSQ + QSQ) * iw - 2.0 * S.sum()
    if msd < 0.0:
        msd = 0.0
    rmsd_ = np.sqrt(msd)
    V = np.zeros(3)
    for i in range(3):
        t = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V = V * iw
    return U, V, rmsd_


def kabsch_weighted_fit(
    P: ndarray,
    Q: ndarray,
    W: Optional[ndarray] = None,
    return_rmsd: bool = False,
) -> Tuple[ndarray, Optional[float]]:
    """
    Fit P to Q with optional weights W.
    Also returns the RMSD of the fit if return_rmsd=True.

    Parameters
    ----------
    P    : array
           (N,D) matrix, where N is points and D is dimension.
    Q    : array
           (N,D) matrix, where N is points and D is dimension.
    W    : vector
           (N) vector, where N is points
    rmsd : Bool
           If True, rmsd is returned as well as the fitted coordinates.

    Returns
    -------
    P'   : array
           (N,D) matrix, where N is points and D is dimension.
    RMSD : float
           if the function is called with rmsd=True
    """
    rmsd_: float
    R, T, rmsd_ = kabsch_weighted(Q, P, W)
    PNEW: ndarray = np.dot(P, R.T) + T
    if return_rmsd:
        return (PNEW, rmsd_)

    return (PNEW, None)


def kabsch_weighted_rmsd(P: ndarray, Q: ndarray, W: Optional[ndarray] = None) -> float:
    """
    Calculate the RMSD between P and Q with optional weights W

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : vector
        (N) vector, where N is points

    Returns
    -------
    RMSD : float
    """
    _, _, w_rmsd = kabsch_weighted(P, Q, W)
    return w_rmsd


def quaternion_rmsd(P: ndarray, Q: ndarray, **kwargs: Any) -> float:
    """
    Rotate matrix P unto Q and calculate the RMSD
    based on doi:10.1016/1049-9660(91)90036-O

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
    """
    rot = quaternion_rotate(P, Q)
    P = np.dot(P, rot)
    return compute_rmsd(P, Q)


def quaternion_transform(r: ndarray) -> ndarray:
    """
    Get optimal rotation
    note: translation will be zero when the centroids of each molecule are the
    same
    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot: ndarray = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1: float, r2: float, r3: float, r4: float = 0) -> ndarray:
    """
    matrix involved in quaternion rotation
    """
    W = np.asarray(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return W


def makeQ(r1: float, r2: float, r3: float, r4: float = 0) -> ndarray:
    """
    matrix involved in quaternion rotation
    """
    Q = np.asarray(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return Q


def quaternion_rotate(X: ndarray, Y: ndarray) -> ndarray:
    """
    Calculate the rotation

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Y: array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rot : matrix
        Rotation matrix (D,D)
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    # NOTE UNUSED W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot


def centroid(X: ndarray) -> ndarray:
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : ndarray
        centroid
    """
    C: ndarray = X.mean(axis=0)
    return C


def hungarian_vectors(
    p_vecs: ndarray, q_vecs: ndarray, sigma: float = 1e-0, use_kernel: bool = True
) -> ndarray:
    """

    Hungarian cost assignment of a similiarty molecule kernel.

    Note: Assumes p and q are atoms of same type

    Parameters
    ----------
    p_vecs : array
        (N,L) matrix, where N is no. of atoms and L is representation length
    q_vecs : array
        (N,L) matrix, where N is no. of atoms and L is representation length

    Returns
    -------
    indices_b : array
        (N) view vector of reordered assignment

    """

    if use_kernel:
        # Calculate cost matrix from similarity kernel
        kernel = laplacian_kernel(p_vecs, q_vecs, sigma)
        kernel *= -1.0
        kernel += 1.0

    else:
        kernel = distance_matrix(p_vecs, q_vecs)

    _, indices_q = linear_sum_assignment(kernel)

    return indices_q


def reorder_similarity(
    p_atoms: ndarray,
    q_atoms: ndarray,
    p_coord: ndarray,
    q_coord: ndarray,
    use_kernel: bool = True,
    **kwargs: Any,
) -> ndarray:
    """
    Re-orders the input atom list and xyz coordinates using QML similarity
    the Hungarian method for assignment.

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of atom alignment based on the
             coordinates of the atoms
    """

    elements = np.unique(p_atoms)
    n_atoms = p_atoms.shape[0]
    distance_cut = 20.0

    parameters = {
        "elements": elements,
        "pad": n_atoms,
        "rcut": distance_cut,
        "acut": distance_cut,
    }

    p_vecs = generate_fchl19(p_atoms, p_coord, **parameters)
    q_vecs = generate_fchl19(q_atoms, q_coord, **parameters)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)

    for atom in elements:
        (p_atom_idx,) = np.where(p_atoms == atom)
        (q_atom_idx,) = np.where(q_atoms == atom)

        p_vecs_atom = p_vecs[p_atom_idx]
        q_vecs_atom = q_vecs[q_atom_idx]

        view = hungarian_vectors(p_vecs_atom, q_vecs_atom, use_kernel=use_kernel)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def reorder_distance(
    p_atoms: ndarray,
    q_atoms: ndarray,
    p_coord: ndarray,
    q_coord: ndarray,
    **kwargs: Any,
) -> ndarray:
    """
    Re-orders the input atom list and xyz coordinates by atom type and then by
    distance of each atom from the centroid.

    Parameters
    ----------
    atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    atoms_reordered : array
        (N,1) matrix, where N is points holding the ordered atoms' names
    coords_reordered : array
        (N,D) matrix, where N is points and D is dimension (rows re-ordered)
    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)

    for atom in unique_atoms:
        (p_atom_idx,) = np.where(p_atoms == atom)
        (q_atom_idx,) = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        # Calculate distance from each atom to centroid
        A_norms = np.linalg.norm(A_coord, axis=1)
        B_norms = np.linalg.norm(B_coord, axis=1)

        reorder_indices_A = np.argsort(A_norms)
        reorder_indices_B = np.argsort(B_norms)

        # Project the order of P onto Q
        translator = np.argsort(reorder_indices_A)
        view = reorder_indices_B[translator]
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def _hungarian(A: ndarray, B: ndarray) -> ndarray:
    """
    Hungarian reordering, minimizing the cost distances[winner_rows, winner_cols].sum())

    Assume A and B are coordinates for atoms of SAME type only.
    """

    distances = cdist(A, B, "euclidean")

    # Perform Hungarian analysis on distance matrix between atoms of 1st
    # structure and trial structure
    winner_rows, winner_cols = linear_sum_assignment(distances)

    return winner_cols


def reorder_hungarian(
    p_atoms: ndarray,
    q_atoms: ndarray,
    p_coord: ndarray,
    q_coord: ndarray,
    **kwargs: Any,
) -> ndarray:
    """
    Re-orders the input atom array and coordinates using the Hungarian-Distance
    sum assignment method. Returns view of q-atoms.

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atom types
    p_atoms : array
        (N,1) matrix, where N is points holding the atom types
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of atom alignment based on the
             coordinates of the atoms

    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        (p_atom_idx,) = np.where(p_atoms == atom)
        (q_atom_idx,) = np.where(q_atoms == atom)

        _p_coord = p_coord[p_atom_idx]
        _q_coord = q_coord[q_atom_idx]

        view = _hungarian(_p_coord, _q_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def reorder_inertia_hungarian(
    p_atoms: ndarray,
    q_atoms: ndarray,
    p_coord: ndarray,
    q_coord: ndarray,
    **kwargs: Any,
) -> ndarray:
    """
    First, align structures with the intertia moment eignvectors, then using
    distance hungarian, assign the best possible atom pair combinations. While
    also checking all possible reflections of intertia moments, selecting the
    one with minimal RMSD.

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of `Q` atom alignment based on the
             coordinates of the atoms

    """

    p_coord = np.array(p_coord, copy=True)
    q_coord = np.array(q_coord, copy=True)

    p_coord -= get_cm(p_atoms, p_coord)
    q_coord -= get_cm(q_atoms, p_coord)

    # Calculate inertia vectors for both structures
    inertia_p = get_inertia_tensor(p_atoms, p_coord)
    eigval_p, eigvec_p = np.linalg.eig(inertia_p)

    eigvec_p = eigvec_p.T
    eigvec_p = eigvec_p[np.argsort(eigval_p)]
    eigvec_p = eigvec_p.T

    inertia_q = get_inertia_tensor(q_atoms, q_coord)
    eigval_q, eigvec_q = np.linalg.eig(inertia_q)

    eigvec_q = eigvec_q.T
    eigvec_q = eigvec_q[np.argsort(eigval_q)]
    eigvec_q = eigvec_q.T

    # Reset the p coords, so the inertia vectors align with axis
    p_coord = np.dot(p_coord, eigvec_p)

    best_rmsd = np.inf
    best_review = np.arange(len(p_atoms))

    for mirror in AXIS_REFLECTIONS:
        tmp_eigvec = eigvec_q * mirror.T
        tmp_coord = np.dot(q_coord, tmp_eigvec)

        test_review = reorder_hungarian(p_atoms, q_atoms, p_coord, tmp_coord)
        test_rmsd = kabsch_rmsd(tmp_coord[test_review], p_coord)

        if test_rmsd < best_rmsd:
            best_rmsd = test_rmsd
            best_review = test_review

    return best_review


def generate_permutations(elements: List[int], n: int) -> Iterator[List[int]]:
    """
    Heap's algorithm for generating all n! permutations in a list
    https://en.wikipedia.org/wiki/Heap%27s_algorithm

    """
    c = [0] * n
    yield elements
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                elements[0], elements[i] = elements[i], elements[0]
            else:
                elements[c[i]], elements[i] = elements[i], elements[c[i]]
            yield elements
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1


def brute_permutation(A: ndarray, B: ndarray) -> ndarray:
    """
    Re-orders the input atom list and xyz coordinates using the brute force
    method of permuting all rows of the input coordinates

    Parameters
    ----------
    A : array
        (N,D) matrix, where N is points and D is dimension
    B : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view : array
        (N,1) matrix, reordered view of B projected to A
    """

    rmsd_min = np.inf
    view_min: ndarray

    # Sets initial ordering for row indices to [0, 1, 2, ..., len(A)], used in
    # brute-force method

    num_atoms = A.shape[0]
    initial_order = list(range(num_atoms))

    for reorder_indices in generate_permutations(initial_order, num_atoms):
        # Re-order the atom array and coordinate matrix
        coords_ordered = B[reorder_indices]

        # Calculate the RMSD between structure 1 and the Hungarian re-ordered
        # structure 2
        rmsd_temp = kabsch_rmsd(A, coords_ordered)

        # Replaces the atoms and coordinates with the current structure if the
        # RMSD is lower
        if rmsd_temp < rmsd_min:
            rmsd_min = rmsd_temp
            view_min = np.asarray(copy.deepcopy(reorder_indices))

    return view_min


def reorder_brute(
    p_atoms: ndarray,
    q_atoms: ndarray,
    p_coord: ndarray,
    q_coord: ndarray,
    **kwargs: Any,
) -> ndarray:
    """
    Re-orders the input atom list and xyz coordinates using all permutation of
    rows (using optimized column results)

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    q_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    view_reorder : array
        (N,1) matrix, reordered indexes of atom alignment based on the
        coordinates of the atoms

    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        (p_atom_idx,) = np.where(p_atoms == atom)
        (q_atom_idx,) = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        view = brute_permutation(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def check_reflections(
    p_atoms: ndarray,
    q_atoms: ndarray,
    p_coord: ndarray,
    q_coord: ndarray,
    reorder_method: Optional[ReorderCallable] = None,
    rmsd_method: RmsdCallable = kabsch_rmsd,
    keep_stereo: bool = False,
) -> Tuple[float, ndarray, ndarray, ndarray]:
    """
    Minimize RMSD using reflection planes for molecule P and Q

    Warning: This will affect stereo-chemistry

    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    q_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension

    Returns
    -------
    min_rmsd
    min_swap
    min_reflection
    min_review

    """

    if reorder_method is None:
        assert (p_atoms == q_atoms).all(), (
            "No reorder method selected, but atoms are not ordered"
        )

    min_rmsd = np.inf
    min_swap: ndarray
    min_reflection: ndarray
    min_review: ndarray = np.array(range(len(p_atoms)))
    tmp_review: ndarray = min_review
    swap_mask = [1, -1, -1, 1, -1, 1]
    reflection_mask = [1, -1, -1, -1, 1, 1, 1, -1]

    for swap, i in zip(AXIS_SWAPS, swap_mask):
        for reflection, j in zip(AXIS_REFLECTIONS, reflection_mask):
            # skip enantiomers
            if keep_stereo and i * j == -1:
                continue

            tmp_atoms = copy.copy(q_atoms)
            tmp_coord = copy.deepcopy(q_coord)
            tmp_coord = tmp_coord[:, swap]
            tmp_coord = np.dot(tmp_coord, np.diag(reflection))
            tmp_coord -= centroid(tmp_coord)

            # Reorder
            if reorder_method is not None:
                tmp_review = reorder_method(p_atoms, tmp_atoms, p_coord, tmp_coord)
                tmp_coord = tmp_coord[tmp_review]
                tmp_atoms = tmp_atoms[tmp_review]

            # Rotation
            this_rmsd = rmsd_method(p_coord, tmp_coord)

            if this_rmsd < min_rmsd:
                min_rmsd = this_rmsd
                min_swap = swap
                min_reflection = reflection
                min_review = tmp_review

    assert (p_atoms == q_atoms[min_review]).all(), "error: Not aligned"

    return min_rmsd, min_swap, min_reflection, min_review


def rotation_matrix_vectors(v1: ndarray, v2: ndarray) -> ndarray:
    """
    Returns the rotation matrix that rotates v1 onto v2
    using Rodrigues' rotation formula.
    (see https://math.stackexchange.com/a/476311)
    ----------
    v1 : array
        Dim 3 float array
    v2 : array
        Dim 3 float array

    Return
    ------
    output : 3x3 matrix
        Rotation matrix
    """

    rot: ndarray

    if (v1 == v2).all():
        rot = np.eye(3)

    # return a rotation of pi around the y-axis
    elif (v1 == -v2).all():
        rot = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    else:
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.vdot(v1, v2)

        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

        rot = np.eye(3) + vx + np.dot(vx, vx) * ((1.0 - c) / (s * s))

    return rot


def get_cm(atoms: ndarray, V: ndarray) -> ndarray:
    """
    Get the center of mass of V.
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates

    Return
    ------
    output : (3) array
        The CM vector
    """

    weights: Union[List[float], ndarray] = [ELEMENT_WEIGHTS[x] for x in atoms]
    weights = np.asarray(weights)
    center_of_mass: ndarray = np.average(V, axis=0, weights=weights)

    return center_of_mass


def get_inertia_tensor(atoms: ndarray, coord: ndarray) -> ndarray:
    """
    Get the tensor of intertia of V.
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates

    Return
    ------
    output : 3x3 float matrix
        The tensor of inertia
    """

    coord -= get_cm(atoms, coord)

    Ixx = 0.0
    Iyy = 0.0
    Izz = 0.0
    Ixy = 0.0
    Ixz = 0.0
    Iyz = 0.0

    for sp, acoord in zip(atoms, coord):
        amass = ELEMENT_WEIGHTS[sp]
        Ixx += amass * (acoord[1] * acoord[1] + acoord[2] * acoord[2])
        Iyy += amass * (acoord[0] * acoord[0] + acoord[2] * acoord[2])
        Izz += amass * (acoord[0] * acoord[0] + acoord[1] * acoord[1])
        Ixy += -amass * acoord[0] * acoord[1]
        Ixz += -amass * acoord[0] * acoord[2]
        Iyz += -amass * acoord[1] * acoord[2]

    atomic_masses = np.asarray([ELEMENT_WEIGHTS[a] for a in atoms])

    mass_matrix = np.diag(atomic_masses)
    helper = coord.T.dot(mass_matrix).dot(coord)
    inertia_tensor: np.ndarray = np.diag(np.ones(3)) * helper.trace() - helper
    return inertia_tensor


def get_principal_axis(atoms: ndarray, V: ndarray) -> ndarray:
    """
    Get the molecule's principal axis.
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates

    Return
    ------
    output : array
        Array of dim 3 containing the principal axis
    """
    inertia = get_inertia_tensor(atoms, V)

    eigval, eigvec = np.linalg.eig(inertia)

    principal_axis: ndarray = eigvec[np.argmax(eigval)]

    return principal_axis


def set_coordinates(
    atoms: ndarray,
    V: ndarray,
    title: str = "",
    decimals: int = 8,
    set_atoms_as_symbols: bool = True,
) -> str:
    """
    Print coordinates V with corresponding atoms to stdout in XYZ format.
    Parameters
    ----------
    atoms : list
        List of atomic types
    V : array
        (N,3) matrix of atomic coordinates
    title : string (optional)
        Title of molecule
    decimals : int (optional)
        number of decimals for the coordinates

    Return
    ------
    output : str
        Molecule in XYZ format

    """
    N, D = V.shape

    if N != len(atoms):
        raise ValueError("Mismatch between expected atoms and coordinate size")

    if not isinstance(atoms[0], str) and set_atoms_as_symbols:
        atoms = np.array([str_atom(atom) for atom in atoms])

    fmt = "{:<2}" + (" {:15." + str(decimals) + "f}") * 3

    out = list()
    out += [str(N)]
    out += [title]

    for i in range(N):
        atom = atoms[i]
        out += [fmt.format(atom, V[i, 0], V[i, 1], V[i, 2])]

    newline = "\n"

    return newline.join(out)


def get_coordinates(
    filename: Path, fmt: str, is_gzip: bool = False, return_atoms_as_int: bool = False
) -> Tuple[ndarray, ndarray]:
    """
    Get coordinates from filename in format fmt. Supports XYZ and PDB.
    Parameters
    ----------
    filename : string
        Filename to read
    fmt : string
        Format of filename. Either xyz or pdb.
    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """
    if fmt == "xyz":
        get_func = get_coordinates_xyz

    elif fmt == "pdb":
        get_func = get_coordinates_pdb

    else:
        raise ValueError("Could not recognize file format: {:s}".format(fmt))

    val = get_func(filename, is_gzip=is_gzip, return_atoms_as_int=return_atoms_as_int)

    return val


def _parse_pdb_alphacarbon_line(line: str) -> bool:
    """Try to read Alpha carbons based on PDB column-based format"""

    atom_col = line[12:16]
    atom = atom_col[0:2]
    atom = re.sub(r"\d", " ", atom)
    atom = atom.strip()
    atom = atom.capitalize()
    location = atom_col[2]

    if atom == "C" and location == "A":
        return True

    return False


def _parse_pdb_atom_line(line: str) -> Optional[str]:
    """
    Will try it best to find atom from an atom-line. The standard of PDB
    *should* be column based, however, there are many examples of non-standard
    files. We try our best to have a minimal reader.

    From PDB Format 1992 pdf:

        ATOM Atomic coordinate records for "standard" groups
        HETATM Atomic coordinate records for "non-standard" groups

        Cols. 1 - 4  ATOM
           or 1 - 6  HETATM

              7 - 11 Atom serial number(i)
             13 - 16 Atom name(ii)
             17      Alternate location indicator(iii)
             18 - 20 Residue name(iv,v)
             22      Chain identifier, e.g., A for hemoglobin α chain
             23 - 26 Residue seq. no.
             27      Code for insertions of residues, e.g., 66A, 66B, etc.
             31 - 38 X
             39 - 46 Y Orthogonal Å coordinates
             47 - 54 Z
             55 - 60 Occupancy
             61 - 66 Temperature factor(vi)
             68 - 70 Footnote number

        For (II)

        Within each residue the atoms occur in the order specified by the
        superscripts. The extra oxygen atom of the carboxy terminal amino acid
        is designated OXT

        Four characters are reserved for these atom names. They are assigned as
        follows:

        1-2 Chemical symbol - right justified
        3 Remoteness indicator (alphabetic)
        4 Branch designator (numeric)

        For protein coordinate sets containing hydrogen atoms, the IUPAC-IUB
        rules1 have been followed. Recommendation rule number 4.4 has been
        modified as follows: When more than one hydrogen atom is bonded to a
        single non-hydrogen atom, the hydrogen atom number designation is given
        as the first character of the atom name rather than as the last
        character (e.g. Hβ1 is denoted as 1HB). Exceptions to these rules may
        occur in certain data sets at the depositors's request. Any such
        exceptions will be delineated clearly in FTNOTE and REMARK records

    but, from [PDB Format Version 2.1]

        In large het groups it sometimes is not possible to follow the
        convention of having the first two characters be the chemical symbol
        and still use atom names that are meaningful to users. A example is
        nicotinamide adenine dinucleotide, atom names begin with an A or N,
        depending on which portion of the molecule they appear in, e.g., AC6 or
        NC6, AN1 or NN1.

        Hydrogen naming sometimes conflicts with IUPAC conventions. For
        example, a hydrogen named HG11 in columns 13 - 16 is differentiated
        from a mercury atom by the element symbol in columns 77 - 78. Columns
        13 - 16 present a unique name for each atom.

    """

    atom_col = line[12:16]
    atom = atom_col[0:2]
    atom = re.sub(r"\d", " ", atom)
    atom = atom.strip()
    atom = atom.capitalize()

    # Highly unlikely that it is Mercury, Helium, Hafnium etc. See comment in
    # function description. [PDB Format v2.1]
    if len(atom) == 2 and atom[0] == "H":
        atom = "H"

    if atom in NAMES_ELEMENT.keys():
        return atom

    tokens = line.split()
    atom = tokens[2][0]
    if atom in NAMES_ELEMENT.keys():
        return atom

    # e.g. 1HD1
    atom = tokens[2][1]
    if atom.upper() == "H":
        return atom

    return None


def _parse_pdb_coord_line(line: str) -> Optional[ndarray]:
    """
    Try my best to coordinates from a PDB ATOM or HETATOM line

    The coordinates should be located in
        31 - 38 X
        39 - 46 Y
        47 - 54 Z

    as defined in PDB, ATOMIC COORDINATE AND BIBLIOGRAPHIC ENTRY FORMAT DESCRIPTION, Feb, 1992
    """

    # If that doesn't work, use hardcoded indices
    try:
        x = line[30:38]
        y = line[38:46]
        z = line[46:54]
        coord = np.asarray([x, y, z], dtype=float)
        return coord

    except ValueError:
        coord = None

    tokens = line.split()

    x_column: Optional[int] = None

    # look for x column
    for i, x in enumerate(tokens):
        if "." in x and "." in tokens[i + 1] and "." in tokens[i + 2]:
            x_column = i
            break

    if x_column is None:
        return None

    # Try to read the coordinates
    try:
        coord = np.asarray(tokens[x_column : x_column + 3], dtype=float)
        return coord
    except ValueError:
        coord = None

    return None


def get_coordinates_pdb(
    filename: Path,
    is_gzip: bool = False,
    return_atoms_as_int: bool = False,
    only_alpha_carbon: bool = False,
) -> Tuple[ndarray, ndarray]:
    """
    Get coordinates from the first chain in a pdb file
    and return a vectorset with all the coordinates.

    Parameters
    ----------
    filename : string
        Filename to read

    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """

    # PDB files tend to be a bit of a mess. The x, y and z coordinates
    # are supposed to be in column 31-38, 39-46 and 47-54, but this is
    # not always the case.
    # Because of this the three first columns containing a decimal is used.
    # Since the format doesn't require a space between columns, we use the
    # above column indices as a fallback.

    V: Union[List[ndarray], ndarray] = list()
    assert isinstance(V, list)

    # Same with atoms and atom naming.
    # The most robust way to do this is probably
    # to assume that the atomtype is given in column 3.

    atoms: List[str] = list()
    alpha_carbons: List[bool] = list()
    assert isinstance(atoms, list)
    openfunc: Any

    if is_gzip:
        openfunc = gzip.open
        openarg = "rt"
    else:
        openfunc = open
        openarg = "r"

    with openfunc(filename, openarg) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("TER") or line.startswith("END"):
                break

            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = _parse_pdb_atom_line(line)
            if atom is None:
                raise ValueError(f"error: Parsing for atom line: {line}")
            atoms.append(atom)

            coord = _parse_pdb_coord_line(line)
            if coord is None:
                raise ValueError(f"error: Parsing coordinates for line: {line}")
            V.append(coord)

            # Check if alpha-carbon
            is_alpha = _parse_pdb_alphacarbon_line(line)
            alpha_carbons.append(is_alpha)

    if return_atoms_as_int:
        _atoms = np.asarray([int_atom(str(atom)) for atom in atoms])
    else:
        _atoms = np.asarray(atoms)

    V = np.asarray(V)
    assert isinstance(V, ndarray)

    assert isinstance(_atoms, ndarray)
    assert V.shape[0] == _atoms.size

    if only_alpha_carbon:
        # Check that any alpha carbons were found
        if not sum(alpha_carbons):
            raise ValueError(
                "Trying to filter for alpha carbons, but couldn't find any"
            )

        _alpha_carbons = np.asarray(alpha_carbons, dtype=bool)

        V = V[_alpha_carbons, :]
        _atoms = _atoms[_alpha_carbons]

    return _atoms, V


def get_coordinates_xyz_lines(
    lines: List[str], return_atoms_as_int: bool = False
) -> Tuple[ndarray, ndarray]:
    V: Union[List[ndarray], ndarray] = list()
    atoms: Union[List[str], ndarray] = list()
    n_atoms = 0

    assert isinstance(V, list)
    assert isinstance(atoms, list)

    # Read the first line to obtain the number of atoms to read
    try:
        n_atoms = int(lines[0])
    except ValueError:
        exit("error: Could not obtain the number of atoms in the .xyz file.")

    # Skip the title line
    # Use the number of atoms to not read beyond the end of a file
    for lines_read, line in enumerate(lines[2:]):
        line = line.strip()

        if lines_read == n_atoms:
            break

        values = line.split()

        if len(values) < 4:
            atom = re.findall(r"[a-zA-Z]+", line)[0]
            atom = atom.upper()
            numbers = re.findall(r"[-]?\d+\.\d*(?:[Ee][-\+]\d+)?", line)
            numbers = [float(number) for number in numbers]
        else:
            atom = values[0]
            numbers = [float(number) for number in values[1:]]

        # The numbers are not valid unless we obtain exacly three
        if len(numbers) >= 3:
            V.append(np.array(numbers)[:3])
            atoms.append(atom)
        else:
            msg = (
                f"Reading the .xyz file failed in line {lines_read + 2}."
                "Please check the format."
            )
            exit(msg)

    try:
        # I've seen examples where XYZ are written with integer atoms types
        atoms_ = [int(atom) for atom in atoms]
        atoms = [str_atom(atom) for atom in atoms_]

    except ValueError:
        # Correct atom spelling
        atoms = [atom.capitalize() for atom in atoms]

    if return_atoms_as_int:
        atoms_ = [int_atom(atom) for atom in atoms]
        atoms = np.array(atoms_)
    else:
        atoms = np.array(atoms)

    V = np.array(V)

    return atoms, V


def get_coordinates_xyz(
    filename: Path,
    is_gzip: bool = False,
    return_atoms_as_int: bool = False,
) -> Tuple[ndarray, ndarray]:
    """
    Get coordinates from filename and return a vectorset with all the
    coordinates, in XYZ format.

    Parameters
    ----------
    filename : string
        Filename to read

    Returns
    -------
    atoms : list
        List of atomic types
    V : array
        (N,3) where N is number of atoms
    """

    openfunc: Any

    if is_gzip:
        openfunc = gzip.open
        openarg = "rt"
    else:
        openfunc = open
        openarg = "r"

    with openfunc(filename, openarg) as f:
        lines = f.readlines()

    atoms, V = get_coordinates_xyz_lines(lines, return_atoms_as_int=return_atoms_as_int)

    return atoms, V


# added by Andreas
def _prepare_atoms_array(
    atoms_in: Union[List[str], List[int], np.ndarray], natoms: int
) -> np.ndarray:
    """
    Helper function to convert various atom type inputs to a NumPy array of integers.
    """
    if not isinstance(atoms_in, (list, np.ndarray)):
        raise TypeError("Atom types must be a list or NumPy array.")
    if len(atoms_in) != natoms:
        raise ValueError(
            f"Number of atom types ({len(atoms_in)}) must match number of coordinates ({natoms})."
        )

    if isinstance(atoms_in, np.ndarray) and np.issubdtype(atoms_in.dtype, np.integer):
        return atoms_in.astype(int)

    atoms_out = np.zeros(natoms, dtype=int)
    is_str_list = all(isinstance(a, str) for a in atoms_in)
    is_int_list = all(isinstance(a, int) for a in atoms_in)

    if is_str_list:
        try:
            atoms_out = np.array([int_atom(a) for a in atoms_in], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"Unknown atom symbol: {e}. Ensure all atom symbols are valid."
            )
    elif is_int_list:
        atoms_out = np.array(atoms_in, dtype=int)
    elif isinstance(atoms_in, np.ndarray):  # Could be an object array or float array
        try:
            # Attempt direct conversion, ensure it's clean
            converted_atoms = atoms_in.astype(int)
            if not np.array_equal(
                converted_atoms.astype(atoms_in.dtype), atoms_in
            ):  # check if conversion was lossy for floats
                raise ValueError(
                    "Atom types array could not be cleanly converted to integers."
                )
            atoms_out = converted_atoms
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Atom types array could not be converted to integers: {e}. "
                "Provide symbols or integers."
            )
    else:  # Mixed list
        raise ValueError(
            "Atom types must be a list of strings (symbols), a list of integers (atomic numbers), "
            "or a NumPy array of integers."
        )
    return atoms_out


# added by Andreas
def rmsd_from_numpy(
    p_atoms_in: Union[List[str], List[int], np.ndarray],
    p_coord_in: np.ndarray,
    q_atoms_in: Union[List[str], List[int], np.ndarray],
    q_coord_in: np.ndarray,
    reorder: bool = False,
    reorder_method_str: str = REORDER_NONE,
    rotation_method_str: str = METHOD_KABSCH,
    use_reflections: bool = False,
    use_reflections_keep_stereo: bool = False,
) -> float:
    """
    Calculate RMSD between two molecules specified by NumPy arrays.

    Parameters
    ----------
    p_atoms_in : Union[List[str], List[int], np.ndarray]
        Atom types for molecule P (list of symbols, list of atomic numbers, or ndarray of atomic numbers).
    p_coord_in : np.ndarray
        Coordinates for molecule P (N, 3) NumPy array.
    q_atoms_in : Union[List[str], List[int], np.ndarray]
        Atom types for molecule Q.
    q_coord_in : np.ndarray
        Coordinates for molecule Q (N, 3) NumPy array.
    reorder : bool, optional
        Whether to reorder atoms to find the best match. Defaults to False.
    reorder_method_str : str, optional
        The method for reordering atoms if `reorder` is True.
        Valid methods: "none", "qml", "hungarian", "inertia-hungarian", "brute", "distance".
        Defaults to "none".
    rotation_method_str : str, optional
        The rotation method to use.
        Valid methods: "kabsch", "quaternion", "none".
        Defaults to "kabsch".
    use_reflections : bool, optional
        Whether to check reflections to minimize RMSD. This may change stereochemistry.
        Defaults to False.
    use_reflections_keep_stereo : bool, optional
        Whether to check reflections while attempting to preserve stereochemistry.
        Defaults to False.

    Returns
    -------
    float
        The calculated RMSD value.

    Raises
    ------
    ValueError
        If inputs are inconsistent (e.g., mismatched array sizes, unknown methods,
        atom type mismatch when reordering is not enabled).
    ImportError
        If "qml" reorder method is selected but `qmllib` is not installed.
    """

    # Coordinate preparation
    p_coord = np.asarray(p_coord_in, dtype=float)
    q_coord = np.asarray(q_coord_in, dtype=float)

    if p_coord.ndim != 2 or p_coord.shape[1] != 3:
        raise ValueError("P coordinates must be an (N, 3) array.")
    if q_coord.ndim != 2 or q_coord.shape[1] != 3:
        raise ValueError("Q coordinates must be an (N, 3) array.")
    if p_coord.shape[0] != q_coord.shape[0]:
        raise ValueError(
            f"Number of atoms in P ({p_coord.shape[0]}) and Q ({q_coord.shape[0]}) must be identical."
        )
    N = p_coord.shape[0]
    if N == 0:
        return 0.0  # RMSD is 0 if there are no atoms

    # Atom preparation
    p_atoms = _prepare_atoms_array(p_atoms_in, N)
    q_atoms = _prepare_atoms_array(q_atoms_in, N)

    # Atom counts check if reordering or reflections are involved
    if reorder or use_reflections or use_reflections_keep_stereo:
        p_unique_atoms, p_counts = np.unique(p_atoms, return_counts=True)
        q_unique_atoms, q_counts = np.unique(q_atoms, return_counts=True)
        p_sort_idx = np.argsort(p_unique_atoms)
        q_sort_idx = np.argsort(q_unique_atoms)
        if not (
            np.array_equal(p_unique_atoms[p_sort_idx], q_unique_atoms[q_sort_idx])
            and np.array_equal(p_counts[p_sort_idx], q_counts[q_sort_idx])
        ):
            raise ValueError(
                "For reordering or reflection checks, the set of atom types and their "
                "counts must be identical between the two molecules."
            )

    # Check for atom type/order mismatch if not reordering explicitly
    # (reflections might reorder internally, so this check is conditional)
    if not reorder and not (use_reflections or use_reflections_keep_stereo):
        if np.any(p_atoms != q_atoms):
            raise ValueError(
                "Atom types or order mismatch, and reordering is not enabled. "
                "Set reorder=True and specify reorder_method_str, or ensure atoms match."
            )

    # Create working copies for coordinates (atoms are used for indexing or comparison)
    p_coord_c = np.copy(p_coord)
    q_coord_c = np.copy(q_coord)

    # Centering
    p_coord_c -= centroid(p_coord_c)
    q_coord_c -= centroid(q_coord_c)

    # Select RMSD method
    actual_rmsd_method: RmsdCallable
    if rotation_method_str == METHOD_KABSCH:
        actual_rmsd_method = kabsch_rmsd
    elif rotation_method_str == METHOD_QUATERNION:
        actual_rmsd_method = quaternion_rmsd
    elif rotation_method_str == METHOD_NOROTATION:
        actual_rmsd_method = compute_rmsd
    else:
        raise ValueError(
            f"Unknown rotation method: '{rotation_method_str}'. Valid are: {ROTATION_METHODS}"
        )

    # Select Reorder method
    actual_reorder_method: Optional[ReorderCallable] = None
    if reorder:  # Only process reorder_method_str if reorder is True
        if reorder_method_str == REORDER_QML:
            if qmllib is None:
                raise ImportError(
                    "QML reordering selected, but qmllib is not installed. "
                    "Install with: pip install qmllib"
                )
            actual_reorder_method = reorder_similarity
        elif reorder_method_str == REORDER_HUNGARIAN:
            actual_reorder_method = reorder_hungarian
        elif reorder_method_str == REORDER_INERTIA_HUNGARIAN:
            # This method uses ELEMENT_WEIGHTS. Ensure atom types used are compatible.
            # For generic identical particles, using type 1 (Hydrogen) is a safe bet.
            # If using other types (e.g., 0), this method might fail if the type isn't in ELEMENT_WEIGHTS.
            actual_reorder_method = reorder_inertia_hungarian
        elif reorder_method_str == REORDER_BRUTE:
            actual_reorder_method = reorder_brute
        elif reorder_method_str == REORDER_DISTANCE:
            actual_reorder_method = reorder_distance
        elif reorder_method_str == REORDER_NONE:
            actual_reorder_method = None  # Explicitly None for "no reordering"
        else:
            raise ValueError(
                f"Unknown reorder method: '{reorder_method_str}'. Valid are: {REORDER_METHODS}"
            )
    elif reorder_method_str != REORDER_NONE:
        # User specified a reorder method but reorder=False. This is potentially confusing.
        # We will ignore reorder_method_str if reorder=False.
        # Or, one could raise a warning/error. For now, matching main CLI logic.
        pass

    result_rmsd: float

    if use_reflections or use_reflections_keep_stereo:
        reorder_meth_for_reflection = actual_reorder_method if reorder else None

        if reorder_meth_for_reflection is None and np.any(p_atoms != q_atoms):
            raise ValueError(
                "Atom types or order mismatch. Enable reordering (reorder=True and "
                "specify reorder_method_str not 'none') or ensure atoms are identical "
                "and in order when using reflections without internal reordering."
            )

        rmsd_val, _, _, _ = check_reflections(
            p_atoms,
            q_atoms,
            p_coord_c,
            q_coord_c,
            reorder_method=reorder_meth_for_reflection,
            rmsd_method=actual_rmsd_method,
            keep_stereo=use_reflections_keep_stereo,
        )
        result_rmsd = rmsd_val

    elif reorder and actual_reorder_method is not None:
        # Atom counts already checked if reorder is True (step 5)
        q_review_indices = actual_reorder_method(p_atoms, q_atoms, p_coord_c, q_coord_c)

        if not (p_atoms == q_atoms[q_review_indices]).all():
            # This should not happen if atom counts (per type) match and reorder alg is correct.
            raise AssertionError(
                "Internal error: Atom alignment failed after reordering. "
                "Ensure atom counts per type match between molecules."
            )

        q_coord_reordered = q_coord_c[q_review_indices]
        result_rmsd = actual_rmsd_method(p_coord_c, q_coord_reordered)
    else:
        # This covers:
        # 1. reorder=False (atom matching handled by the check in step `if not reorder and not (use_reflections...`)
        # 2. reorder=True AND actual_reorder_method is None (i.e., reorder_method_str was REORDER_NONE)
        if np.any(p_atoms != q_atoms):
            raise ValueError(
                "Atom types or order mismatch. Ensure atoms are identical and in order, "
                "or select a valid reorder_method (not 'none') when reorder=True."
            )
        result_rmsd = actual_rmsd_method(p_coord_c, q_coord_c)

    return result_rmsd


def remove_duplicates(structures, atom_types, rmsd_cutoff: float = 1e-1) -> np.ndarray:
    """Remove almost identical structures from a list of structures.
    If there are multiple identical structures (RMSD below cutoff), keep only one.

    structures: array-like of shape (B, N, 3)
    atom_types: array-like of shape (B, N)
    where B is the number of structures, N is the number of atoms, and 3 is the number of spatial dimensions (x, y, z).

    Returns:
    Boolean mask of structures to keep.

    Usage:
    structures_to_keep_mask = remove_duplicates(structures, atom_types)
    structures = structures[structures_to_keep_mask]
    atom_types = atom_types[structures_to_keep_mask]
    """
    # naive implementation, no parallelization
    # possibly two ways to parallelize:
    # 1. loop over i and parallelise over j (less parallelism but less computation)
    # 2. parallelize over all pairs i and j (max parallelism but more computation)
    B = len(structures)
    structures_to_keep_mask = np.ones(B, dtype=bool)
    for i in range(B):
        if not structures_to_keep_mask[i]:
            continue
        for j in range(i + 1, B):
            if structures_to_keep_mask[j]:
                rmsd = rmsd_from_numpy(
                    p_atoms_in=np.asarray(atom_types[i], dtype=int),
                    p_coord_in=np.asarray(structures[i], dtype=float),
                    q_atoms_in=np.asarray(atom_types[j], dtype=int),
                    q_coord_in=np.asarray(structures[j], dtype=float),
                )
                if rmsd < rmsd_cutoff:
                    # j is a duplicate of i
                    # keep i, remove j
                    structures_to_keep_mask[j] = False
    return structures_to_keep_mask


def print_pairwise_rmsd(structures, atom_types):
    """Print pairwise RMSDs between structures."""
    print("\nPairwise RMSDs:")
    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            rmsd_val = rmsd_from_numpy(
                p_atoms_in=atom_types[i],
                p_coord_in=structures[i],
                q_atoms_in=atom_types[j],
                q_coord_in=structures[j],
            )
            print(f"RMSD between structure {i} and {j}: {rmsd_val:.6f}")


def parse_arguments(
    arguments: Optional[Union[str, List[str]]] = None,
) -> argparse.Namespace:
    sep = ", "

    parser = argparse.ArgumentParser(
        usage="calculate_rmsd [options] FILE_A FILE_B",
        description=__intro__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__details__,
    )

    # Input structures
    parser.add_argument(
        "structure_a",
        metavar="FILE_A",
        type=str,
        help="Structures in .xyz or .pdb format",
    )
    parser.add_argument("structure_b", metavar="FILE_B", type=str)

    # Admin
    parser.add_argument("-v", "--version", action="version", version=__version__)

    # Rotation
    parser.add_argument(
        "-r",
        "--rotation",
        action="store",
        default="kabsch",
        help=(
            f"Select rotation method. Valid methods are: {sep.join(ROTATION_METHODS)}. Default is `Kabsch`."
        ),
        metavar="METHOD",
        choices=ROTATION_METHODS,
    )

    # Reorder arguments
    reorder_group = parser.add_argument_group(description="Atom reordering arguments")
    reorder_group.add_argument(
        "-e",
        "--reorder",
        action="store_true",
        help="Align the atoms of molecules",
    )
    reorder_group.add_argument(
        "--reorder-method",
        action="store",
        default="inertia-hungarian",
        metavar="METHOD",
        help=(
            "Select reorder method. Valid method are "
            f"{sep.join(REORDER_METHODS)}. "
            "Default is Inertia-Hungarian."
        ),
        choices=REORDER_METHODS,
    )

    reflection_group = parser.add_argument_group(description="Reflection arguments")
    reflection_group.add_argument(
        "--use-reflections",
        action="store_true",
        help=(
            "Scan through reflections in planes "
            "(eg Y transformed to -Y -> X, -Y, Z) "
            "and axis changes, (eg X and Z coords exchanged -> Z, Y, X). "
            "This will affect stereo-chemistry."
        ),
    )
    reflection_group.add_argument(
        "--use-reflections-keep-stereo",
        action="store_true",
        help=(
            "Scan through reflections in planes "
            "(eg Y transformed to -Y -> X, -Y, Z) "
            "and axis changes, (eg X and Z coords exchanged -> Z, Y, X). "
            "Stereo-chemistry will be kept."
        ),
    )

    # Filter
    index_group_title = parser.add_argument_group(
        description="Atom filtering arguments"
    )
    index_group = index_group_title.add_mutually_exclusive_group()
    index_group.add_argument(
        "--only-alpha-carbons",
        action="store_true",
        help="Use only alpha carbons (only for PDB format)",
    )
    index_group.add_argument(
        "-n",
        "--ignore-hydrogen",
        "--no-hydrogen",
        action="store_true",
        help="Ignore Hydrogens when calculating RMSD",
    )
    index_group.add_argument(
        "--remove-idx",
        nargs="+",
        type=int,
        help="Index list of atoms NOT to consider",
        metavar="IDX",
    )
    index_group.add_argument(
        "--add-idx",
        nargs="+",
        type=int,
        help="Index list of atoms to consider",
        metavar="IDX",
    )

    parser.add_argument(
        "--format",
        action="store",
        help=f"Format of input files. valid format are {sep.join(FORMATS)}.",
        metavar="FMT",
    )
    parser.add_argument(
        "--format-is-gzip",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "-p",
        "--output",
        "--print",
        action="store_true",
        help=(
            "Print out structure B, centered and rotated unto structure A's coordinates in XYZ format"
        ),
    )

    parser.add_argument(
        "--print-only-rmsd-atoms",
        action="store_true",
        help=(
            "Print only atoms used in finding optimal RMSD calculation (relevant if filtering e.g. Hydrogens)"
        ),
    )

    args = parser.parse_args(arguments)

    # Check illegal combinations
    if (
        args.output
        and args.reorder
        and (args.ignore_hydrogen or args.add_idx or args.remove_idx)
    ):
        print(
            "error: Cannot reorder atoms and print structure, when excluding atoms (such as --ignore-hydrogen)"
        )
        sys.exit(5)

    if (
        args.use_reflections
        and args.output
        and (args.ignore_hydrogen or args.add_idx or args.remove_idx)
    ):
        print(
            "error: Cannot use reflections on atoms and print, "
            "when excluding atoms (such as --ignore-hydrogen)"
        )
        sys.exit(5)

    # Check methods
    args.rotation = args.rotation.lower()
    if args.rotation not in ROTATION_METHODS:
        print(
            f"error: Unknown rotation method: '{args.rotation}'. "
            f"Please use {ROTATION_METHODS}"
        )
        sys.exit(5)

    # Check reorder methods
    args.reorder_method = args.reorder_method.lower()
    if args.reorder_method not in REORDER_METHODS:
        print(
            f'error: Unknown reorder method: "{args.reorder_method}". '
            f"Please use {REORDER_METHODS}"
        )
        sys.exit(5)

    # Check fileformat
    if args.format is None:
        filename = args.structure_a
        suffixes = Path(filename).suffixes

        if len(suffixes) == 0:
            ext = None

        elif suffixes[-1] == ".gz":
            args.format_is_gzip = True
            ext = suffixes[-2].strip(".")

        else:
            ext = suffixes[-1].strip(".")

        args.format = ext

    # Check if format exist
    if args.format not in FORMATS:
        print(f"error: Format not supported {args.format}")
        sys.exit(5)

    # Check illegal argument
    if args.format != FORMAT_PDB and args.only_alpha_carbons:
        print("Alpha carbons only exist in pdb files")
        sys.exit(5)

    # Check QML is installed
    if args.reorder_method == REORDER_QML and qmllib is None:
        print(
            "'qmllib' is not installed. Package is avaliable from: github.com/qmlcode/qmllib or pip install qmllib."
        )
        sys.exit(1)

    return args


def example_file(args: Optional[List[str]] = None) -> str:
    # Parse arguments
    settings = parse_arguments(args)

    # Define the read function
    if settings.format == FORMAT_XYZ:
        get_coordinates = partial(
            get_coordinates_xyz,
            is_gzip=settings.format_is_gzip,
            return_atoms_as_int=True,
        )

    elif settings.format == FORMAT_PDB:
        get_coordinates = partial(
            get_coordinates_pdb,
            is_gzip=settings.format_is_gzip,
            return_atoms_as_int=True,
            only_alpha_carbon=settings.only_alpha_carbons,
        )
    else:
        print(f"Unknown format: {settings.format}")
        sys.exit(1)

    # As default, load the extension as format
    # Parse pdb.gz and xyz.gz as pdb and xyz formats
    p_atoms, p_coord = get_coordinates(
        settings.structure_a,
    )

    q_atoms, q_coord = get_coordinates(
        settings.structure_b,
    )

    p_size = p_coord.shape[0]
    q_size = q_coord.shape[0]

    if not p_size == q_size:
        print("error: Structures not same size")
        sys.exit()

    if np.count_nonzero(p_atoms != q_atoms) and not settings.reorder:
        msg = """
        error: Atoms are not in the same order.

        Use --reorder to align the atoms (can be expensive for large structures).

        Please see --help or documentation for more information or
        https://github.com/charnley/rmsd for further examples.
        """
        print(msg)
        sys.exit()

    # Typing
    index: Union[Set[int], List[int], ndarray]

    # Set local view
    p_view: Optional[ndarray] = None
    q_view: Optional[ndarray] = None
    use_view: bool = True

    if settings.ignore_hydrogen:
        (p_view,) = np.where(p_atoms != 1)
        (q_view,) = np.where(q_atoms != 1)

    elif settings.remove_idx:
        index = np.array(list(set(range(p_size)) - set(settings.remove_idx)))
        p_view = index
        q_view = index

    elif settings.add_idx:
        p_view = settings.add_idx
        q_view = settings.add_idx

    else:
        use_view = False

    # Set local view
    if use_view:
        p_coord_sub = copy.deepcopy(p_coord[p_view])
        q_coord_sub = copy.deepcopy(q_coord[q_view])
        p_atoms_sub = copy.deepcopy(p_atoms[p_view])
        q_atoms_sub = copy.deepcopy(q_atoms[q_view])

    else:
        p_coord_sub = copy.deepcopy(p_coord)
        q_coord_sub = copy.deepcopy(q_coord)
        p_atoms_sub = copy.deepcopy(p_atoms)
        q_atoms_sub = copy.deepcopy(q_atoms)

    # Recenter to centroid
    p_cent_sub = centroid(p_coord_sub)
    q_cent_sub = centroid(q_coord_sub)
    p_coord_sub -= p_cent_sub
    q_coord_sub -= q_cent_sub

    rmsd_method: RmsdCallable
    reorder_method: Optional[ReorderCallable]

    # set rotation method
    if settings.rotation == METHOD_KABSCH:
        rmsd_method = kabsch_rmsd
    elif settings.rotation == METHOD_QUATERNION:
        rmsd_method = quaternion_rmsd
    else:
        rmsd_method = compute_rmsd

    # set reorder method
    reorder_method = None
    if settings.reorder_method == REORDER_QML:
        reorder_method = reorder_similarity
    elif settings.reorder_method == REORDER_HUNGARIAN:
        reorder_method = reorder_hungarian
    elif settings.reorder_method == REORDER_INERTIA_HUNGARIAN:
        reorder_method = reorder_inertia_hungarian
    elif settings.reorder_method == REORDER_BRUTE:
        reorder_method = reorder_brute  # pragma: no cover
    elif settings.reorder_method == REORDER_DISTANCE:
        reorder_method = reorder_distance

    # Save the resulting RMSD
    result_rmsd: Optional[float] = None

    # Collect changes to be done on q coords
    q_swap = None
    q_reflection = None
    q_review = None

    if settings.use_reflections:
        result_rmsd, q_swap, q_reflection, q_review = check_reflections(
            p_atoms_sub,
            q_atoms_sub,
            p_coord_sub,
            q_coord_sub,
            reorder_method=reorder_method,
            rmsd_method=rmsd_method,
        )

    elif settings.use_reflections_keep_stereo:
        result_rmsd, q_swap, q_reflection, q_review = check_reflections(
            p_atoms_sub,
            q_atoms_sub,
            p_coord_sub,
            q_coord_sub,
            reorder_method=reorder_method,
            rmsd_method=rmsd_method,
            keep_stereo=True,
        )

    elif settings.reorder:
        assert reorder_method is not None, (
            "Cannot reorder without selecting --reorder method"
        )
        q_review = reorder_method(p_atoms_sub, q_atoms_sub, p_coord_sub, q_coord_sub)

    # If there is a reorder, then apply before print
    if q_review is not None:
        q_atoms_sub = q_atoms_sub[q_review]
        q_coord_sub = q_coord_sub[q_review]

        assert all(p_atoms_sub == q_atoms_sub), (
            "error: Structure not aligned. Please submit bug report at http://github.com/charnley/rmsd"
        )

    # Calculate the RMSD value
    if result_rmsd is None:
        result_rmsd = rmsd_method(p_coord_sub, q_coord_sub)

    # print result
    if settings.output:
        if q_swap is not None:
            q_coord_sub = q_coord_sub[:, q_swap]

        if q_reflection is not None:
            q_coord_sub = np.dot(q_coord_sub, np.diag(q_reflection))

        U = kabsch(q_coord_sub, p_coord_sub)

        if settings.print_only_rmsd_atoms or not use_view:
            q_coord_sub = np.dot(q_coord_sub, U)
            q_coord_sub += p_cent_sub
            return set_coordinates(
                q_atoms_sub,
                q_coord_sub,
                title=f"Rotated '{settings.structure_b}' to match '{settings.structure_a}', with a RMSD of {result_rmsd:.8f}",
            )

        # Swap, reflect, rotate and re-center on the full atom and coordinate set
        q_coord -= q_cent_sub

        if q_swap is not None:
            q_coord = q_coord[:, q_swap]

        if q_reflection is not None:
            q_coord = np.dot(q_coord, np.diag(q_reflection))

        q_coord = np.dot(q_coord, U)
        q_coord += p_cent_sub
        return set_coordinates(
            q_atoms,
            q_coord,
            title=f"Rotated {settings.structure_b} to match {settings.structure_a}, with RMSD of {result_rmsd:.8f}",
        )

    return str(result_rmsd)


def example_numpy():
    print("Running RMSD calculation from NumPy inputs:")

    # Example 1: Two water molecules, slightly translated
    # Molecule P
    p_atoms1 = ["O", "H", "H"]
    p_coord1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.957], [0.0, 0.926, -0.239]])

    # Molecule Q (translated version of P)
    q_atoms1 = ["O", "H", "H"]
    q_coord1 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1.457], [0.5, 1.426, 0.261]])

    print("\n--- Example 1: Simple translation (no reordering) ---")
    rmsd1 = rmsd_from_numpy(p_atoms1, p_coord1, q_atoms1, q_coord1)
    print(f"RMSD (P vs Q, no reorder): {rmsd1:.4f}")

    # Example 2: Two water molecules, one with atoms reordered
    # Molecule P (same as p_atoms1, p_coord1)

    # Molecule Q_reordered (same coordinates as p_coord1 but different atom order)
    # Original order: O, H1, H2
    # New order: H1, H2, O
    q_atoms2_reordered = ["H", "H", "O"]  # Intentionally different order
    q_coord2_reordered = np.array(
        [[0.0, 0.0, 0.957], [0.0, 0.926, -0.239], [0.0, 0.0, 0.0]]  # H1  # H2  # O
    )

    print("\n--- Example 2: Atom order mismatch ---")
    try:
        rmsd2_fail = rmsd_from_numpy(
            p_atoms1, p_coord1, q_atoms2_reordered, q_coord2_reordered
        )
        print(
            f"RMSD (P vs Q_reordered, no reorder): {rmsd2_fail:.4f} (Should ideally raise error or be high)"
        )
    except ValueError as e:
        print(f"RMSD (P vs Q_reordered, no reorder) failed as expected: {e}")

    print("\n--- Example 3: Atom order mismatch with Hungarian reordering ---")
    # Using atom numbers for Q
    q_atoms2_numbers_reordered = [1, 1, 8]  # H, H, O
    rmsd2_reorder = rmsd_from_numpy(
        p_atoms1,
        p_coord1,
        q_atoms2_numbers_reordered,
        q_coord2_reordered,  # Use the reordered Q
        reorder=True,
        reorder_method_str=REORDER_HUNGARIAN,
    )
    print(f"RMSD (P vs Q_reordered, with Hungarian reorder): {rmsd2_reorder:.4f}")

    # Example 4: Using atomic numbers and reflections
    p_atoms_num = np.array([8, 1, 1])
    p_coord_reflect = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.957], [0.0, 0.926, -0.239]]
    )
    q_atoms_num = np.array([8, 1, 1])
    # Mirrored Q coordinates (e.g. reflected across xy plane by z -> -z)
    q_coord_reflect = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.957], [0.0, 0.926, 0.239]]
    )
    print("\n--- Example 4: Reflected coordinates, check reflections (keep stereo) ---")
    rmsd_reflect = rmsd_from_numpy(
        p_atoms_num,
        p_coord_reflect,
        q_atoms_num,
        q_coord_reflect,
        reorder=False,  # Atoms are already in order
        use_reflections_keep_stereo=True,
    )
    print(
        f"RMSD (P_reflect vs Q_reflect, with reflections_keep_stereo): {rmsd_reflect:.4f}"
    )

    # Example 5: Benzene-like structures, one rotated and atoms permuted
    # Structure 1 (planar hexagon)
    p_atoms_benzene = ["C"] * 6 + ["H"] * 6
    p_coord_benzene = (
        np.array(
            [
                [1.395, 0.000, 0.000],
                [-0.697, 1.208, 0.000],
                [-0.697, -1.208, 0.000],  # C
                [0.697, -1.208, 0.000],
                [-1.395, 0.000, 0.000],
                [0.697, 1.208, 0.000],  # C
                [2.475, 0.000, 0.000],
                [-1.237, 2.143, 0.000],
                [-1.237, -2.143, 0.000],  # H
                [1.237, -2.143, 0.000],
                [-2.475, 0.000, 0.000],
                [1.237, 2.143, 0.000],  # H
            ]
        )
        * 0.7
    )  # scale down a bit

    # Structure 2 (rotated and permuted version of structure 1)
    q_atoms_benzene = ["C"] * 6 + ["H"] * 6
    # Permute carbons, permute hydrogens, then concatenate
    perm_c = np.random.permutation(6)
    perm_h = np.random.permutation(6) + 6

    rotation_angle = np.pi / 4  # 45 degrees
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    q_coord_benzene_rotated = p_coord_benzene @ rotation_matrix.T
    q_coord_benzene = np.concatenate(
        (q_coord_benzene_rotated[perm_c], q_coord_benzene_rotated[perm_h])
    )

    print("\n--- Example 5: Benzene-like structures, reordered and rotated ---")
    rmsd_benzene_no_reorder = rmsd_from_numpy(
        p_atoms_benzene,
        p_coord_benzene,
        q_atoms_benzene,
        q_coord_benzene,
        reorder=False,
    )
    print(
        f"RMSD (Benzene, no reorder): {rmsd_benzene_no_reorder:.4f} (expected to be high)"
    )

    rmsd_benzene_reorder = rmsd_from_numpy(
        p_atoms_benzene,
        p_coord_benzene,
        q_atoms_benzene,
        q_coord_benzene,
        reorder=True,
        reorder_method_str=REORDER_HUNGARIAN,  # or REORDER_INERTIA_HUNGARIAN
    )
    print(
        f"RMSD (Benzene, with Hungarian reorder): {rmsd_benzene_reorder:.4f} (expected to be low)"
    )

    # Example 6: System of identical, generic particles (e.g., Lennard-Jones)
    print("\n--- Example 6: Identical generic particles (Lennard-Jones like) ---")
    N_particles = 5
    # Structure P: A simple arrangement
    p_lj_coord = np.random.rand(N_particles, 3) * 5.0
    # For identical particles, use the same atomic number for all, e.g., 1 (Hydrogen) or 0.
    # Using 1 is safer if reorder_inertia_hungarian might be used.
    p_lj_atoms = np.ones(N_particles, dtype=int)  # Represent all as type 1

    print(f"Permute only")
    # Structure Q: Structure P permuted
    permutation = np.random.permutation(N_particles)
    q_lj_coord = p_lj_coord[permutation, :]  # Permute only
    q_lj_atoms = np.ones(N_particles, dtype=int)

    # Calculate RMSD without reordering (will likely be high due to permutation)
    rmsd_lj_no_reorder = rmsd_from_numpy(
        p_lj_atoms, p_lj_coord, q_lj_atoms, q_lj_coord, reorder=False
    )
    print(
        f"RMSD (LJ-like, no reorder): {rmsd_lj_no_reorder:.4f} (permute only -> should be high)"
    )

    # Calculate RMSD with Hungarian reordering (should be low)
    rmsd_lj_hungarian = rmsd_from_numpy(
        p_lj_atoms,
        p_lj_coord,
        q_lj_atoms,
        q_lj_coord,
        reorder=True,
        reorder_method_str=REORDER_HUNGARIAN,
    )
    print(
        f"RMSD (LJ-like, Hungarian reorder): {rmsd_lj_hungarian:.4f} (permute only -> should be zero)"
    )

    # Example using atomic number 0 (ensure not to use inertia-hungarian if 0 is not in ELEMENT_WEIGHTS)
    p_lj_atoms_zero = np.zeros(N_particles, dtype=int)
    q_lj_atoms_zero = np.zeros(N_particles, dtype=int)
    rmsd_lj_hungarian_zeros = rmsd_from_numpy(
        p_lj_atoms_zero,
        p_lj_coord,  # Using coords from above for simplicity
        q_lj_atoms_zero,
        q_lj_coord,
        reorder=True,
        reorder_method_str=REORDER_HUNGARIAN,
    )
    print(
        f"RMSD (LJ-like, Hungarian reorder, atoms as 0): {rmsd_lj_hungarian_zeros:.4f} (permute only -> should be zero)"
    )

    # Example 6.5: System of identical, generic particles (e.g., Lennard-Jones)
    print(f"\nPermute and perturb")
    # Structure Q: Structure P permuted and slightly perturbed
    permutation = np.random.permutation(N_particles)
    q_lj_coord = (
        p_lj_coord[permutation, :] + np.random.rand(N_particles, 3) * 0.1
    )  # Permute and perturb
    q_lj_atoms = np.ones(N_particles, dtype=int)

    # Calculate RMSD without reordering (will likely be high due to permutation)
    rmsd_lj_no_reorder = rmsd_from_numpy(
        p_lj_atoms, p_lj_coord, q_lj_atoms, q_lj_coord, reorder=False
    )
    print(
        f"RMSD (LJ-like, no reorder): {rmsd_lj_no_reorder:.4f} (permute and perturb -> should be high)"
    )

    # Calculate RMSD with Hungarian reordering (should be low)
    rmsd_lj_hungarian = rmsd_from_numpy(
        p_lj_atoms,
        p_lj_coord,
        q_lj_atoms,
        q_lj_coord,
        reorder=True,
        reorder_method_str=REORDER_HUNGARIAN,
    )
    print(
        f"RMSD (LJ-like, Hungarian reorder): {rmsd_lj_hungarian:.4f} (permute and perturb -> should be low)"
    )

    # Example using atomic number 0 (ensure not to use inertia-hungarian if 0 is not in ELEMENT_WEIGHTS)
    p_lj_atoms_zero = np.zeros(N_particles, dtype=int)
    q_lj_atoms_zero = np.zeros(N_particles, dtype=int)
    rmsd_lj_hungarian_zeros = rmsd_from_numpy(
        p_lj_atoms_zero,
        p_lj_coord,  # Using coords from above for simplicity
        q_lj_atoms_zero,
        q_lj_coord,
        reorder=True,
        reorder_method_str=REORDER_HUNGARIAN,
    )
    print(
        f"RMSD (LJ-like, Hungarian reorder, atoms as 0): {rmsd_lj_hungarian_zeros:.4f} (permute and perturb -> should be low)"
    )


# def example_rgd1():

#     # load rgd1 dataset

#     # get transition state, reactant, product from prompt

if __name__ == "__main__":
    # result = example_file() # Keep this if you want to run from files via CLI
    # print(result)
    example_numpy()
