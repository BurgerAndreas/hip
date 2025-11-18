# Adapted from pysisyphus

import numpy as np
import scipy.constants as spc
import torch

from hip.masses import MASS_DICT
from hip.ff_lmdb import Z_TO_ATOM_SYMBOL

"""
Adapted from 
dependencies/pysisyphus/pysisyphus/Geometry.py
"""
# from pysisyphus.constants import AU2J, BOHR2ANG, C, R, AU2KJPERMOL, NA
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
HARTREE2EV = spc.physical_constants["Hartree energy in eV"][0]


def inertia_tensor(coords3d, masses):
    """Inertita tensor.

                          | x² xy xz |
    (x y z)^T . (x y z) = | xy y² yz |
                          | xz yz z² |
    """
    x, y, z = coords3d.T
    squares = np.sum(coords3d**2 * masses[:, None], axis=0)
    I_xx = squares[1] + squares[2]
    I_yy = squares[0] + squares[2]
    I_zz = squares[0] + squares[1]
    I_xy = -np.sum(masses * x * y)
    I_xz = -np.sum(masses * x * z)
    I_yz = -np.sum(masses * y * z)
    return np.array(((I_xx, I_xy, I_xz), (I_xy, I_yy, I_yz), (I_xz, I_yz, I_zz)))


def get_trans_rot_vectors(cart_coords, masses, rot_thresh=1e-6):
    """Vectors describing translation and rotation.

    These vectors are used for the Eckart projection by constructing
    a projector from them.

    See Martin J. Field - A Pratcial Introduction to the simulation
    of Molecular Systems, 2007, Cambridge University Press, Eq. (8.23),
    (8.24) and (8.26) for the actual projection.

    See also https://chemistry.stackexchange.com/a/74923.

    Parameters
    ----------
    cart_coords : np.array, 1d, shape (3 * atoms.size, )
        Atomic masses in amu.
    masses : iterable, 1d, shape (atoms.size, )
        Atomic masses in amu.

    Returns
    -------
    ortho_vecs : np.array(6, 3*atoms.size)
        2d array containing row vectors describing translations
        and rotations.
    """

    coords3d = np.reshape(cart_coords, (-1, 3))
    total_mass = masses.sum()
    com = 1 / total_mass * np.sum(coords3d * masses[:, None], axis=0)
    coords3d_centered = coords3d - com[None, :]

    _, Iv = np.linalg.eigh(inertia_tensor(coords3d, masses))
    Iv = Iv.T

    masses_rep = np.repeat(masses, 3)
    sqrt_masses = np.sqrt(masses_rep)  # (3N,)
    num = len(masses)

    def get_trans_vecs():
        """Mass-weighted unit vectors of the three cartesian axes."""

        for vec in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            _ = sqrt_masses * np.tile(vec, num)
            yield _ / np.linalg.norm(_)  # (3N,)

    def get_rot_vecs():
        """As done in geomeTRIC."""

        rot_vecs = np.zeros((3, cart_coords.size))
        # p_vecs = Iv.dot(coords3d_centered.T).T
        for i in range(masses.size):
            p_vec = Iv.dot(coords3d_centered[i])
            for ix in range(3):
                rot_vecs[0, 3 * i + ix] = Iv[2, ix] * p_vec[1] - Iv[1, ix] * p_vec[2]
                rot_vecs[1, 3 * i + ix] = Iv[2, ix] * p_vec[0] - Iv[0, ix] * p_vec[2]
                rot_vecs[2, 3 * i + ix] = Iv[0, ix] * p_vec[1] - Iv[1, ix] * p_vec[0]
        rot_vecs *= sqrt_masses[None, :]
        return rot_vecs  # (3, 3N)

    trans_vecs = list(get_trans_vecs())  # (3, 3N)
    rot_vecs = np.array(get_rot_vecs())  # (3, 3N)
    # Drop vectors with vanishing norms
    rot_vecs = rot_vecs[np.linalg.norm(rot_vecs, axis=1) > rot_thresh]
    tr_vecs = np.concatenate((trans_vecs, rot_vecs), axis=0)  # (6, 3N)
    tr_vecs = np.linalg.qr(tr_vecs.T)[0].T
    return tr_vecs  # (6, 3N)


def get_trans_rot_projector(cart_coords, masses, full=False):
    tr_vecs = get_trans_rot_vectors(cart_coords, masses=masses)
    if full:
        # Full projector
        P = np.eye(cart_coords.size)
        for tr_vec in tr_vecs:
            P -= np.outer(tr_vec, tr_vec)
    else:
        # SVD
        U, s, _ = np.linalg.svd(tr_vecs.T)
        P = U[:, s.size :].T
    return P


def mass_weigh_hessian(hessian, masses3d):
    """mass-weighted hessian M^(-1/2) H M^(-1/2)
    Inverted square root of the mass matrix."""
    mm_sqrt_inv = np.diag(1 / (masses3d**0.5))
    return mm_sqrt_inv.dot(hessian).dot(mm_sqrt_inv)


def unweight_mw_hessian(mw_hessian, masses3d):
    """Unweight a mass-weighted hessian.
    Mass-weighted hessian to be unweighted
    ->
    2d array containing the hessian.
    """
    mm_sqrt = np.diag(masses3d**0.5)
    return mm_sqrt.dot(mw_hessian).dot(mm_sqrt)


def massweigh_and_eckartprojection_np(hessian, cart_coords, atomsymbols):
    """Do Eckart projection starting from not-mass-weighted Hessian.
    hessian: np.array (N*3, N*3)
    cart_coords: np.array (N*3)
    atomsymbols: list[str] (N)
    """
    masses = np.array([MASS_DICT[atom.lower()] for atom in atomsymbols])
    masses3d = np.repeat(masses, 3)
    mw_hessian = mass_weigh_hessian(hessian, masses3d)
    P = get_trans_rot_projector(cart_coords, masses=masses, full=False)
    proj_hessian = P.dot(mw_hessian).dot(P.T)
    # Projection seems to slightly break symmetry (sometimes?). Resymmetrize.
    return (proj_hessian + proj_hessian.T) / 2


def eigval_to_wavenumber(ev):
    """Convert Hessian eigenvalues (Hartree/amu/bohr^2) to wavenumbers (cm^-1)."""
    # This approach seems numerically more unstable
    # conv = AU2J / (AMU2KG * BOHR2M ** 2) / (2 * np.pi * 3e10)**2
    # w2nu = np.sign(ev) * np.sqrt(np.abs(ev) * conv)
    # The two lines below are adopted from Psi4 and seem more stable,
    # compared to the approach above.
    conv = np.sqrt(NA * AU2J * 1.0e19) / (2 * np.pi * C * BOHR2ANG)
    w2nu = np.sign(ev) * np.sqrt(np.abs(ev)) * conv
    return w2nu


def eigval_to_vibfreq(ev):
    """Convert mass-weighted Hessian eigenvalues in eV to vibrational frequencies (cm^-1)."""
    ev_au = np.asarray(ev) / HARTREE2EV
    return eigval_to_wavenumber(ev_au)


def analyze_frequencies_np(
    hessian: np.ndarray | str,  # Hartree/Bohr^2
    cart_coords: np.ndarray,  # Bohr
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
):
    proj_hessian = massweigh_and_eckartprojection_np(hessian, cart_coords, atomsymbols)
    eigvals, eigvecs = np.linalg.eigh(proj_hessian)
    sorted_inds = np.argsort(eigvals)
    eigvals = eigvals[sorted_inds]
    eigvecs = eigvecs[:, sorted_inds]

    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = sum(neg_inds)
    # eigval_str = np.array2string(eigvals[:10], precision=4)
    if neg_num > 0:
        wavenumbers = eigval_to_wavenumber(neg_eigvals)
        # wavenum_str = np.array2string(wavenumbers, precision=2)
    else:
        wavenumbers = None
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "wavenumbers": wavenumbers,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(atomsymbols),
    }


#################################################################################
# Torch version
#################################################################################


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


def massweigh_hessian_torch(hessian, masses3d):
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


def massweigh_and_eckartprojection_torch(
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

    mw_hessian_t = massweigh_hessian_torch(hessian, masses3d_t)
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

    proj_hessian = massweigh_and_eckartprojection_torch(
        hessian, cart_coords, atomsymbols
    )
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


if __name__ == "__main__":
    # Minimal self-test comparing numpy and torch implementations
    rng = np.random.default_rng(0)
    atoms = ["H", "O", "H", "C"]
    nat = len(atoms)
    n3 = 3 * nat
    coords3d = rng.normal(size=(nat, 3))
    A = rng.normal(size=(n3, n3))
    hessian = (A + A.T) / 2.0

    print("coords3d.shape", coords3d.shape)
    print("hessian.shape", hessian.shape)

    # 1) Test mass weighting directly
    masses = np.array([MASS_DICT[a.lower()] for a in atoms])
    masses3d = np.repeat(masses, 3)
    mw_np = mass_weigh_hessian(hessian.copy(), masses3d)
    mw_torch = massweigh_hessian_torch(hessian.copy(), masses3d)
    mw_torch_np = mw_torch.detach().cpu().numpy()
    ok_mw = np.allclose(mw_np, mw_torch_np, rtol=1e-6, atol=1e-8)
    print(
        f"mass_weigh_hessian match: {ok_mw}, max diff: {np.max(np.abs(mw_np - mw_torch_np)):.3e}"
    )
    assert ok_mw

    # 2) Test trans/rot vectors subspace (compare projectors onto the span)
    cc = coords3d.reshape(-1)
    tr_np = get_trans_rot_vectors(cc, masses)
    tr_t = get_trans_rot_vectors_torch(cc, masses)
    tr_t_np = tr_t.detach().cpu().numpy()
    # Build projectors P = V^T V where rows are orthonormal vectors
    P_np = tr_np.T @ tr_np
    P_t = tr_t_np.T @ tr_t_np
    ok_tr = np.allclose(P_np, P_t, rtol=1e-6, atol=1e-8)
    print(
        f"trans/rot subspace projector match: {ok_tr}, max diff: {np.max(np.abs(P_np - P_t)):.3e}"
    )
    assert ok_tr

    # 3) Test full Eckart pipeline via eigenvalues of projected Hessian
    np_proj = massweigh_and_eckartprojection_np(hessian.copy(), cc.copy(), atoms)
    torch_proj = massweigh_and_eckartprojection_torch(hessian.copy(), cc.copy(), atoms)
    torch_proj_np = torch_proj.detach().cpu().numpy()
    ok_proj = np.allclose(np_proj, torch_proj_np, rtol=1e-6, atol=1e-8)
    print(
        f"Eckart projected matrix match: {ok_proj}, max diff: {np.max(np.abs(np_proj - torch_proj_np)):.3e}"
    )

    evals_np, _ = np.linalg.eigh(np_proj)
    evals_t, _ = np.linalg.eigh(torch_proj_np)
    ok_eigs = np.allclose(evals_np, evals_t, rtol=1e-6, atol=1e-8)
    print(
        f"Eckart projected eigvals match: {ok_eigs}, max diff: {np.max(np.abs(evals_np - evals_t)):.3e}"
    )
    assert ok_eigs
