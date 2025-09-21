from __future__ import division
from __future__ import print_function
import numpy as np

from collections import OrderedDict, namedtuple, Counter
import logging
import sys

# taken from
# geometric/normal_modes.py
# from geometric.molecule import Molecule, PeriodicTable
# from geometric.nifty import logger, kb, kb_si, hbar, au2kj, au2kcal, ang2bohr, bohr2ang, c_lightspeed, avogadro, cm2au, amu2au, ambervel2au, wq_wait, getWorkQueue, commadash, bak

from ocpmodels.units import PeriodicTable, bohr2ang, au2kj, c_lightspeed


class FrequencyError(Exception):
    pass


class RawStreamHandler(logging.StreamHandler):
    """
    Exactly like StreamHandler, except no newline character is printed at the end of each message.
    This is done in order to ensure functions in molecule.py and nifty.py work consistently
    across multiple packages.
    """

    def __init__(self, stream=sys.stdout):
        super(RawStreamHandler, self).__init__(stream)
        self.terminator = ""


class RawFileHandler(logging.FileHandler):
    """
    Exactly like FileHandler, except no newline character is printed at the end of each message.
    This is done in order to ensure functions in molecule.py and nifty.py work consistently
    across multiple packages.
    """

    def __init__(self, *args, **kwargs):
        super(RawFileHandler, self).__init__(*args, **kwargs)
        self.terminator = ""


logger = logging.getLogger("NiftyLogger")
logger.setLevel(logging.INFO)
handler = RawStreamHandler()
logger.addHandler(handler)
if __name__ == "__main__":
    package = "LPW-nifty.py"
else:
    package = __name__.split(".")[0]


def frequency_analysis(
    coords,
    Hessian,
    elem=None,
    mass=None,
    energy=0.0,
    temperature=300.0,
    pressure=1.0,
    verbose=0,
    outfnm=None,
    note=None,
    wigner=None,
    ignore=0,
    normalized=True,
):
    """
    Parameters
    ----------
    coords : np.array
        n_atoms*3 length array containing coordinates in bohr
    Hessian : np.array
        (n_atoms*3)*(n_atoms*3) length array containing Hessian elements in au
        (i.e. Hessian/bohr^2), the typical units output by QM calculations
    elem : list
        n_atoms length list containing atomic symbols.
        Used in printing displacements and for looking up masses if mass = None.
    mass : list or np.array
        n_atoms length list or 1D array containing atomic masses in amu.
        If provided, masses will not be looked up using elem.
        If neither mass nor elem will be provided, will assume masses are all unity.
    energy : float
        Electronic energy passed to the harmonic free energy module
    temperature : float
        Temperature passed to the harmonic free energy module
    pressure : float
        Pressure passed to the harmonic free energy module
    verbose : int
        Print debugging info
    outfnm : str
        If provided, write vibrational data to a ForceBalance-parsable vdata.txt file
    note : str
        If provided, write a note into the comment line of the xyz structure in vdata.txt
    wigner : tuple
        If provided, should be a 2-tuple containing (nSamples, dirname)
        containing the output folder and number of samples and the output folder
        to which samples should be written
    ignore : int
        Ignore the free energy contributions from the lowest N vibrational modes
        (including negative force constants if there are any).
    normalized : bool
        If True, normalize the un-mass-weighted Cartesian displacements of each normal mode (default)
        If False, return the un-normalized vectors (necessary for IR and Raman intensities)

    Returns
    -------
    freqs_wavenumber : np.array
        n_vibmodes length array containing vibrational frequencies in wavenumber
        (imaginary frequencies are reported as negative)
    normal_modes_cart : np.array
        n_vibmodes*n_atoms length array containing un-mass-weighted Cartesian displacements
        of each normal mode
    """
    # Create a copy of coords and reshape into a 2D array
    coords = coords.copy().reshape(-1, 3)
    na = coords.shape[0]
    if mass is not None:
        mass = np.array(mass)
        assert len(mass) == na
    elif elem:
        mass = np.array([PeriodicTable[j] for j in elem])
        assert len(elem) == na
    else:
        logger.warning("neither elem nor mass is provided; assuming all masses unity")
        mass = np.ones(na)
    assert coords.shape == (na, 3)
    assert Hessian.shape == (3 * na, 3 * na)

    # Convert Hessian eigenvalues into wavenumbers:
    #
    # omega = sqrt(k/m)
    #
    # X hartree bohr^-2 amu^-1 * 2625.5 (kJ mol^-1 / hartree) * (1/0.0529 bohr/nm)^2
    # --> omega^2 = 938211*X ps^-2
    #
    # Frequencies in units of inverse ps:
    # nu = sqrt(938211*X)/2*pi
    #
    # Convert to inverse wavelength in units of cm^-1:
    #
    # 1/lambda = nu/c = (sqrt(938211*X)/2*pi) / (2.998e8 m/s) * (m/100cm) * 10^12ps/s

    bohr2nm = bohr2ang / 10
    mwHess_wavenumber = 1e10 * np.sqrt(au2kj / bohr2nm**2) / (2 * np.pi * c_lightspeed)

    TotDOF = 3 * na
    # Compute the mass weighted Hessian matrix
    # Each element is H[i, j] / sqrt(m[i]) / sqrt(m[j])
    invsqrtm3 = 1.0 / np.sqrt(np.repeat(mass, 3))
    wHessian = Hessian.copy() * np.outer(invsqrtm3, invsqrtm3)

    if verbose >= 2:
        # Eigenvalues before projection of translation and rotation
        logger.info("Eigenvalues before projection of translation and rotation\n")
        w_eigvals = np.linalg.eigvalsh(wHessian)
        for i in range(TotDOF):
            val = mwHess_wavenumber * np.sqrt(abs(w_eigvals[i]))
            logger.info("%5i % 10.3f\n" % (i, val))

    # =============================================#
    # | Remove rotational and translational modes |#
    # =============================================#

    # Compute the center of mass
    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0) / np.sum(mass)

    # Coordinates in the center-of-mass frame
    xcm = coords - cxyz[np.newaxis, :]

    # Moment of inertia tensor
    I = np.sum(
        [
            mass[i] * (np.eye(3) * (np.dot(xcm[i], xcm[i])) - np.outer(xcm[i], xcm[i]))
            for i in range(na)
        ],
        axis=0,
    )

    # Principal moments
    Ivals, Ivecs = np.linalg.eigh(I)
    # Eigenvectors are in the rows after transpose
    Ivecs = Ivecs.T

    # Obtain the number of rotational degrees of freedom
    RotDOF = 0
    for i in range(3):
        if abs(Ivals[i]) > 1.0e-10:
            RotDOF += 1
    TR_DOF = 3 + RotDOF
    if TR_DOF not in (5, 6):
        raise FrequencyError(
            "Unexpected number of trans+rot DOF: %i not in (5, 6)" % TR_DOF
        )

    if verbose >= 2:
        logger.info(
            "Center of mass: % .12f % .12f % .12f\n" % (cxyz[0], cxyz[1], cxyz[2])
        )
        logger.info("Moment of inertia tensor:\n")
        for i in range(3):
            logger.info("   % .12f % .12f % .12f\n" % (I[i, 0], I[i, 1], I[i, 2]))
        logger.info("Principal moments of inertia:\n")
        for i in range(3):
            logger.info(
                "Eigenvalue = %.12f   Eigenvector = % .12f % .12f % .12f\n"
                % (Ivals[i], Ivecs[i, 0], Ivecs[i, 1], Ivecs[i, 2])
            )
        logger.info("Translational-Rotational degrees of freedom: %i\n" % TR_DOF)

    # Internal coordinates of the Eckart frame
    ic_eckart = np.zeros((6, TotDOF))
    for i in range(na):
        # The dot product of (the coordinates of the atoms with respect to the center of mass) and
        # the corresponding row of the matrix used to diagonalize the moment of inertia tensor
        p_vec = np.dot(Ivecs, xcm[i])
        smass = np.sqrt(mass[i])
        ic_eckart[0, 3 * i] = smass
        ic_eckart[1, 3 * i + 1] = smass
        ic_eckart[2, 3 * i + 2] = smass
        for ix in range(3):
            ic_eckart[3, 3 * i + ix] = smass * (
                Ivecs[2, ix] * p_vec[1] - Ivecs[1, ix] * p_vec[2]
            )
            ic_eckart[4, 3 * i + ix] = smass * (
                Ivecs[2, ix] * p_vec[0] - Ivecs[0, ix] * p_vec[2]
            )
            ic_eckart[5, 3 * i + ix] = smass * (
                Ivecs[0, ix] * p_vec[1] - Ivecs[1, ix] * p_vec[0]
            )

    if verbose >= 2:
        logger.info("Coordinates in Eckart frame:\n")
        for i in range(ic_eckart.shape[0]):
            for j in range(ic_eckart.shape[1]):
                logger.info(" % .12f " % ic_eckart[i, j])
            logger.info("\n")

    # Sort the rotation ICs by their norm in descending order, then normalize them
    ic_eckart_norm = np.sqrt(np.sum(ic_eckart**2, axis=1))
    # If the norm is equal to zero, then do not scale.
    ic_eckart_norm += ic_eckart_norm == 0.0
    sortidx = np.concatenate(
        (np.array([0, 1, 2]), 3 + np.argsort(ic_eckart_norm[3:])[::-1])
    )
    ic_eckart1 = ic_eckart[sortidx, :]
    ic_eckart1 /= ic_eckart_norm[sortidx, np.newaxis]
    ic_eckart = ic_eckart1.copy()

    if verbose >= 2:
        logger.info("Eckart frame basis vectors:\n")
        for i in range(ic_eckart.shape[0]):
            for j in range(ic_eckart.shape[1]):
                logger.info(" % .12f " % ic_eckart[i, j])
            logger.info("\n")

    # Using Gram-Schmidt orthogonalization, create a basis where translation
    # and rotation is projected out of Cartesian coordinates
    proj_basis = np.identity(TotDOF)
    maxIt = 100
    for iteration in range(maxIt):
        max_overlap = 0.0
        for i in range(TotDOF):  # Loop through all 3N basis vectors
            # For each basis vector, project out the TR_DOF (5-6) translational/rotational modes
            for n in range(TR_DOF):
                # Subtract the projection of the current basis vector onto the TR basis vectors
                proj_basis[i] -= np.dot(ic_eckart[n], proj_basis[i]) * ic_eckart[n]
            # Compute the overlap of the current basis vector with the TR basis vectors
            overlap = np.sum(np.dot(ic_eckart, proj_basis[i]))
            max_overlap = max(overlap, max_overlap)
        if verbose >= 2:
            logger.info("Gram-Schmidt Iteration %i: % .12f\n" % (iteration, overlap))
        if max_overlap < 1e-12:
            break
        if iteration == maxIt - 1:
            raise FrequencyError(
                "Gram-Schmidt orthogonalization failed after %i iterations" % maxIt
            )

    # Diagonalize the overlap matrix to create (3N-6) orthonormal basis vectors
    # constructed from translation and rotation-projected proj_basis
    proj_overlap = np.dot(proj_basis, proj_basis.T)
    if verbose >= 3:
        logger.info("Overlap matrix:\n")
        for i in range(proj_overlap.shape[0]):
            for j in range(proj_overlap.shape[1]):
                logger.info(" % .12f " % proj_overlap[i, j])
            logger.info("\n")
    proj_vals, proj_vecs = np.linalg.eigh(proj_overlap)
    proj_vecs = proj_vecs.T
    if verbose >= 3:
        logger.info("Eigenvectors of overlap matrix:\n")
        for i in range(proj_vecs.shape[0]):
            for j in range(proj_vecs.shape[1]):
                logger.info(" % .12f " % proj_vecs[i, j])
            logger.info("\n")

    # Make sure number of vanishing eigenvalues is roughly equal to TR_DOF
    numzero_upper = np.sum(
        abs(proj_vals) < 1.0e-8
    )  # Liberal counting of zeros - should be more than TR_DOF
    numzero_lower = np.sum(
        abs(proj_vals) < 1.0e-12
    )  # Conservative counting of zeros - should be less than TR_DOF
    if numzero_upper == TR_DOF and numzero_lower == TR_DOF:
        if 0:
            logger.info("Expected number of vanishing eigenvalues: %i\n" % TR_DOF)
    elif numzero_upper < TR_DOF:
        raise FrequencyError(
            "Not enough vanishing eigenvalues: %i < %i (detected < expected)"
            % (numzero_upper, TR_DOF)
        )
    elif numzero_lower > TR_DOF:
        raise FrequencyError(
            "Too many vanishing eigenvalues: %i > %i (detected > expected)"
            % (numzero_lower, TR_DOF)
        )
    else:
        logger.warning(
            "Eigenvalues near zero: N(<1e-12) = %i, N(1e-12-1e-8) = %i Expected = %i\n"
            % (numzero_lower, numzero_upper, TR_DOF)
        )

    # Construct eigenvectors of unit length in the space of Cartesian displacements
    VibDOF = TotDOF - TR_DOF
    norm_vecs = proj_vecs[TR_DOF:] / np.sqrt(proj_vals[TR_DOF:, np.newaxis])

    if verbose >= 3:
        logger.info("Coefficients of Gram-Schmidt orthogonalized vectors:\n")
        for i in range(norm_vecs.shape[0]):
            for j in range(norm_vecs.shape[1]):
                logger.info(" % .12f " % norm_vecs[i, j])
            logger.info("\n")

    # These are the orthonormal, TR-projected internal coordinates
    ic_basis = np.dot(norm_vecs, proj_basis)

    # ===========================================#
    # | Calculate frequencies and displacements |#
    # ===========================================#

    # Calculate the internal coordinate Hessian and diagonalize
    ic_hessian = np.linalg.multi_dot((ic_basis, wHessian, ic_basis.T))
    ichess_vals, ichess_vecs = np.linalg.eigh(ic_hessian)
    ichess_vecs = ichess_vecs.T
    normal_modes = np.dot(ichess_vecs, ic_basis)

    # Undo mass weighting to get Cartesian displacements
    normal_modes_cart = normal_modes * invsqrtm3[np.newaxis, :]
    if normalized:
        normal_modes_cart /= np.linalg.norm(normal_modes_cart, axis=1)[:, np.newaxis]

    # Convert IC Hessian eigenvalues to wavenumbers
    freqs_wavenumber = (
        mwHess_wavenumber * np.sqrt(np.abs(ichess_vals)) * np.sign(ichess_vals)
    )

    if verbose:
        logger.info(
            "\n-=# Vibrational Frequencies (wavenumber) and Cartesian displacements #=-\n\n"
        )
        i = 0
        while True:
            j = min(i + 3, VibDOF)
            for k in range(i, j):
                logger.info("  Frequency(cm^-1): % 12.6f     " % freqs_wavenumber[k])
            logger.info("\n")
            for k in range(i, j):
                logger.info("--------------------------------     ")
            logger.info("\n")
            for n in range(na):
                for k in range(i, j):
                    if elem:
                        logger.info("%-2s " % elem[n])
                    else:
                        logger.info("   ")
                    logger.info("% 9.6f " % normal_modes_cart[k, 3 * n])
                    logger.info("% 9.6f " % normal_modes_cart[k, 3 * n + 1])
                    logger.info("% 9.6f " % normal_modes_cart[k, 3 * n + 2])
                    logger.info("    ")
                logger.info("\n")
            if i + 3 >= VibDOF:
                break
            logger.info("\n")
            i += 3

    return freqs_wavenumber, normal_modes_cart
