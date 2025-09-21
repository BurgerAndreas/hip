from typing import Union, Callable

# import copy

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import LSODA
from ase import Atoms
from ase.utils import basestring
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from time import perf_counter

from sella.utilities.math import modified_gram_schmidt
from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.eigensolvers import rayleigh_ritz
from sella.internal import Internals, Constraints, DuplicateInternalError


def _timed_method(method):
    name = method.__qualname__

    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_time_stats"):
            self._time_stats = {}
        start = perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            self._time_stats[name] = self._time_stats.get(name, 0.0) + elapsed

    return wrapper


def copy_atoms(atoms: Atoms) -> Atoms:
    """
    Simple function to copy an atoms object to prevent mutability.
    """
    calc = atoms.calc
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms


class PES:
    """Wrapper around an ASE `Atoms` object that exposes a Potential
    Energy Surface (PES) API.

    Responsibilities:
    - evaluate energy and gradient with constraints
    - maintain and update an approximate Hessian
    - manage constraint bases and projections
    - provide trust-region step evaluation via `kick`
    """

    def __init__(
        self,
        atoms: Atoms,
        H0: np.ndarray = None,
        constraints: Constraints = None,
        eigensolver: str = "jd0",
        trajectory: Union[str, Trajectory] = None,
        eta: float = 1e-4,
        v0: np.ndarray = None,
        proj_trans: bool = None,
        proj_rot: bool = None,
        hessian_function: Callable[[Atoms], np.ndarray] = None,
    ) -> None:
        self.atoms = atoms
        if constraints is None:
            constraints = Constraints(self.atoms)
        if proj_trans is None:
            if constraints.internals["translations"]:
                proj_trans = False
            else:
                proj_trans = True
        if proj_trans:
            try:
                constraints.fix_translation()
            except DuplicateInternalError:
                pass

        if proj_rot is None:
            if np.any(atoms.pbc):
                proj_rot = False
            else:
                proj_rot = True
        if proj_rot:
            try:
                constraints.fix_rotation()
            except DuplicateInternalError:
                pass
        self.cons = constraints
        self.eigensolver = eigensolver

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                self.traj = Trajectory(trajectory, "w", self.atoms)
            else:
                self.traj = trajectory
        else:
            self.traj = None

        self.eta = eta
        self.v0 = v0

        self.neval = 0
        self.curr = dict(
            x=None,
            f=None,
            g=None,
        )
        self.last = self.curr.copy()

        # Internal coordinate specific things
        self.int = None
        self.dummies = None

        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        if H0 is None:
            self.set_H(None, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        self.savepoint = dict(apos=None, dpos=None)
        self.first_diag = True

        self.hessian_function = hessian_function

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def save(self):
        self.savepoint = dict(apos=self.apos, dpos=self.dpos)

    def restore(self):
        apos = self.savepoint["apos"]
        dpos = self.savepoint["dpos"]
        assert apos is not None
        self.atoms.positions = apos
        if dpos is not None:
            self.dummies.positions = dpos

    # Position getter/setter
    @_timed_method
    def set_x(self, target):
        diff = target - self.get_x()
        self.atoms.positions = target.reshape((-1, 3))
        return diff, diff, self.curr.get("g", np.zeros_like(diff))

    @_timed_method
    def get_x(self):
        return self.apos.ravel().copy()

    # Hessian getter/setter
    @_timed_method
    def get_H(self):
        return self.H

    @_timed_method
    def set_H(self, target, *args, **kwargs):
        self.H = ApproximateHessian(self.dim, self.ncart, target, *args, **kwargs)

    # Hessian of the constraints
    def get_Hc(self):
        return self.cons.hessian().ldot(self.curr["L"])

    # Hessian of the Lagrangian
    def get_HL(self):
        return self.get_H() - self.get_Hc()

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.residual()

    @_timed_method
    def get_drdx(self):
        return self.cons.jacobian()

    @_timed_method
    def _calc_basis(self):
        drdx = self.get_drdx()
        U, S, VT = np.linalg.svd(drdx)
        ncons = np.sum(S > 1e-6)
        Ucons = VT[:ncons].T
        Ufree = VT[ncons:].T
        Unred = np.eye(self.dim)
        return drdx, Ucons, Unred, Ufree

    @_timed_method
    def write_traj(self):
        if self.traj is not None:
            self.traj.write()

    @_timed_method
    def eval(self):
        self.neval += 1
        f = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        self.write_traj()
        return f, g

    @_timed_method
    def _calc_eg(self, x):
        self.save()
        self.set_x(x)

        f, g = self.eval()

        self.restore()
        return f, g

    @_timed_method
    def get_scons(self):
        """Returns displacement vector for linear constraint correction."""
        Ucons = self.get_Ucons()

        scons = (
            -Ucons
            @ np.linalg.lstsq(
                self.get_drdx() @ Ucons,
                self.get_res(),
                rcond=None,
            )[0]
        )
        return scons

    @_timed_method
    def _update(self, feval=True):
        x = self.get_x()
        new_point = True
        if self.curr["x"] is not None and np.all(x == self.curr["x"]):
            if feval and self.curr["f"] is None:
                new_point = False
            else:
                return False
        drdx, Ucons, Unred, Ufree = self._calc_basis()

        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr["x"] = x
        self.curr["f"] = f
        self.curr["g"] = g
        self._update_basis()
        return True

    @_timed_method
    def _update_basis(self):
        drdx, Ucons, Unred, Ufree = self._calc_basis()
        self.curr["drdx"] = drdx
        self.curr["Ucons"] = Ucons
        self.curr["Unred"] = Unred
        self.curr["Ufree"] = Ufree

        if self.curr["g"] is None:
            L = None
        else:
            L = np.linalg.lstsq(drdx.T, self.curr["g"], rcond=None)[0]
        self.curr["L"] = L

    @_timed_method
    def _update_H(self, dx, dg):
        if self.last["x"] is None or self.last["g"] is None:
            return
        self.H.update(dx, dg)

    @_timed_method
    def get_f(self):
        self._update()
        return self.curr["f"]

    @_timed_method
    def get_g(self):
        self._update()
        return self.curr["g"].copy()

    @_timed_method
    def get_Unred(self):
        self._update(False)
        return self.curr["Unred"]

    @_timed_method
    def get_Ufree(self):
        self._update(False)
        return self.curr["Ufree"]

    def get_Ucons(self):
        self._update(False)
        return self.curr["Ucons"]

    @_timed_method
    def diag(self, gamma=0.1, threepoint=False, maxiter=None):
        """Diagonalize/update the approximate Hessian in the free subspace.

        Optionally uses a three-point finite-difference numerical Hessian
        in the subspace and refines the internal low-rank representation
        via Rayleigh–Ritz, then rotates and updates `self.H`.
        """
        # Ensure energy/gradient and basis information are current
        if self.curr["f"] is None:
            self._update(feval=True)

        # Compute basis of free coordinates (orthogonal to constraints)
        Ufree = self.get_Ufree()
        nfree = Ufree.shape[1]

        # Project Lagrangian Hessian onto free subspace
        P = self.get_HL().project(Ufree)

        # Select initial subspace vector, preferably gradient in free space
        if P.B is None or self.first_diag:
            v0 = self.v0
            if v0 is None:
                v0 = self.get_g() @ Ufree
        else:
            v0 = None

        # Extract overlap matrix; default to identity if none available
        if P.B is None:
            P = np.eye(nfree)
        else:
            P = P.asarray()

        # Build numerical Hessian operator restricted to free subspace
        Hproj = NumericalHessian(
            self._calc_eg, self.get_x(), self.get_g(), self.eta, threepoint, Ufree
        )
        # Constraint Hessian contribution
        Hc = self.get_Hc()
        # Compute Ritz pairs in subspace
        rayleigh_ritz(
            Hproj - Ufree.T @ Hc @ Ufree,
            gamma,
            P,
            v0=v0,
            method=self.eigensolver,
            maxiter=maxiter,
        )

        # Extract eigensolver iterates (subspace basis and its image)
        Vs = Hproj.Vs
        AVs = Hproj.AVs

        # Form symmetric projected operator (with constraint correction)
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2) - Vs.T @ Hc @ Vs
        # Diagonalize in the subspace to obtain Ritz vectors
        _, X = eigh(Atilde)

        # Rotate subspace and images to align with Ritz eigenvectors
        Vs = Vs @ X
        AVs = AVs @ X

        # Update low-rank Hessian approximation with (Vs, AVs)
        self.H.update(Vs, AVs)

        # Mark that initial diagonalization is done
        self.first_diag = False

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        return -((Ufree @ Ufree.T) @ g).reshape((-1, 3))

    def converged(self, fmax, cmax=1e-5):
        fmax1 = np.linalg.norm(self.get_projected_forces(), axis=1).max()
        cmax1 = np.linalg.norm(self.get_res())
        conv = (fmax1 < fmax) and (cmax1 < cmax)
        return conv, fmax1, cmax1

    def wrap_dx(self, dx):
        return dx

    @_timed_method
    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        return g.T @ dx + (dx.T @ H @ dx) / 2.0

    @_timed_method
    def kick(self, dx, diag=False, **diag_kwargs):
        """Apply a trial displacement and update the model.

        - Moves the system by `dx` (respecting coordinate representation).
        - Compares predicted vs actual energy change to form a trust ratio.
        - Updates the quasi-Newton Hessian with the realized secant pair.
        - Optionally re-diagonalizes or computes an exact Hessian if
          `diag=True`.

        Returns
        -------
        ratio : float | None
            Actual/predicted energy change; used by the trust-radius logic.
        """
        # Snapshot current position, energy, gradient, and Hessian model
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H.asarray()

        # Apply step; get intended and realized displacements, and parallel part of gradient
        dx_initial, dx_final, g_par = self.set_x(x0 + dx)

        # Predicted energy change from quadratic model
        df_pred = self.get_df_pred(dx_initial, g0, B0)
        # Actual changes from evaluations at the new point
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        # Trust ratio: actual over predicted decrease (may be None if no model)
        if df_pred is None:
            ratio = None
        else:
            ratio = df_actual / df_pred

        # Update quasi-Newton Hessian with realized secant pair
        self._update_H(dx_final, dg_actual)

        # Optionally (re)diagonalize or compute exact Hessian
        if diag:
            if self.hessian_function is not None:
                self.calculate_hessian()
            else:
                self.diag(**diag_kwargs)

        return ratio

    @_timed_method
    def calculate_hessian(self):
        """Set `self.H` from a user-provided exact Hessian function."""
        assert self.hessian_function is not None
        self.H.set_B(self.hessian_function(self.atoms))

    def print_total_time_per_function(self, top=5):
        """Print cumulative time spent in each PES method for this instance."""
        if not hasattr(self, "_time_stats") or not self._time_stats:
            print("No timing data recorded.")
            return
        for name, total in list(
            sorted(self._time_stats.items(), key=lambda kv: kv[1], reverse=True)
        )[:top]:
            print(f"{name}: {total:.6f}s")


class InternalPES(PES):
    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        H0: np.ndarray = None,
        iterative_stepper: int = 0,
        auto_find_internals: bool = True,
        # Sella adds dummy atoms to handle internal coordinates
        write_dummies_to_traj: bool = True,
        **kwargs,
    ):
        self.int_orig = internals
        new_int = internals.copy()
        if auto_find_internals:
            new_int.find_all_bonds()
            new_int.find_all_angles()
            new_int.find_all_dihedrals()
        new_int.validate_basis()

        PES.__init__(
            self,
            atoms,
            *args,
            constraints=new_int.cons,
            H0=None,
            proj_trans=False,
            proj_rot=False,
            **kwargs,
        )

        self.int = new_int
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof

        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.jacobian()
            P = B @ np.linalg.pinv(B)
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper

        # Andreas
        self.write_dummies_to_traj = write_dummies_to_traj

    dpos = property(lambda self: self.dummies.positions.copy())

    @_timed_method
    def _set_x_iterative(self, target):
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        pos1 = None
        dpos1 = None
        x0 = self.get_x()
        dx_initial = target - x0
        g0 = np.linalg.lstsq(
            self.int.jacobian(),
            self.curr.get("g", np.zeros_like(dx_initial)),
            rcond=None,
        )[0]
        for _ in range(10):
            dx = np.linalg.lstsq(
                self.int.jacobian(),
                self.wrap_dx(target - self.get_x()),
                rcond=None,
            )[0].reshape((-1, 3))
            if np.sqrt((dx**2).sum() / len(dx)) < 1e-6:
                break
            self.atoms.positions += dx[: len(self.atoms)]
            self.dummies.positions += dx[len(self.atoms) :]
            if pos1 is None:
                pos1 = self.atoms.positions.copy()
                dpos1 = self.dummies.positions.copy()
        else:
            print("Iterative stepper failed!")
            if self.iterative_stepper == 2:
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                return
            self.atoms.positions = pos1
            self.dummies.positions = dpos1
        dx_final = self.get_x() - x0
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    # Andreas
    def get_cartesian_from_internal(self, coord_internal):
        """
        target: np.ndarray[self.dim,]:
            Internal coordinate vector
        return: tuple[np.ndarray[3*len(self.atoms),], np.ndarray[3*len(self.dummies),]]
            Atom and dummy positions in cartesian coordinates
        """
        return self.set_x(
            target=coord_internal, return_y=True, force_not_iterative=True
        )

    # Position getter/setter
    @_timed_method
    def set_x(self, target, return_y=False, force_not_iterative=False):
        """
        target: np.ndarray[self.dim,]:
            Internal coordinate vector
        return_y: bool
            If True, return atom and dummy positions in cartesian coordinates
        """
        if self.iterative_stepper and not force_not_iterative:
            res = self._set_x_iterative(target)
            if res is not None:
                return res
        dx = target - self.get_x()

        t0 = 0.0
        Binv = np.linalg.pinv(self.int.jacobian())
        y0 = np.hstack(
            (
                self.apos.ravel(),
                self.dpos.ravel(),
                Binv @ dx,
                Binv @ self.curr.get("g", np.zeros_like(dx)),
            )
        )
        ode = LSODA(self._q_ode, t0, y0, t_bound=1.0, atol=1e-6)

        if return_y:
            # buffer atom and dummy positions in cartesian coordinates
            # because _q_ode sets atoms.positions
            atoms_pos_tmp = self.atoms.positions.copy()
            dummies_pos_tmp = self.dummies.positions.copy()

        while ode.status == "running":
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                print("Bad internals found!")
                break
            if ode.nfev > 1000:
                view(self.atoms + self.dummies)
                raise RuntimeError(
                    "Geometry update ODE is taking too long to converge!"
                )

        if ode.status == "failed":
            raise RuntimeError("Geometry update ODE failed to converge!")

        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        if return_y:
            self.atoms.positions = atoms_pos_tmp
            self.dummies.positions = dummies_pos_tmp
            # return atom and dummy positions in cartesian coordinates
            return (
                y[0, :nxa].reshape((-1, 3)),
                y[0, nxa:].reshape((-1, 3)),
            )
        # set positions to self
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    @_timed_method
    def get_x(self):
        """Get the internal coordinate vector."""
        return self.int.calc()

    # Hessian of the constraints
    @_timed_method
    def get_Hc(self):
        D_cons = self.cons.hessian().ldot(self.curr["L"])
        B_int = self.int.jacobian()
        Binv_int = np.linalg.pinv(B_int)
        B_cons = self.cons.jacobian()
        L_int = self.curr["L"] @ B_cons @ Binv_int
        D_int = self.int.hessian().ldot(L_int)
        Hc = Binv_int.T @ (D_cons - D_int) @ Binv_int
        return Hc

    @_timed_method
    def get_drdx(self):
        # dr/dq = dr/dx dx/dq
        return PES.get_drdx(self) @ np.linalg.pinv(self.int.jacobian())

    @_timed_method
    def _calc_basis(self, internal=None, cons=None):
        if internal is None:
            internal = self.int
        if cons is None:
            cons = self.cons
        B = internal.jacobian()
        Ui, Si, VTi = np.linalg.svd(B)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Vnred = VTi[:nnred].T
        Siinv = np.diag(1 / Si[:nnred])
        drdxnred = cons.jacobian() @ Vnred @ Siinv
        drdx = drdxnred @ Unred.T
        Uc, Sc, VTc = np.linalg.svd(drdxnred)
        ncons = np.sum(Sc > 1e-6)
        Ucons = Unred @ VTc[:ncons].T
        Ufree = Unred @ VTc[ncons:].T
        return drdx, Ucons, Unred, Ufree

    @_timed_method
    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = np.linalg.pinv(self.int.jacobian())
        return f, g_cart @ Binv[: len(g_cart)]

    @_timed_method
    def update_internals(self, dx):
        self._update(True)

        nold = 3 * (len(self.atoms) + len(self.dummies))

        # FIXME: Testing to see if disabling this works
        # if self.bad_int is not None:
        #    for bond in self.bad_int['bonds']:
        #        self.int_orig.forbid_bond(bond)
        #    for angle in self.bad_int['angles']:
        #        self.int_orig.forbid_angle(angle)

        # Find new internals, constraints, and dummies
        new_int = self.int_orig.copy()
        new_int.find_all_bonds()
        new_int.find_all_angles()
        new_int.find_all_dihedrals()
        new_int.validate_basis()
        new_cons = new_int.cons

        # Calculate B matrix and its inverse for new and old internals
        Blast = self.int.jacobian()
        B = new_int.jacobian()
        Binv = np.linalg.pinv(B)
        Dlast = self.int.hessian()
        D = new_int.hessian()

        # # Projection matrices
        # P2 = B[:, nold:] @ Binv[nold:, :]

        # Update the info in self.curr
        x = new_int.calc()
        g = -self.atoms.get_forces().ravel() @ Binv[: 3 * len(self.atoms)]
        drdx, Ucons, Unred, Ufree = self._calc_basis(
            internal=new_int,
            cons=new_cons,
        )
        L = np.linalg.lstsq(drdx.T, g, rcond=None)[0]

        # Update H using old data where possible. For new (dummy) atoms,
        # use the guess hessian info.
        H = self.get_H().asarray()
        Hcart = Blast.T @ H @ Blast
        Hcart += Dlast.ldot(self.curr["g"])
        Hnew = Binv.T[:, :nold] @ (Hcart - D.ldot(g)) @ Binv
        self.dim = len(x)
        self.set_H(Hnew)

        self.int = new_int
        self.cons = new_cons

        self.curr.update(
            x=x,
            g=g,
            drdx=drdx,
            Ufree=Ufree,
            Unred=Unred,
            Ucons=Ucons,
            L=L,
            B=B,
            Binv=Binv,
        )

    @_timed_method
    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        # dx_r = self.wrap_dx(dx) @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.0

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        B = self.int.jacobian()
        return -((Ufree @ Ufree.T) @ g @ B).reshape((-1, 3))

    def wrap_dx(self, dx):
        return self.int.wrap(dx)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        D = self.int.hessian()
        Binv = np.linalg.pinv(self.int.jacobian())
        D_tmp = -Binv @ D.rdot(dxdt)
        dydt[1] = D_tmp @ dxdt
        dydt[2] = D_tmp @ g

        return dydt.ravel()

    @_timed_method
    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        # FIXME: Testing to see if this works
        # if self.bad_int is not None:
        #    self.update_internals(dx)
        #    self.bad_int = None

        return ratio

    @_timed_method
    def write_traj(self):
        if self.traj is not None:
            # Andreas start
            if hasattr(self.traj, "save_full_atoms"):
                self.traj.write(atoms=self.atoms, dummies=self.dummies)
                return
            # Andreas end
            energy = self.atoms.calc.results["energy"]
            forces = np.zeros((len(self.atoms) + len(self.dummies), 3))
            forces[: len(self.atoms)] = self.atoms.calc.results["forces"]
            # # Andreas start
            # if self.write_dummies_to_traj:
            #     atoms_tmp = copy.deepcopy(self.atoms) + copy.deepcopy(self.dummies)
            # else:
            #     atoms_tmp = copy.deepcopy(self.atoms)
            #     if hasattr(self.traj, "trajectory_dummies"):
            #         self.traj.trajectory_dummies.append(copy.deepcopy(self.dummies))
            # # Andreas end
            atoms_tmp = self.atoms + self.dummies
            atoms_tmp.calc = SinglePointCalculator(
                atoms_tmp, energy=energy, forces=forces
            )
            self.traj.write(atoms_tmp)
        return

    @_timed_method
    def _update(self, feval=True):
        if not PES._update(self, feval=feval):
            return

        B = self.int.jacobian()
        Binv = np.linalg.pinv(B)
        self.curr.update(B=B, Binv=Binv)
        return True

    @_timed_method
    def _convert_cartesian_hessian_to_internal(
        self,
        Hcart: np.ndarray,
    ) -> np.ndarray:
        ncart = 3 * len(self.atoms)
        # Get Jacobian and calculate redundant and non-redundant spaces
        B = self.int.jacobian()[:, :ncart]
        Ui, Si, VTi = np.linalg.svd(B)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Ured = Ui[:, nnred:]

        # Calculate inverse Jacobian in non-redundant space
        Bnred_inv = VTi[:nnred].T @ np.diag(1 / Si[:nnred])

        # Convert Cartesian Hessian to non-redundant internal Hessian
        Hcart_coupled = self.int.hessian().ldot(self.get_g())[:ncart, :ncart]
        Hcart_corr = Hcart - Hcart_coupled
        Hnred = Bnred_inv.T @ Hcart_corr @ Bnred_inv

        # Find eigenvalues of non-redundant internal Hessian
        lnred, _ = np.linalg.eigh(Hnred)

        # The redundant part of the Hessian will be initialized to the
        # geometric mean of the non-redundant eigenvalues
        lnred_mean = np.exp(np.log(np.abs(lnred)).mean())

        # finish reconstructing redundant internal Hessian
        return Unred @ Hnred @ Unred.T + lnred_mean * Ured @ Ured.T

    @_timed_method
    def _convert_internal_hessian_to_cartesian(
        self,
        Hint: np.ndarray,
    ) -> np.ndarray:
        B = self.int.jacobian()
        return B.T @ Hint @ B + self.int.hessian().ldot(self.get_g())

    @_timed_method
    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(
            self._convert_cartesian_hessian_to_internal(
                self.hessian_function(self.atoms)
            )
        )
