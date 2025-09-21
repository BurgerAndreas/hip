#!/usr/bin/env python3

import warnings
from time import localtime, strftime, perf_counter
from typing import Union, Callable, Optional

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils import basestring
from ase.io.trajectory import Trajectory

from .restricted_step import get_restricted_step
from sella.peswrapper import PES, InternalPES
from sella.internal import Internals, Constraints

_default_kwargs = dict(
    minimum=dict(
        delta0=1e-1,
        sigma_inc=1.15,
        sigma_dec=0.90,
        rho_inc=1.035,
        rho_dec=100,
        method="rfo",
        eig=False,
    ),
    saddle=dict(
        delta0=0.1,
        sigma_inc=1.15,
        sigma_dec=0.65,
        rho_inc=1.035,
        rho_dec=5.0,
        method="prfo",
        eig=True,
    ),
)


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


class Sella(Optimizer):
    """Second-order optimizer with constrained trust-region steps.

    Wraps a `PES`/`InternalPES` object and coordinates step prediction,
    optional Hessian (re)diagonalization, trust-radius adaptation, and
    logging. Supports both Cartesian and internal coordinates.
    """

    @_timed_method
    def __init__(
        self,
        atoms: Atoms,
        restart: bool = None,
        logfile: str = "-",
        # The name of the file. Traditionally ends in .traj
        trajectory: Union[str, Trajectory] = None,  # Sella PES trajectory
        asetraj: Union[str, Trajectory] = None,  # ASE trajectory
        master: bool = None,
        delta0: float = None,
        sigma_inc: float = None,
        sigma_dec: float = None,
        rho_dec: float = None,
        rho_inc: float = None,
        order: int = 1,
        eig: bool = None,
        eta: float = 1e-4,
        method: str = None,
        gamma: float = 0.1,
        threepoint: bool = False,
        constraints: Constraints = None,
        constraints_tol: float = 1e-5,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        append_trajectory: bool = False,
        rs: str = None,
        nsteps_per_diag: int = 3,
        diag_every_n: Optional[int] = None,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        log_every_n: int = 100,
        **kwargs,
    ):
        if order == 0:
            default = _default_kwargs["minimum"]
        else:
            default = _default_kwargs["saddle"]

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(
                    filename=trajectory, mode=mode, atoms=atoms, master=master
                )

        self.peskwargs = kwargs.copy()
        self.user_internal = internal
        # saves self.pes.traj
        self.initialize_pes(
            atoms,
            trajectory,
            order,
            eta,
            constraints,
            v0,
            internal,
            hessian_function,
            **kwargs,
        )

        if rs is None:
            rs = "mis" if internal else "ras"
        self.rs = get_restricted_step(rs)
        # this overwrites self.trajectory
        Optimizer.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=asetraj,
            master=master,
        )

        if delta0 is None:
            delta0 = default["delta0"]
        if rs in ["mis", "ras"]:
            self.delta = delta0
        else:
            self.delta = delta0 * self.pes.get_Ufree().shape[1]

        if sigma_inc is None:
            self.sigma_inc = default["sigma_inc"]
        else:
            self.sigma_inc = sigma_inc

        if sigma_dec is None:
            self.sigma_dec = default["sigma_dec"]
        else:
            self.sigma_dec = sigma_dec

        if rho_inc is None:
            self.rho_inc = default["rho_inc"]
        else:
            self.rho_inc = rho_inc

        if rho_dec is None:
            self.rho_dec = default["rho_dec"]
        else:
            self.rho_dec = rho_dec

        if method is None:
            self.method = default["method"]
        else:
            self.method = method

        if eig is None:
            self.eig = default["eig"]
        else:
            self.eig = eig

        self.ord = order
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.diagkwargs = dict(gamma=gamma, threepoint=threepoint)
        self.rho = 1.0
        self.log_every_n = log_every_n

        if self.ord != 0 and not self.eig:
            warnings.warn(
                "Saddle point optimizations with eig=False will "
                "most likely fail!\n Proceeding anyway, but you "
                "shouldn't be optimistic."
            )

        self.initialized = False
        self.xi = 1.0
        self.nsteps_per_diag = nsteps_per_diag
        self.nsteps_since_diag = 0
        self.diag_every_n = np.inf if diag_every_n is None else diag_every_n

    @_timed_method
    def initialize_pes(
        self,
        atoms: Atoms,
        trajectory: str = None,
        order: int = 1,
        eta: float = 1e-4,
        constraints: Constraints = None,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        **kwargs,
    ):
        """Create and configure a `PES` or `InternalPES` based on settings.

        If `internal` is True, build an `InternalPES` using either a provided
        `Internals` object or auto-detected internals; otherwise build a
        Cartesian `PES`. Also wires the trajectory and optional Hessian hook.
        """
        if internal:
            if isinstance(internal, Internals):
                auto_find_internals = False
                if constraints is not None:
                    raise ValueError(
                        "Internals object and Constraint object cannot both "
                        "be provided to Sella. Instead, you must pass the "
                        "Constraints object to the constructor of the "
                        "Internals object."
                    )
            else:
                auto_find_internals = True
                internal = Internals(atoms, cons=constraints)
            self.internal = internal.copy()
            self.constraints = None
            self.pes = InternalPES(
                atoms,
                internals=internal,
                trajectory=trajectory,
                eta=eta,
                v0=v0,
                auto_find_internals=auto_find_internals,
                hessian_function=hessian_function,
                write_dummies_to_traj=False,
                **kwargs,
            )
            print("InternalPES deduced internals (dyn.pes.int.internals):")
            for k, v in self.pes.int.internals.items():
                print(f" {k}: {len(v)}")
            print(f" dyn.pes.dim={self.pes.dim}, dyn.pes.ncart={self.pes.ncart}")
        else:
            self.internal = None
            if constraints is None:
                constraints = Constraints(atoms)
            self.constraints = constraints
            self.pes = PES(
                atoms,
                constraints=constraints,
                trajectory=trajectory,
                eta=eta,
                v0=v0,
                hessian_function=hessian_function,
                **kwargs,
            )
        self.trajectory = self.pes.traj
        return

    @_timed_method
    def _predict_step(self):
        """Construct a restricted trust-region step in the free subspace.

        Ensures the PES is initialized (and optionally diagonalized) on the
        first call. Then uses the configured restricted-step method to produce
        a step `s` and its norm `smag` that respect inequality constraints.
        """
        if not self.initialized:
            self.pes.get_g()
            if self.eig:
                if self.pes.hessian_function is not None:
                    self.pes.calculate_hessian()
                else:
                    self.pes.diag(**self.diagkwargs)
                self.nsteps_since_diag = -1
            self.initialized = True

        self.pes.cons.disable_satisfied_inequalities()
        self.pes._update_basis()
        self.pes.save()
        all_valid = False
        x0 = self.pes.get_x()
        while not all_valid:
            s, smag = self.rs(
                self.pes, self.ord, self.delta, method=self.method
            ).get_s()
            self.pes.set_x(x0 + s)
            all_valid = self.pes.cons.validate_inequalities()
            self.pes._update_basis()
            self.pes.restore()
        self.pes._update_basis()
        return s, smag

    @_timed_method
    def step(self):
        """Perform one optimization iteration.

        - Predict a step and decide whether to re-diagonalize the Hessian.
        - Apply the step via `pes.kick`, obtaining the trust ratio.
        - Handle bad internal coordinates (if any) by reinitializing the PES.
        - Update the trust radius based on the ratio.
        """
        # Predict a constrained trust-region step and its magnitude
        s, smag = self._predict_step()

        # Decide whether to re-diagonalize based on cadence and curvature
        if self.nsteps_since_diag >= self.diag_every_n:
            ev = True
        elif self.eig and self.nsteps_since_diag >= self.nsteps_per_diag:
            if self.pes.H.evals is None:
                ev = True
            else:
                Unred = self.pes.get_Unred()
                ev = (self.pes.get_HL().project(Unred).evals[: self.ord] > 0).any()
        else:
            ev = False

        # Update diagonalization cadence counter
        if ev:
            self.nsteps_since_diag = 0
        else:
            self.nsteps_since_diag += 1

        # Take the step and get the trust ratio from PES
        rho = self.pes.kick(s, diag=ev, **self.diagkwargs)

        # Check for bad internals, and if found, reset PES object.
        # This skips the trust radius update.
        if self.internal and self.pes.int.check_for_bad_internals():
            self.initialize_pes(
                atoms=self.pes.atoms,
                trajectory=self.pes.traj,
                order=self.ord,
                eta=self.pes.eta,
                constraints=self.constraints,
                v0=None,  # TODO: use leftmost eigenvector from old H
                internal=self.user_internal,
                hessian_function=self.pes.hessian_function,
            )
            self.initialized = False
            self.rho = 1
            return

        # Update trust radius based on ratio; shrink on poor agreement,
        # grow if the model agrees within bounds
        if rho is None:
            pass
        elif rho < 1.0 / self.rho_dec or rho > self.rho_dec:
            self.delta = max(smag * self.sigma_dec, self.delta_min)
        elif 1.0 / self.rho_inc < rho < self.rho_inc:
            self.delta = max(self.sigma_inc * smag, self.delta)
        self.rho = rho
        if self.rho is None:
            self.rho = 1.0

    @_timed_method
    def converged(self, forces=None):
        """Return True if both force and constraint tolerances are met."""
        return self.pes.converged(self.fmax)[0]

    @_timed_method
    def log(self, forces=None):
        """Write a periodic log line with energy, force, constraint and TR info."""
        if self.logfile is None:
            return
        _, fmax, cmax = self.pes.converged(self.fmax)
        e = self.pes.get_f()
        T = strftime("%H:%M:%S", localtime())
        name = self.__class__.__name__
        buf = " " * len(name)
        if self.nsteps == 0:
            self.logfile.write(
                buf
                + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} {:>12s} {:>12s}\n".format(
                    "Step", "Time", "Energy", "fmax", "cmax", "rtrust", "rho"
                )
            )
        # Andreas
        if self.nsteps % self.log_every_n == 0:
            self.logfile.write(
                "{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                "{:>12.4f} {:>12.4f}\n".format(
                    name, self.nsteps, T, e, fmax, cmax, self.delta, self.rho
                )
            )
        self.logfile.flush()

    def print_total_time_per_function(self, top=5):
        """Print cumulative time spent in each Sella method for this instance."""
        print("Sella timings:")
        if hasattr(self, "_time_stats") and self._time_stats:
            for name, total in list(
                sorted(self._time_stats.items(), key=lambda kv: kv[1], reverse=True)
            )[:top]:
                print(f"{name}: {total:.6f}s")
        else:
            print("No timing data recorded.")

        # Also print timings from the associated PES object if available
        if hasattr(self, "pes") and hasattr(self.pes, "print_total_time_per_function"):
            print("PES timings:")
            self.pes.print_total_time_per_function(top=top)
