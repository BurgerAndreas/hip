import argparse
import time
import sys
import pathlib
import contextlib
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import logging
import h5py
import pandas as pd
import wandb
import shutil

import torch
from torch_geometric.data import DataLoader as TGDataLoader

try:
    from pysisyphus.Geometry import Geometry  # Geometry API + coordinate systems
    from pysisyphus.calculators.MLFF import MLFF
    from pysisyphus.calculators.Calculator import (
        Calculator,
    )  # base class to wrap/override
    from pysisyphus.optimizers.FIRE import FIRE  # first-order baseline
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer  # second-order RFO + BFGS
    from pysisyphus.optimizers.BFGS import BFGS
    from pysisyphus.optimizers.SteepestDescent import SteepestDescent
    from pysisyphus.optimizers.ConjugateGradient import ConjugateGradient
    from pysisyphus.optimizers.BacktrackingOptimizer import BacktrackingOptimizer

    # from pysisyphus.helpers_pure import eigval_to_wavenumber
    # from pysisyphus.helpers import _do_hessian
    # from pysisyphus.io.hessian import save_hessian
    from pysisyphus.constants import AU2EV, BOHR2ANG
    from pysisyphus.helpers import procrustes

    from ReactBench.Calculators.equiformer import PysisEquiformer
except ImportError:
    print()
    traceback.print_exc()
    print("\nFollow the instructions here: https://github.com/BurgerAndreas/ReactBench")
    exit()


from ase import Atoms
from ase.io import read
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.vibrations.data import VibrationsData
from ase.vibrations import Vibrations
from ase.optimize import BFGS
from ase.mep import NEB
from sella import Sella, Constraints, IRC

from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path, ROOT_DIR
from nets.prediction_utils import (
    GLOBAL_ATOM_SYMBOLS,
    GLOBAL_ATOM_NUMBERS,
    compute_extra_props,
)

from hip.t1x_dft_dataloader import Dataloader as T1xDFTDataloader

# try:
#     from transition1x import Dataloader as T1xDataloader
# except ImportError:
#     print(
#         "Transition1x not found, please install it by:\n"
#         "git clone https://gitlab.com/matschreiner/Transition1x.git" + "\n"
#         "uv run Transition1x/download_t1x.py Transition1x/data" + "\n"
#         "uv pip install -e Transition1x" + "\n"
#     )
from hip.colours import (
    COLOUR_LIST,
    OPTIM_TO_COLOUR,
    ANNOTATION_FONT_SIZE,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

METRIC_TO_LABEL = {
    "steps": "Steps to Convergence",
    "wall_time_s": "Wall Time [s]",
}

"""
run as:
python scripts/second_order_relaxation_pysiyphus.py --coord redund

It implements four variants:

1. baseline: first-order (FIRE)
2. no-Hessian: RFO+BFGS with unit initial Hessian
3. initial-only: RFO+BFGS with learned only at step 0
4. periodic replace: RFO+BFGS with learned every k in {3,1}

- baseline: FIRE -> first-order only, no Hessian
- no-Hessian: RFOptimizer(hessian_init='unit', hessian_update='bfgs') -> quasi-Newton with a diagonal initial guess (no external Hessian) ([pysisyphus.readthedocs.io][1])
- initial-only: RFOptimizer(hessian_init='calc', hessian_recalc=None) -> your H at step 0, then BFGS updates only ([pysisyphus.readthedocs.io][1])
- periodic replace: RFOptimizer(hessian_init='calc', hessian_recalc=k) -> your H injected every k steps (k=3,1) ([pysisyphus.readthedocs.io][1])

- Coordinate systems: use redund (RIC) or dlc/tric for stability

Notes
- Metrics collected: gradient evaluations (counted at the calculator), wall time, steps (cycles), success flag. Optional trust-region diagnostics vary across versions; if you need them we can also parse the HDF5 dump that pysisyphus writes. ([pysisyphus.readthedocs.io][1])
If you want the optional diagnostics
- Step rejection rate and trust-radius statistics can be pulled from the optimizer's dump (HDF5) if you instantiate RFOptimizer/FIRE with dump=True and parse optimization.h5 afterwards. pysisyphus exposes trust-region controls, and the docs list the trust-radius options. ([pysisyphus.readthedocs.io][1])

Caveats on units
pysisyphus uses Hartree/Bohr.
The cartesian Hessian is then transformed into internals by pysisyphus under the hood.


[1]: https://pysisyphus.readthedocs.io/en/latest/min_optimization.html "7. Minimization - pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
[2]: https://pysisyphus.readthedocs.io/en/latest/coordinate_systems.html "5. Coordinate Systems - pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
[3]: https://pysisyphus.readthedocs.io/en/latest/calculators.html "6. Calculators - pysisyphus 0.8.0b1.dev151+gaa70dd3 documentation"
"""

"""
Optimizer args that apply to all optimizers:
# Convergence threshold.
thresh: Thresh = "gau_loose", https://psicode.org/psi4manual/master/optking.html#table-optkingconv
# Maximum absolute component of the allowed step vector. Utilized in
# optimizers that don't support a trust region or line search.
max_step: float = 0.04,
# Maximum number of allowed optimization cycles.
max_cycles: int = 150,
# Minimum norm of an allowed step. If the step norm drops below
# this value a ZeroStepLength-exception is raised. The unit depends
# on the coordinate system of the supplied geometry.
min_step_norm: float = 1e-8,
# Flag that controls whether the norm of the proposed step is check
# for being too small.
assert_min_step: bool = True,
# Root-mean-square of the force from which user-defined thresholds
# are derived. When 'rms_force' is given 'thresh' is ignored.
rms_force: Optional[float] = None,
# When set, convergence is signalled only based on rms(forces).
rms_force_only: bool = False,
# When set, convergence is signalled only based on max(|forces|).
max_force_only: bool = False,
# When set, convergence is signalled only based on max(|forces|) and rms(forces).
force_only: bool = False,
# Threshold for the RMSD with another geometry. When the RMSD drops
# below this threshold convergence is signalled. Only used with
# Growing Newton trajectories.
converge_to_geom_rms_thresh: float = 0.05,
# Flag that controls whether the geometry is aligned in every step
# onto the coordinates of the previous step. Must not be used with
# internal coordinates.
align: bool = False,
# Factor that controls the strength of the alignment. 1.0 means
# full alignment, 0.0 means no alignment. The factor mixes the
# rotation matrix of the alignment with the identity matrix.
align_factor: float = 1.0,
# Flag to control dumping/writing of optimization progress to the
# filesystem
dump: bool = False,
# Flag to control whether restart information is dumped to the
# filesystem.
dump_restart: bool = False,
# Report optimization progress every nth cycle.
print_every: int = 1,
# Short string that is prepended to several files created by
# the optimizer. Allows distinguishing several optimizations carried
# out in the same directory.
prefix: str = "",
# Controls the minimal allowed similarity between coordinates
# after two successive reparametrizations. Convergence is signalled
# if the coordinates did not change significantly.
reparam_thresh: float = 1e-3,
# Whether to check for (too) similar coordinates after reparametrization.
reparam_check_rms: bool = True,
# Reparametrize before or after calculating the step. Can also be turned
# off by setting it to None.
reparam_when: Optional[Literal["before", "after"]] = "after",
# Signal convergence when max(forces) and rms(forces) fall below the
# chosen threshold, divided by this factor. Convergence of max(step) and
# rms(step) is ignored.
overachieve_factor: float = 0.0,
# Check the eigenvalues of the modes we maximize along. Convergence requires
# them to be negative. Useful if TS searches are started from geometries close
# to a minimum.
check_eigval_structure: bool = False,
# Restart information. Undocumented.
restart_info=None,
# Whether coordinates of chain-of-sates images are checked for being
# too similar.
check_coord_diffs: bool = True,
# Unitless threshold for similary checking of COS image coordinates.
# The first image is assigned 0, the last image is assigned to 1.
coord_diff_thresh: float = 0.01,
# Tuple of lists containing atom indices, defining two fragments.
fragments: Optional[Tuple] = None,
# Monitor fragment distances for N cycles. The optimization is terminated
# when the interfragment distances falls below the initial value after N
# cycles.
monitor_frag_dists: int = 0,
# Basename of the HDF5 file used for dumping.
h5_fn: str = "optimization.h5",
# Groupname used for dumping of this optimization.
h5_group_name: str = "opt",
"""

pysis_all_optimizers = [
    "BFGS",
    "ConjugateGradient",
    "CubicNewton",
    "FIRE",
    "LayerOpt",
    "LBFGS",
    "MicroOptimizer",
    "NCOptimizer",
    "PreconLBFGS",
    "PreconSteepestDescent",
    "QuickMin",
    "RFOptimizer",
    "SteepestDescent",
    "StringOptimizer",
    "StabilizedQNMethod",
]

# --------------------------
#  Utilities
# --------------------------


Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def load_xyz(fn):
    """Read a single-geometry XYZ (Angstrom). Returns atoms (list[str]), coords (1d array, 3N)."""
    lines = pathlib.Path(fn).read_text().strip().splitlines()
    try:
        n = int(lines[0].strip())
    except Exception:
        raise ValueError("XYZ: first line must be atom count")
    body = lines[2 : 2 + n]
    atoms, coords3d = [], []
    for ln in body:
        el, x, y, z = ln.split()[:4]
        atoms.append(el)
        coords3d.append([float(x), float(y), float(z)])
    coords = np.asarray(coords3d, float).reshape(-1)  # 3N (Å)
    return atoms, coords


def clean_str(s):
    return "".join(c for c in s.replace(" ", "_") if c.isalnum()).lower()


COORD_TO_NAME = {
    "cart": "Cartesian Coordinates",
    "redund": "Redundant Internal Coordinates",
    "tric": "Translation & Rotation Internal Coordinates",
    "dlc": "Delocalized Internal Coordinates",
}


def regularize_minimum_hessian(H, eps=1e-6):
    """Make sure Hessian is positive definite for minima (eigendecomp + floor)."""
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


# --------------------------
#  Calculator wrappers
# --------------------------


class CountingCalc(Calculator):
    """
    Wrap any Calculator; count energy/gradient/Hessian calls.
    """

    def __init__(self, inner, assert_pd_hessians=False, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.reset()
        self.assert_pd_hessians = assert_pd_hessians

    def reset(self):
        self.energy_calls = 0
        self.grad_calls = 0
        self.hessian_calls = 0
        self.calculate_calls = 0
        self.calculate_energy_calls = 0
        self.calculate_gradient_calls = 0
        self.calculate_hessian_calls = 0
        self.cnt_not_pd = 0
        super().reset()

    @property
    def model(self):
        return self.inner.model

    # Delegate / count
    def get_energy(self, atoms, coords, **kw):
        self.energy_calls += 1
        return self.inner.get_energy(atoms, coords, **kw)

    def get_forces(self, atoms, coords, **kw):
        self.grad_calls += 1
        return self.inner.get_forces(atoms, coords, **kw)

    def get_hessian(self, atoms, coords, **kw):
        self.hessian_calls += 1
        results = self.inner.get_hessian(atoms, coords, **kw)
        # check if hessian is positive definite
        if not np.all(np.linalg.eigvals(results["hessian"]) > 0):
            print("Predicted Hessian is not positive definite")
            self.cnt_not_pd += 1
            if self.assert_pd_hessians:
                raise ValueError("Predicted Hessian is not positive definite")
        return results

    def get_num_hessian(self, atoms, coords, prepare_kwargs={}):
        self.hessian_calls += 1
        results = self.inner.get_num_hessian(atoms, coords, **prepare_kwargs)
        # check if hessian is positive definite
        if not np.all(np.linalg.eigvals(results["hessian"]) > 0):
            print("Numerical Hessian is not positive definite")
            self.cnt_not_pd += 1
            if self.assert_pd_hessians:
                raise ValueError("Numerical Hessian is not positive definite")
        return results

    def calculate(self, atom=None, properties=None, **kwargs):
        self.calculate_calls += 1
        if properties is None:
            properties = kwargs.get("properties", None)
        if properties is not None:
            if "energy" in properties:
                self.calculate_energy_calls += 1
            if "gradient" in properties:
                self.calculate_gradient_calls += 1
            if "hessian" in properties:
                self.calculate_hessian_calls += 1
        return self.inner.calculate(atom, properties, **kwargs)


# --------------------------
#  Optimizer runners
# --------------------------


class NaiveSteepestDescent(BacktrackingOptimizer):
    def __init__(self, geometry, **kwargs):
        super(NaiveSteepestDescent, self).__init__(geometry, alpha=0.1, **kwargs)

    def optimize(self):
        if self.is_cos and self.align:
            procrustes(self.geometry)

        self.forces.append(self.geometry.forces)

        step = self.alpha * self.forces[-1]
        step = self.scale_by_max_step(step)
        return step


def _run_opt_safely(
    geom,
    opt,
    method_name,
    out_dir,
    verbose=False,
    start_clean=True,
    dft_hessian_is_pd=True,
):
    # logging
    if start_clean:
        geom.calculator.reset()
        assert geom.calculator.grad_calls == 0, (
            f"Calculator counts {geom.calculator.grad_calls} gradient calls"
        )
        # assert geom._masses is None, f"Masses are not None: {geom._masses}" # computed in Geometry.__init__
        assert geom._energy is None, f"Energy is not None: {geom._energy}"
        assert geom._forces is None, f"Forces are not None: {geom._forces}"
        assert geom._hessian is None, f"Hessian is not None: {geom._hessian}"
        assert geom._all_energies is None, (
            f"All energies are not None: {geom._all_energies}"
        )

    method_name_clean = clean_str(method_name)
    log_path = os.path.join(out_dir, f"optrun_{method_name_clean}.txt")

    # wrapper to run optimizer and return results
    def _try_to_run(_opt):
        try:
            t0 = time.perf_counter()
            _opt.run()
            t1 = time.perf_counter()
            steps = _opt.cur_cycle
            return {
                "name": method_name,
                "converged": bool(getattr(_opt, "is_converged", False)),
                "steps": int(steps) if steps is not None else None,
                "grad_calls": geom.calculator.grad_calls,
                "hessian_calls": geom.calculator.hessian_calls,
                "energy_calls": geom.calculator.energy_calls,
                "calculate_calls": geom.calculator.calculate_calls,
                "calculate_energy_calls": geom.calculator.calculate_energy_calls,
                "calculate_gradient_calls": geom.calculator.calculate_gradient_calls,
                "calculate_hessian_calls": geom.calculator.calculate_hessian_calls,
                "cnt_not_pd": geom.calculator.cnt_not_pd,
                "wall_time_s": t1 - t0,
                "dft_hessian_is_pd": dft_hessian_is_pd,
            }
        except Exception as e:
            print(f"Error running {method_name} optimization: {e}", flush=True)
            traceback.print_exc()
            return {
                "name": method_name,
                "converged": False,
                "steps": None,
                "grad_calls": None,
                "hessian_calls": None,
                "energy_calls": None,
                "calculate_calls": None,
                "calculate_energy_calls": None,
                "calculate_gradient_calls": None,
                "calculate_hessian_calls": None,
                "cnt_not_pd": None,
                "wall_time_s": None,
                "dft_hessian_is_pd": dft_hessian_is_pd,
            }

    # run optimizer and return results
    if verbose:
        return _try_to_run(opt)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"Saving log to {log_path}")
        with open(log_path, "a") as _log_fh, contextlib.redirect_stdout(
            _log_fh
        ), contextlib.redirect_stderr(_log_fh):
            return _try_to_run(opt)


def get_rfo_optimizer(
    geom,
    *,
    hessian_init,
    thresh,
    hessian_update="bfgs",
    hessian_recalc=None,
    trust_radius=0.3,
    max_cycles=200,
    out_dir=".",
    verbose=False,
):
    # RFO with flexible Hessian policies. hessian_init ∈ {'unit','calc',...}; hessian_recalc = k or None.
    opt = RFOptimizer(
        # line_search: bool = True
        # Whether to carry out implicit line searches.
        # gediis: bool = False
        # Whether to enable GEDIIS.
        # gdiis: bool = True
        # Whether to enable GDIIS.
        # gdiis_thresh: float = 2.5e-3
        # Threshold for rms(forces) to enable GDIIS.
        # gediis_thresh: float = 1e-2
        # Threshold for rms(step) to enable GEDIIS.
        # gdiis_test_direction: bool = True
        # Whether to the overlap of the RFO step and the GDIIS step.
        # max_micro_cycles: int = 25
        # Number of restricted-step microcycles. Disabled by default.
        # adapt_step_func: bool = False
        # # HessianOptimizer
        # Whether to switch between shifted Newton and RFO-steps.
        # trust_radius: float = 0.5
        # Initial trust radius in whatever unit the optimization is carried out.
        # trust_update: bool = True
        # Whether to update the trust radius throughout the optimization.
        # trust_min: float = 0.1
        # Minimum trust radius.
        # trust_max: float = 1
        # Maximum trust radius.
        # max_energy_incr: Optional[float] = None
        # Maximum allowed energy increased after a faulty step. Optimization is
        # aborted when the threshold is exceeded.
        # hessian_update: HessUpdate = "bfgs"
        # Type of Hessian update. Defaults to BFGS for minimizations and Bofill
        # for saddle point searches.
        # hessian_init: HessInit = "fischer"
        # Type of initial model Hessian.
        # hessian_recalc: Optional[int] = None
        # Recalculate exact Hessian every n-th cycle instead of updating it.
        # hessian_recalc_adapt: Optional[float] = None
        # Use a more flexible scheme to determine Hessian recalculation. Undocumented.
        # hessian_xtb: bool = False
        # Recalculate the Hessian at the GFN2-XTB level of theory.
        # hessian_recalc_reset: bool = False
        # Whether to skip Hessian recalculation after reset. Undocumented.
        # small_eigval_thresh: float = 1e-8
        # Threshold for small eigenvalues. Eigenvectors belonging to eigenvalues
        # below this threshold are discardewd.
        # line_search: bool = False
        # Whether to carry out a line search. Not implemented by a subclassing
        # optimizers.
        # alpha0: float = 1.0
        # Initial alpha for restricted-step (RS) procedure.
        # max_micro_cycles: int = 25
        # Maximum number of RS iterations.
        # rfo_overlaps: bool = False
        # Enable mode-following in RS procedure.
        # Geometry to be optimized.
        geom,
        thresh=thresh,
        trust_radius=trust_radius,
        # np.array, .h5 path, calc, fischer, unit, simple
        hessian_init=hessian_init,
        hessian_update=hessian_update,
        hessian_recalc=hessian_recalc,
        line_search=True,
        out_dir=out_dir,
        max_cycles=max_cycles,
        # # TS opt in ReactBench uses
        # # pysisyphus.tsoptimizers.RSPRFOptimizer
        # type: rsprfo
        # do_hess: True
        # thresh: gau
        # max_cycles: 50
        # trust_radius: 0.2 # here we use 0.3
        # hessian_recalc: 1
        allow_write=False,
    )
    return opt


# --------------------------
#  Main harness
# --------------------------


def get_geom(atomssymbols, coords, coord_type, base_calc, args):
    geom = Geometry(atomssymbols, coords, coord_type=coord_type)
    base_calc.reset()
    counting_calc = CountingCalc(base_calc, assert_pd_hessians=args.pdpredonly)
    geom.set_calculator(counting_calc)
    return geom


def print_header(i, method):
    print("\n" + "=" * 10 + " " + str(i) + " " + method + " " + "=" * 10)


# match OPTIM_TO_COLOUR
METHOD_TO_CATEGORY = {
    "NaiveSteepestDescent": "First-Order",
    "SteepestDescent": "First-Order",
    "FIRE": "First-Order",
    "ConjugateGradient": "First-Order",
    "RFO-BFGS (unit init)": "Quasi-Second-Order",
    "RFO-BFGS (DFT init)": "Quasi-Second-Order",
    "RFO-BFGS (autograd init)": "Quasi-Second-Order",
    "RFO-BFGS (NumHess init)": "Quasi-Second-Order",
    "RFO-BFGS (learned init)": "Quasi-Second-Order",
    "RFO-BFGS (learned k3)": "Quasi-Second-Order",
    "RFO (NumHess)": "Second-Order",
    "RFO (NumHess 4)": "Second-Order",
    "RFO (autograd)": "Second-Order",
    # "RFO (learned)": "ours",
    "RFO (learned)": "Second-Order",
}
rename_categories = {
    "First-Order": "No Hessians",
    "Quasi-Second-Order": "Quasi-Hessian",
    "Second-Order": "Hessian",
}
METHOD_TO_CATEGORY = {k: rename_categories[v] for k, v in METHOD_TO_CATEGORY.items()}
METHOD_TO_COLOUR = {
    m: OPTIM_TO_COLOUR[METHOD_TO_CATEGORY[m]] for m in METHOD_TO_CATEGORY
}
DO_METHOD = [
    "NaiveSteepestDescent",
    # "SteepestDescent",
    "FIRE",
    "RFO (autograd)",
    "RFO (NumHess)",
    # "ConjugateGradient",
    "RFO-BFGS (unit init)",
    # "RFO-BFGS (DFT init)",
    "RFO-BFGS (autograd init)",
    "RFO-BFGS (NumHess init)",
    # "RFO-BFGS (learned k3)",
    # "RFO (NumHess 4)",
    "RFO (learned)",
    "RFO-BFGS (learned init)",
]

# Plot again
COMPETATIVE_METHODS_STEPS = [
    "RFO-BFGS (unit init)",
    "RFO-BFGS (DFT init)",
    "RFO-BFGS (NumHess init)",
    "RFO-BFGS (learned init)",
    "RFO-BFGS (learned k3)",
    "RFO-BFGS (autograd init)",
    "RFO (learned)",
]
COMPETATIVE_METHODS_WALL_TIME = [
    # "NaiveSteepestDescent",
    # "SteepestDescent",
    "FIRE",
    "ConjugateGradient",
    "RFO-BFGS (unit init)",
    # "RFO-BFGS (DFT init)",
    "RFO-BFGS (NumHess init)",
    "RFO-BFGS (learned init)",
    "RFO-BFGS (learned k3)",
    "RFO-BFGS (autograd init)",
    "RFO (learned)",
]

RENAME_METHODS_PLOT = {
    "NaiveSteepestDescent": "SteepestDescent",
    "RFO-BFGS (NumHess init)": "RFO-BFGS (FiniteDifference init)",
    "RFO (NumHess)": "RFO (FiniteDifference)",
}


def do_relaxations(out_dir, source_label, args):
    print("Loading dataset...")
    print(f"Dataset: {args.xyz}. is file: {os.path.isfile(args.xyz)}")
    data_is_xyz = False
    data_is_t1x = False
    data_is_lmdb = False
    if "t1x" in args.xyz and os.path.isfile(args.xyz):
        # /ssd/Code/hip/data/t1x_val_reactant_hessian_100_noiserms0.03.h5
        dataset_path = args.xyz
        print(f"Loading T1x dataset from {dataset_path}")
        dataset = T1xDFTDataloader(dataset_path, datasplit="val", only_final=True)
        data_is_t1x = True
    elif args.xyz in ["t1x"]:
        # dataset_path = "Transition1x/data/transition1x.h5"
        dataset_path = "data/t1x_val_reactant_hessian_100.h5"
        dataset = T1xDFTDataloader(dataset_path, datasplit="val", only_final=True)
        data_is_t1x = True
    elif args.xyz.startswith("t1x"):
        # dataset_path = "Transition1x/data/transition1x.h5"
        noise_str = args.xyz.split("_")[-1]
        dataset_path = f"data/t1x_val_reactant_hessian_100_noiserms{noise_str}.h5"
        dataset = T1xDFTDataloader(dataset_path, datasplit="val", only_final=True)
        data_is_t1x = True
        dataset_path = (
            f"data/t1x_val_reactant_hessian_100_noiserms{noise_str.replace('.', '')}.h5"
        )
    elif os.path.isfile(args.xyz):
        # is xyz file
        dataset = [args.xyz]
        dataset_path = args.xyz
        data_is_xyz = True
        # folder of xyz files
    elif os.path.isdir(args.xyz):
        dataset = [
            os.path.join(args.xyz, f)
            for f in os.listdir(args.xyz)
            if f.endswith(".xyz")
        ]
        dataset_path = args.xyz
        data_is_xyz = True
    # lmdb path
    elif args.xyz.endswith(".lmdb"):
        dataset_path = fix_dataset_path(args.xyz)
        dataset = LmdbDataset(dataset_path)
        data_is_lmdb = True
        # dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)
    else:
        raise ValueError(f"Invalid dataset path: {args.xyz}")

    try:
        len_dataset = len(dataset)
    except:
        len_dataset = 1_000

    if args.max_samples > len_dataset:
        args.max_samples = len_dataset

    if args.redo:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Out directory: {out_dir}")

    rng = np.random.default_rng(seed=42)

    print("\nRunning relaxations...")
    csv_path = os.path.join(out_dir, f"relaxation_results.csv")
    if not os.path.exists(csv_path) or args.redo:
        print("\nInitializing model...")
        # base_calc = MLFF(
        base_calc = PysisEquiformer(
            charge=0,
            ckpt_path=args.ckpt_path,
            config_path="auto",
            device="cuda",
            hessianmethod_name="predict",
            hessian_method="predict",  # "autograd", "predict"
            mem=4000,
            method="equiformer",
            mult=1,
            pal=1,
            # out_dir=yaml_dir / OUT_DIR_DEFAULT,
            # 'out_dir': PosixPath('/ssd/Code/ReactBench/runs/equiformer_alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956_data_predict/rxn9/TSOPT/qm_calcs
        )
        # print(f"Initialized calc MLFF.: {base_calc.__class__.__name__}")
        # print(f"Initialized calc MLFF.model: {base_calc.model.__class__.__name__}")
        # print(
        #     f"Initialized calc MLFF.model.model: {base_calc.model.model.__class__.__name__}"
        # )
        # print(
        #     f"Initialized calc MLFF.model.model.potential: {base_calc.model.model.potential.__class__.__name__}"
        # )

        print("\nTesting model with pysisyphus...")
        counting_calc = CountingCalc(base_calc)
        if data_is_xyz:
            atoms, coords = load_xyz(dataset[0])
        elif data_is_t1x:
            molecule = next(iter(dataset))
            if "positions_noised" in molecule["reactant"]:
                coords = molecule["reactant"]["positions_noised"]
            else:
                coords = molecule["reactant"]["positions"]
            # ts = molecule["transition_state"]["positions"]
            # product = molecule["product"]["positions"]
            atoms = np.array(molecule["reactant"]["atomic_numbers"])
            atomssymbols = [Z_TO_SYMBOL[a] for a in atoms]
            coords = coords / BOHR2ANG  # same as *ANG2BOHR
            t1xdataloader = iter(dataset)
        else:
            data = dataset[0]
            atomssymbols = GLOBAL_ATOM_SYMBOLS[
                data.one_hot.long().argmax(dim=1).cpu().numpy()
            ]
            coords = data.pos.numpy() / BOHR2ANG
        geom = Geometry(atomssymbols, coords, coord_type=args.coord)
        geom.set_calculator(counting_calc)
        energy = geom.energy
        forces = geom.forces
        hessian = geom.hessian
        # Test finished

        ts = time.strftime("%Y%m%d-%H%M%S")
        # Accumulate results across all samples
        all_results = []

        np.random.seed(42)
        torch.manual_seed(42)
        random_idx = np.random.permutation(len_dataset)

        print()
        optims_done = 0
        for cnt, idx in enumerate(random_idx):
            if optims_done >= args.max_samples:
                break
            print(
                "",
                "=" * 80,
                f"\tSample {optims_done} (tried cnt={cnt}, idx={idx} / {len_dataset})\t",
                "=" * 80,
                sep="\n",
            )
            if data_is_xyz:
                data = dataset[idx]
                atomssymbols, coords = load_xyz(data)
                # skip if there are any other atomsymbols than C, H, N, O
                if not all(
                    atomssymbols in ["C", "H", "N", "O"]
                    for atomssymbols in atomssymbols
                ):
                    print(
                        "Skipping sample because it contains non-C, H, N, O atoms",
                        atomssymbols,
                    )
                    continue
                hessian_path = data.replace(".xyz", ".hessian.npy")
                initial_dft_hessian = np.load(hessian_path)
            elif data_is_t1x:
                molecule = next(t1xdataloader)
                idx = molecule["reactant"].get("idx", cnt)
                if "positions_noised" in molecule["reactant"]:
                    coords = molecule["reactant"]["positions_noised"]
                    print("Using noised geometry")
                else:
                    coords = molecule["reactant"]["positions"]
                    print("Using original geometry")
                atoms = np.array(molecule["reactant"]["atomic_numbers"])
                atomssymbols = [Z_TO_SYMBOL[a] for a in atoms]
                coords = coords / BOHR2ANG  # same as *ANG2BOHR
                initial_dft_hessian = molecule["reactant"]["wB97x_6-31G(d).hessian"]
                # eV/Angstrom^2 -> Hartree/Bohr^2
                # initial_dft_hessian = initial_dft_hessian * AU2EV * BOHR2ANG * BOHR2ANG
            else:
                data = dataset[idx]
                indices = data.one_hot.long().argmax(dim=1)
                atomssymbols = GLOBAL_ATOM_SYMBOLS[indices.cpu().numpy()]
                natoms = len(atomssymbols)
                coords = data.pos.numpy() / BOHR2ANG
                # eV/Angstrom^2 -> Hartree/Bohr^2
                initial_dft_hessian = (
                    data.hessian.numpy().reshape(natoms * 3, natoms * 3)
                    / AU2EV
                    * BOHR2ANG
                    * BOHR2ANG
                )

            if args.noiserms and args.noiserms > 0.0:
                print(f"Adding noise to geometry with RMS {args.noiserms} Å")
                noise = rng.normal(0.0, 1.0, size=coords.shape)
                # Scale noise so RMS of per-atom Euclidean displacement equals noiserms
                current_rms = float(np.sqrt(np.mean(np.sum(noise * noise, axis=1))))
                scale = (args.noiserms / current_rms) if current_rms > 0.0 else 0.0
                displacement = scale * noise
                coords = coords + (displacement / BOHR2ANG)

            # use a small threshold to account for numerical errors
            eigvals = np.linalg.eigvals(initial_dft_hessian)
            dft_hessian_is_pd = np.all(eigvals > args.pdthresh)
            if not dft_hessian_is_pd:
                print("Initial DFT Hessian is not positive definite (Hartree/Bohr^2)")
                if np.all(eigvals > -1e-3):
                    print(" but is pd approximately (> -1e-3)")
                print(np.sort(eigvals)[:8])
                if args.pddftonly:
                    print("Skipping sample.")
                    continue

            results = []

            # first order:
            method_name = "NaiveSteepestDescent"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                geom_nsd = get_geom(atomssymbols, coords, args.coord, base_calc, args)
                opt = NaiveSteepestDescent(
                    geom_nsd,
                    max_cycles=args.max_cycles,
                    thresh=args.thresh,
                    # line_search=True,
                    out_dir=out_dir_method,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_nsd,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # first order, with backtracking line search
            method_name = "SteepestDescent"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_sd = get_geom(atomssymbols, coords, args.coord, base_calc, args)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = SteepestDescent(
                    geom_sd,
                    max_cycles=args.max_cycles,
                    thresh=args.thresh,
                    # line_search=True,
                    out_dir=out_dir_method,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_sd,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir,
                        verbose=args.verbose,
                    )
                )

            # first-order optimization (FIRE)
            method_name = "FIRE"
            if method_name in DO_METHOD:
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                print_header(cnt, method_name)
                geom_fire = get_geom(atomssymbols, coords, args.coord, base_calc, args)
                """
                https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.170201
                Structure optimization algorithm which is 
                significantly faster than standard implementations of the conjugate gradient method 
                and often competitive with more sophisticated quasi-Newton schemes
                It is based on conventional molecular dynamics 
                with additional velocity modifications and adaptive time steps.
                """
                opt = FIRE(
                    # Geometry providing coords, forces, energy
                    geom_fire,
                    max_cycles=args.max_cycles,
                    thresh=args.thresh,
                    # # Initial time step; adaptively scaled during optimization
                    # dt=0.1,
                    # # Maximum allowed time step when increasing dt
                    # dt_max=1,
                    # # Consecutive aligned steps before accelerating
                    # N_acc=2,
                    # # Factor to increase dt on acceleration
                    # f_inc=1.1,
                    # # Factor to reduce mixing a on acceleration; also shrinks dt on reset here
                    # f_acc=0.99,
                    # # Unused in this implementation; typical FIRE uses to reduce dt on reset
                    # f_dec=0.5,
                    # # Counter of aligned steps since last reset (start at 0)
                    # n_reset=0,
                    # # Initial mixing parameter for velocity/force mixing; restored on reset
                    # a_start=0.1,
                    # String poiting to a directory where optimization progress is
                    # dumped.
                    out_dir=out_dir_method,
                    allow_write=False,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_fire,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # first order:
            method_name = "ConjugateGradient"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_cg = get_geom(atomssymbols, coords, args.coord, base_calc, args)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = ConjugateGradient(
                    geom_cg,
                    max_cycles=args.max_cycles,
                    thresh=args.thresh,
                    # line_search=True,
                    out_dir=out_dir_method,
                    allow_write=False,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_cg,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # 2) No Hessian: BFGS with non-Hessian initial guess (unit) - pure quasi-Newton
            #    RFOptimizer accepts hessian_init and BFGS updates.
            method_name = "RFO-BFGS (unit init)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                # geom2 = Geometry(atomssymbols, coords, coord_type=args.coord)
                # geom2.set_calculator(CountingCalc(base_calc))
                geom_bfgsunit = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgsunit,
                    hessian_init="unit",
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=out_dir_method,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                    allow_write=False,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_bfgsunit,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # 3) Initial-only: RFO+BFGS with DFT Hessian at step 0
            method_name = "RFO-BFGS (DFT init)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_bfgsdft = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgsdft,
                    hessian_init=initial_dft_hessian,
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_bfgsdft,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # Finite difference Hessian
            method_name = "RFO-BFGS (NumHess init)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_bfgsnumhess = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                geom_bfgsnumhess.calculator.num_hess_kwargs = {"acc": 4}  # 2 or 4
                numerical_hessian = geom_bfgsnumhess.calculator.get_num_hessian(
                    geom_bfgsnumhess.atoms, geom_bfgsnumhess._coords
                )["hessian"]
                eigvals = np.linalg.eigvals(numerical_hessian)
                num_hessian_is_pd = np.all(eigvals > args.pdthresh)
                if not num_hessian_is_pd:
                    print("Numerical Hessian is not positive definite (Hartree/Bohr^2)")
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgsnumhess,
                    hessian_init=numerical_hessian,
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_bfgsnumhess,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # we provide your H_pred through the calculator and ask RFOptimizer to pull it:
            #    hessian_init='calc' gets Hessian from the calculator at step 0;
            #    hessian_recalc=k recomputes it every k steps.

            # Initial-only: RFO+BFGS with learned only at step 0
            method_name = "RFO-BFGS (learned init)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_bfgslearned = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgslearned,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_bfgslearned,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # Initial-only: RFO+BFGS with autograd Hessian only at step 0
            method_name = "RFO-BFGS (autograd init)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_bfgsautograd = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                # initial autograd hessian
                hessian_method_before = base_calc.hessian_method
                base_calc.hessian_method = "autograd"
                initial_autograd_hessian = base_calc.get_hessian(
                    geom_bfgsautograd.atoms, geom_bfgsautograd._coords
                )["hessian"]
                base_calc.hessian_method = hessian_method_before
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgsautograd,
                    hessian_init=initial_autograd_hessian,
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_bfgsautograd,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # Periodic replace: k=3
            method_name = "RFO-BFGS (learned k3)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_bfgslearnedk3 = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_bfgslearnedk3,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=3,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_bfgslearnedk3,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # Periodic replace: k=1 (every step)
            method_name = "RFO (learned)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfolearned = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfolearned,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfolearned,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # Periodic replace: k=1 (every step)
            method_name = "RFO (autograd)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                hessian_method_before = base_calc.hessian_method
                base_calc.hessian_method = "autograd"
                geom_rfoautograd = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfoautograd,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfoautograd,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )
                base_calc.hessian_method = hessian_method_before

            # Finite difference Hessian at every step
            method_name = "RFO (NumHess)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfonumhess = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                geom_rfonumhess.calculator.force_num_hessian()
                # numerical_hessian = geom_rfonumhess.calculator.get_num_hessian(geom_rfonumhess.atoms, geom_rfonumhess._coords)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfonumhess,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfonumhess,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            # Finite difference Hessian at every step with higher accuracy
            method_name = "RFO (NumHess 4)"
            if method_name in DO_METHOD:
                print_header(cnt, method_name)
                geom_rfonumhess4 = get_geom(
                    atomssymbols, coords, args.coord, base_calc, args
                )
                geom_rfonumhess4.calculator.num_hess_kwargs = {"acc": 4}  # 2 or 4
                geom_rfonumhess4.calculator.force_num_hessian()
                # numerical_hessian = geom_rfonumhess.calculator.get_num_hessian(geom_rfonumhess.atoms, geom_rfonumhess._coords)
                method_name_clean = clean_str(method_name)
                out_dir_method = os.path.join(out_dir, method_name_clean)
                opt = get_rfo_optimizer(
                    geom_rfonumhess4,
                    hessian_init="calc",
                    hessian_update="bfgs",
                    hessian_recalc=1,
                    out_dir=out_dir_method,
                    verbose=args.verbose,
                    thresh=args.thresh,
                    max_cycles=args.max_cycles,
                )
                results.append(
                    _run_opt_safely(
                        geom=geom_rfonumhess4,
                        opt=opt,
                        method_name=method_name,
                        out_dir=out_dir_method,
                        verbose=args.verbose,
                        dft_hessian_is_pd=dft_hessian_is_pd,
                    )
                )

            #########################################################
            # Sella
            #########################################################
            # sella_default_kwargs = dict(
            #     minimum=dict(
            #         delta0=1e-1,
            #         sigma_inc=1.15,
            #         sigma_dec=0.90,
            #         rho_inc=1.035,
            #         rho_dec=100,
            #         method="rfo",
            #         eig=False,
            #     ),
            #     saddle=dict(
            #         delta0=0.1,
            #         sigma_inc=1.15,
            #         sigma_dec=0.65,
            #         rho_inc=1.035,
            #         rho_dec=5.0,
            #         method="prfo",
            #         eig=True,
            #     ),
            # )
            # mol_ase = Atoms(numbers=atomic_numbers_np, positions=start_pos)
            # dyn = Sella(
            #     atoms=mol_ase,
            #     # constraints=cons,
            #     order=1,  # Explicitly search for first-order saddle point
            #     # eta=5e-5,  # Smaller finite difference step for higher accuracy
            #     # delta0=5e-3,  # Larger initial trust radius for TS search
            #     # gamma=0.1,  # Much tighter convergence for iterative diagonalization
            #     # rho_inc=1.05,  # More conservative trust radius adjustment
            #     # rho_dec=3.0,  # Allow larger trust radius changes
            #     # sigma_inc=1.3,  # Larger trust radius increases
            #     # sigma_dec=0.5,  # More aggressive trust radius decreases
            #     log_every_n=100,
            #     hessian_function=hessian_function,
            #     diag_every_n=diag_every_n,
            #     nsteps_per_diag=nsteps_per_diag,
            #     internal=internal,
            # )
            # _run_kwargs = dict(
            #     fmax=1e-3,
            #     steps=4000,
            # )
            # dyn.run(**_run_kwargs)

            ###########################
            # Pretty print
            print(
                f"\n{'Strategy':>24s} {'coords':>6} {'converged':>6} {'steps':>6} {'grads':>6} {'hessians':>6} {'s':>6}"
            )
            for r in results:
                try:
                    print(
                        f"{r['name']:>24s} {args.coord:>6} {str(r['converged']):>6s} {str(r['steps']):>6s} {str(r['grad_calls']):>6s} {str(r['hessian_calls']):>6s} {r['wall_time_s']:>12.3f}"
                    )
                except:
                    print(f"Error printing {r}")
                    print(r)

            # print positive definite status extra
            print(f"cnt_not_pd:")
            for r in results:
                _msg = f"{r['name']}: {r['cnt_not_pd']}"
                if r["hessian_calls"] is not None and r["hessian_calls"] > 0:
                    _msg += f" ({r['cnt_not_pd'] / r['hessian_calls'] * 100:.2f}%)"
                print(_msg)

            # Collect results with context for CSV
            for r in results:
                r_with_ctx = dict(r)
                r_with_ctx.update(
                    {
                        "sample_index": idx,
                        "coord": args.coord,
                        "source": source_label,
                    }
                )
                all_results.append(r_with_ctx)

            optims_done += 1

        #########################################################
        # Write aggregated CSV
        if len(all_results) > 0:
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            print(f"\nSaved relaxation results to: {csv_path}")
        else:
            print(f"\nNo results to save to {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"\nLoaded relaxation results from: {csv_path}")

    return df


def plot_results(df, out_dir, args):
    # remove all rows where the method is not in DO_METHOD
    df = df[df["name"].isin(DO_METHOD)]

    # print mean and std for each method and metric
    print("\nMean and std for each method and metric:")
    for metric in [
        "steps",
        "grad_calls",
        "hessian_calls",
        # "energy_calls",
        # "calculate_calls",
        # "calculate_energy_calls",
        # "calculate_gradient_calls",
        # "calculate_hessian_calls",
        "cnt_not_pd",
        "wall_time_s",
    ]:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for method in df["name"].unique():
            _d = df[df["name"] == method]
            print(f"{method}: {_d[metric].mean():.2f} ± {_d[metric].std():.2f}")

    # # print % of converged optims where dft_hessian_is_pd is False
    # print("\n% of converged BFGS+DFT init when dft_hessian_is_pd is True:")
    # _d = df[(df["dft_hessian_is_pd"] == True) & (df["name"] == "RFO-BFGS (DFT init)")]
    # print(f"{_d['converged'].mean() * 100:.2f}%")
    # print("\n% of converged BFGS+DFT init when dft_hessian_is_pd is False:")
    # _d = df[(df["dft_hessian_is_pd"] == False) & (df["name"] == "RFO-BFGS (DFT init)")]
    # print(f"{_d['converged'].mean() * 100:.2f}%")

    # # print % of converged optims where no pd Hessian was encountered in calculator
    # print("\n% of converged RFO (learned) when all Hessians were pd:")
    # _d = df[(df["cnt_not_pd"] == 0) & (df["name"] == "RFO (learned)")]
    # print(f"{_d['converged'].mean() * 100:.2f}%")
    # print("\n% of converged RFO (learned) when some Hessians were not pd:")
    # _d = df[(df["cnt_not_pd"] > 0) & (df["name"] == "RFO (learned)")]
    # print(f"{_d['converged'].mean() * 100:.2f}%")

    #########################################################
    # do plotting
    #########################################################
    print()
    sns.set_theme(style="whitegrid")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _remove_outliers(_d, metric_name):
        # remove the most extreme outliers (1st and 99th percentiles)
        low, high = _d[metric_name].quantile([0.01, 0.99])
        len_before = len(_d)
        _d = _d[(_d[metric_name] >= low) & (_d[metric_name] <= high)]
        len_after = len(_d)
        print(f"Removed {len_before - len_after} outliers for {metric_name}")
        return _d

    def _hex_to_rgba(hex_color, alpha):
        try:
            hc = str(hex_color).lstrip("#")
            r = int(hc[0:2], 16)
            g = int(hc[2:4], 16)
            b = int(hc[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return hex_color

    def _prepare_order(dfin, metric):
        dfo = dfin.copy()
        if (
            metric in ("steps", "wall_time_s")
            and "RFO (learned)" in dfo["name"].unique()
        ):
            mask = dfo["name"] == "RFO (learned)"
            k = int(min(5, mask.sum()))
            if k > 0:
                idxs = dfo.loc[mask, metric].nlargest(k).index
                dfo = dfo.drop(idxs)
        return (
            dfo.groupby("name")[metric]
            # .mean()
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def _series_for_method(dfin, method, metric):
        s = dfin[dfin["name"] == method][metric].dropna()
        if metric in ("steps", "wall_time_s") and method == "RFO (learned)":
            k = min(5, len(s))
            if k > 0:
                s = s.drop(s.nlargest(k).index)
        return s

    def _plot_metric_violin_plotly(_df, metric_name, save_path):
        _d = _df.dropna(subset=[metric_name])
        if len(_d) == 0:
            return
        # compute order by descending mean (most left -> least right)
        # after removing outliers for RFO (predicted) a.k.a. "RFO (learned)"
        d_for_order = _d.copy()
        if (
            metric_name in ("steps", "wall_time_s")
            and "RFO (learned)" in d_for_order["name"].unique()
        ):
            mask = d_for_order["name"] == "RFO (learned)"
            k = int(min(5, mask.sum()))
            if k > 0:
                idxs = d_for_order.loc[mask, metric_name].nlargest(k).index
                d_for_order = d_for_order.drop(idxs)
        order = (
            d_for_order.groupby("name")[metric_name]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig = go.Figure()

        display_order = []
        methods_plotted = []
        method_to_display_name = {}
        for method in order:
            series = _d[_d["name"] == method][metric_name].dropna()
            # For RFO (predicted) i.e. method == "RFO (learned)", drop top-5 highest values before plotting
            if metric_name in ("steps", "wall_time_s") and method == "RFO (learned)":
                k = min(5, len(series))
                if k > 0:
                    series = series.drop(series.nlargest(k).index)
            if len(series) == 0:
                continue
            display_name = RENAME_METHODS_PLOT.get(method, method)
            # Rename for Plotly display: learned -> predicted
            if method == "RFO (learned)":
                display_name = "RFO (predicted)"
            elif method == "RFO-BFGS (learned init)":
                display_name = "RFO-BFGS (predicted init)"
            # Keep the (ours) suffix for our methods
            # if method in ("RFO (learned)", "RFO-BFGS (learned init)"):
            #     display_name = f"{display_name} (ours)"
            color = METHOD_TO_COLOUR.get(method, "#1f77b4")
            display_order.append(display_name)
            methods_plotted.append(method)
            method_to_display_name[method] = display_name
            fig.add_trace(
                go.Violin(
                    y=series.astype(float),
                    name=display_name,
                    line_color=color,
                    # Defaults to a half-transparent variant of the line color
                    # fillcolor=color,
                    fillcolor=_hex_to_rgba(color, 0.25),
                    opacity=1.0,
                    box_visible=True,
                    meanline_visible=False,
                    spanmode="hard",
                    points="all",
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(color=color, opacity=0.5, size=4),
                    showlegend=False,
                )
            )
        # Add legend for categories (three colours) using dummy scatter traces
        categories_in_plot = []
        for m in methods_plotted:
            cat = METHOD_TO_CATEGORY.get(m)
            if cat is not None and cat not in categories_in_plot:
                categories_in_plot.append(cat)
        for cat in categories_in_plot:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=OPTIM_TO_COLOUR.get(cat, "#1f77b4"), size=10),
                    name=cat,
                    showlegend=True,
                )
            )

        # Annotate "ours" over highest values of selected methods
        target_methods = [
            "RFO-BFGS (learned init)",
            "RFO (learned)",
        ]
        if metric_name in _d.columns and len(_d[metric_name].dropna()) > 0:
            y_min = float(_d[metric_name].min())
            y_max = float(_d[metric_name].max())
        else:
            y_min = 0.0
            y_max = 0.0
        y_pad = 0.02 * (y_max - y_min) if y_max > y_min else 0.0
        for method in target_methods:
            if method in order:
                series_ann = _d[_d["name"] == method][metric_name].dropna()
                if len(series_ann) == 0:
                    continue
                y_top = float(series_ann.max())
                display_name = method_to_display_name.get(method, method)
                fig.add_annotation(
                    x=display_name,
                    y=y_top + y_pad,
                    text="<b>ours</b>",
                    showarrow=False,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=10),
                )

        # Bold our two methods in tick labels
        bold_targets = set()
        for m in ("RFO (learned)", "RFO-BFGS (learned init)"):
            if m in method_to_display_name:
                bold_targets.add(method_to_display_name[m])
        ticktext = [
            f"<b>{name}</b>" if name in bold_targets else name for name in display_order
        ]
        fig.update_layout(
            template="plotly_white",
            yaxis_title=METRIC_TO_LABEL.get(
                metric_name.lower(), metric_name.replace("_", " ").title()
            ),
            xaxis_title="",
            xaxis=dict(
                categoryorder="array",
                categoryarray=display_order,
                tickvals=display_order,
                ticktext=ticktext,
                tickangle=-25,
            ),
            legend=dict(
                x=1.0,
                y=1.0,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
            showlegend=True,
            height=600,
            width=1000,
            # margin=dict(t=0, b=0, l=0, r=0),
            margin=dict(t=0, b=40, l=20, r=0),
        )
        fig.write_image(save_path, scale=2)
        print(f"Saved\n {save_path}")

    def _plot_metric_violin_plotly_triple(_df, save_path):
        """
        Plotly violin plot with three subplots: steps to convergence, wall time, and wall time (subset)
        """

        # Data variants
        df_steps = _df.dropna(subset=["steps"]).copy()
        df_wall = _df.dropna(subset=["wall_time_s"]).copy()
        df_wall_comp = df_wall[
            df_wall["name"].isin(COMPETATIVE_METHODS_WALL_TIME)
        ].copy()

        order_steps = _prepare_order(df_steps, "steps")
        order_wall = _prepare_order(df_wall, "wall_time_s")
        order_wall_comp = _prepare_order(df_wall_comp, "wall_time_s")

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Steps to Convergence",
                "Wall Time [s]",
                "Wall Time [s] (Subset)",
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.0,
            # column_widths=[1.0, 1.0, 0.8], # make the last subplot less wide
        )

        categories_all = []

        # Build each subplot
        for col_idx, (df_i, metric_i, order_i) in enumerate(
            (
                (df_steps, "steps", order_steps),
                (df_wall, "wall_time_s", order_wall),
                (df_wall_comp, "wall_time_s", order_wall_comp),
            ),
            start=1,
        ):
            if len(df_i) == 0 or len(order_i) == 0:
                continue
            display_order = []
            methods_plotted = []
            method_to_display_name = {}
            for method in order_i:
                series = _series_for_method(df_i, method, metric_i)
                if len(series) == 0:
                    continue
                display_name = RENAME_METHODS_PLOT.get(method, method)
                if method == "RFO (learned)":
                    display_name = "RFO (predicted)"
                elif method == "RFO-BFGS (learned init)":
                    display_name = "RFO-BFGS (predicted init)"
                if method in ("RFO (learned)", "RFO-BFGS (learned init)"):
                    display_name = f"{display_name} (ours)"
                color = METHOD_TO_COLOUR.get(method, "#1f77b4")

                display_order.append(display_name)
                methods_plotted.append(method)
                method_to_display_name[method] = display_name

                fig.add_trace(
                    go.Violin(
                        y=series.astype(float),
                        name=display_name,
                        line_color=color,
                        fillcolor=_hex_to_rgba(color, 0.25),
                        opacity=1.0,
                        box_visible=True,
                        meanline_visible=False,
                        spanmode="hard",
                        points="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(color=color, opacity=0.5, size=4),
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

            # Legend categories to add later
            for m in methods_plotted:
                cat = METHOD_TO_CATEGORY.get(m)
                if cat is not None and cat not in categories_all:
                    categories_all.append(cat)

            # Axis formatting per subplot
            bold_targets = set()
            for m in ("RFO (learned)", "RFO-BFGS (learned init)"):
                if m in method_to_display_name:
                    bold_targets.add(method_to_display_name[m])
            ticktext = [
                f"<b>{name}</b>" if name in bold_targets else name
                for name in display_order
            ]
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=display_order,
                tickvals=display_order,
                ticktext=ticktext,
                tickangle=-25,
                row=1,
                col=col_idx,
            )
            # y-axis title
            fig.update_yaxes(
                title_text=METRIC_TO_LABEL.get(
                    metric_i.lower(), metric_i.replace("_", " ").title()
                ),
                row=1,
                col=col_idx,
            )

            # Annotate "ours" over highest values of selected methods for this subplot
            target_methods = [
                "RFO-BFGS (learned init)",
                "RFO (learned)",
            ]
            if metric_i in df_i.columns and len(df_i[metric_i].dropna()) > 0:
                y_min_i = float(df_i[metric_i].min())
                y_max_i = float(df_i[metric_i].max())
            else:
                y_min_i = 0.0
                y_max_i = 0.0
            y_pad_i = 0.02 * (y_max_i - y_min_i) if y_max_i > y_min_i else 0.0
            for method in target_methods:
                if method in order_i:
                    series_ann_i = _series_for_method(df_i, method, metric_i)
                    if len(series_ann_i) == 0:
                        continue
                    y_top_i = float(series_ann_i.max())
                    display_name_i = method_to_display_name.get(method, method)
                    fig.add_annotation(
                        x=display_name_i,
                        y=y_top_i + y_pad_i,
                        text="<b>ours</b>",
                        showarrow=False,
                        xref=f"x{col_idx}",
                        yref=f"y{col_idx}",
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(size=10),
                    )

        # Add category legend dummies
        for cat in categories_all:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=OPTIM_TO_COLOUR.get(cat, "#1f77b4"), size=10),
                    name=cat,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Match sizing/margins used in speed_comparison.py combined plot
        _height = 400
        _width = _height * 3
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            height=_height,
            width=_width,
            margin=dict(l=0, r=0, b=0, t=20),
            legend=dict(
                x=0.45,
                y=0.9,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
        )
        # # Reduce whitespace on the very left by moving y tick labels inside on subplot 1
        # fig.update_yaxes(
        #     ticklabelposition="inside",
        #     title_standoff=2,
        #     automargin=False,
        #     row=1,
        #     col=1,
        # )
        # Add subplot panel labels (a, b, c) at top-left outside each subplot
        dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.3]
        dom2 = (
            fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.35, 0.65]
        )
        dom3 = fig.layout.xaxis3.domain if hasattr(fig.layout, "xaxis3") else [0.7, 1.0]
        fig.add_annotation(
            x=dom1[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>a</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.add_annotation(
            x=dom2[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>b</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.add_annotation(
            x=dom3[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>c</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.write_image(save_path, width=_width, height=_height, scale=2)
        print(f"Saved\n {save_path}")

    def _plot_metric_violin_plotly_double(_df, save_path):
        """
        Plotly violin plot with two subplots: steps to convergence and wall time
        """
        # Data variants
        df_steps = _df.dropna(subset=["steps"]).copy()
        df_wall_comp = _df.dropna(subset=["wall_time_s"]).copy()
        df_wall_comp = df_wall_comp[
            df_wall_comp["name"].isin(COMPETATIVE_METHODS_WALL_TIME)
        ].copy()

        order_steps = _prepare_order(df_steps, "steps")
        order_wall_comp = _prepare_order(df_wall_comp, "wall_time_s")

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Steps to Convergence",
                "Wall Time [s] (Subset)",
            ),
            horizontal_spacing=0.07,
            vertical_spacing=0.0,
            # 9 vs 6 methods plotted
            column_widths=[1.0, 0.8],
        )

        categories_all = []

        for col_idx, (df_i, metric_i, order_i) in enumerate(
            (
                (df_steps, "steps", order_steps),
                (df_wall_comp, "wall_time_s", order_wall_comp),
            ),
            start=1,
        ):
            if len(df_i) == 0 or len(order_i) == 0:
                continue
            display_order = []
            methods_plotted = []
            method_to_display_name = {}
            for method in order_i:
                series = _series_for_method(df_i, method, metric_i)
                if len(series) == 0:
                    continue
                display_name = RENAME_METHODS_PLOT.get(method, method)
                if method == "RFO (learned)":
                    display_name = "RFO (predicted)"
                elif method == "RFO-BFGS (learned init)":
                    display_name = "RFO-BFGS (predicted init)"
                # if method in ("RFO (learned)", "RFO-BFGS (learned init)"):
                #     display_name = f"{display_name} (ours)"
                color = METHOD_TO_COLOUR.get(method, "#1f77b4")

                display_order.append(display_name)
                methods_plotted.append(method)
                method_to_display_name[method] = display_name

                # Violin plot
                fig.add_trace(
                    go.Violin(
                        y=series.astype(float),
                        name=display_name,
                        line_color=color,
                        fillcolor=_hex_to_rgba(color, 0.1),
                        opacity=1.0,
                        width=0.9,  # fixed width
                        box_visible=False,  # show the boxplot
                        meanline_visible=False,
                        spanmode="hard",
                        points="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(color=color, opacity=0.3, size=4),
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

                # Overlay median as a horizontal tick marker (no box)
                median_value = float(np.median(series.astype(float)))
                fig.add_trace(
                    go.Scatter(
                        x=[display_name],
                        y=[median_value],
                        mode="markers",
                        marker=dict(
                            symbol="line-ew",  # horizontal line marker
                            size=18,
                            color=color,
                            line=dict(color=color, width=2),
                            opacity=1.0,
                        ),
                        hovertemplate="median: %{y:.3g}<extra></extra>",
                        showlegend=False,
                    ),
                    row=1,
                    col=col_idx,
                )

            for m in methods_plotted:
                cat = METHOD_TO_CATEGORY.get(m)
                if cat is not None and cat not in categories_all:
                    categories_all.append(cat)

            bold_targets = set()
            for m in ("RFO (learned)", "RFO-BFGS (learned init)"):
                if m in method_to_display_name:
                    bold_targets.add(method_to_display_name[m])
            ticktext = [
                f"<b>{name}</b>" if name in bold_targets else name
                for name in display_order
            ]
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=display_order,
                tickvals=display_order,
                ticktext=ticktext,
                tickangle=-25,
                row=1,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text=METRIC_TO_LABEL.get(
                    metric_i.lower(), metric_i.replace("_", " ").title()
                ),
                row=1,
                col=col_idx,
            )

            # Annotate "ours" over highest values of selected methods for this subplot
            target_methods = [
                "RFO-BFGS (learned init)",
                "RFO (learned)",
            ]
            if metric_i in df_i.columns and len(df_i[metric_i].dropna()) > 0:
                y_min_i = float(df_i[metric_i].min())
                y_max_i = float(df_i[metric_i].max())
            else:
                y_min_i = 0.0
                y_max_i = 0.0
            y_pad_i = 0.01 * (y_max_i - y_min_i) if y_max_i > y_min_i else 0.0
            for method in target_methods:
                if method in order_i:
                    series_ann_i = _series_for_method(df_i, method, metric_i)
                    if len(series_ann_i) == 0:
                        continue
                    y_top_i = float(series_ann_i.max())
                    display_name_i = method_to_display_name.get(method, method)
                    fig.add_annotation(
                        x=display_name_i,
                        y=y_top_i + y_pad_i,
                        text="<b>ours</b>",
                        showarrow=False,
                        xref=f"x{col_idx}",
                        yref=f"y{col_idx}",
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(size=10),
                    )
        # metrics plotted

        for cat in categories_all:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=OPTIM_TO_COLOUR.get(cat, "#1f77b4"), size=10),
                    name=cat,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        _height = 400
        _width = _height * 2
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            height=_height,
            width=_width,
            margin=dict(l=0, r=0, b=0, t=20),
            # automargin=False,
            # title_standoff=1,
            # ticklabelposition="inside",
            legend=dict(
                x=0.48,
                y=0.95,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
        )
        fig.update_yaxes(title_standoff=1, row=1, col=1)
        fig.update_yaxes(title_standoff=1, row=1, col=2)

        # Panel labels a (left) and b (right)
        dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.48]
        dom2 = (
            fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.52, 1.0]
        )
        fig.add_annotation(
            x=dom1[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>a</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.add_annotation(
            x=dom2[0],
            y=0.999,
            xref="paper",
            yref="paper",
            text="<b>b</b>",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
        fig.write_image(save_path, width=_width, height=_height, scale=3)
        print(f"Saved\n {save_path}")

    def _get_best_method_by_mean(_df, metric_name, prefer="min"):
        # prefer: "min" for metrics where lower is better; "max" where higher is better
        if _df is None or len(_df) == 0 or metric_name not in _df.columns:
            return None
        try:
            means = _df.groupby("name")[metric_name].mean()
            means = means.dropna()
            if len(means) == 0:
                return None
            if prefer == "max":
                return str(means.idxmax())
            return str(means.idxmin())
        except Exception:
            return None

    # df = df.sort_values(
    #     by="name",
    #     key=lambda s: s.map({name: i for i, name in enumerate(DO_METHOD)}),
    # )
    # sort df by average number of steps (highest first)
    df = df.sort_values(by="steps", ascending=False)

    # for DFT init, add 5min to the wall time
    df.loc[df["name"] == "RFO-BFGS (DFT init)", "wall_time_s"] += 5 * 60

    # for metric in ["steps", "grad_calls", "hessian_calls", "wall_time_s"]:
    for metric in ["steps", "wall_time_s"]:
        # Also create interactive Plotly violin ordered by descending mean steps
        _plot_metric_violin_plotly(
            df.copy(),
            metric,
            os.path.join(plots_dir, f"{metric}_violin_plotly.png"),
        )
        if metric == "wall_time_s":
            # # Competitive methods only
            # _d_comp = df.copy()[df["name"].isin(COMPETATIVE_METHODS_WALL_TIME)]
            # if len(_d_comp) > 0:
            #     _plot_metric_violin_plotly(
            #         _d_comp,
            #         metric,
            #         os.path.join(plots_dir, f"{metric}_violin_plotly_competative.png"),
            #     )
            # # Combined 3-panel figure: Steps | Wall Time | Wall Time (Competitive)
            # _plot_metric_violin_plotly_triple(
            #     df.copy(),
            #     os.path.join(plots_dir, "steps_walltime_walltime_competative_plotly.png"),
            # )
            # Combined 2-panel figure: Steps | Wall Time (Competitive)
            _plot_metric_violin_plotly_double(
                df.copy(),
                os.path.join(plots_dir, "steps_walltime_competative_plotly.png"),
            )

    # Convergence rate per method
    if "converged" in df.columns:
        conv = (
            df.groupby("name")["converged"].mean().reset_index(name="convergence_rate")
        )
        # sort by human name:
        conv = conv.sort_values(
            by="name",
            key=lambda s: s.map({name: i for i, name in enumerate(DO_METHOD)}),
        )
        # explicit order for x based on DO_METHOD
        order = [name for name in DO_METHOD if name in conv["name"].unique()]
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=conv,
            x="name",
            y="convergence_rate",
            order=order,
            # palette="tab10",
            # hue="coord",
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Convergence rate")
        ax.set_title(f"Convergence rate ({COORD_TO_NAME[args.coord]})")
        plt.xticks(rotation=25, ha="right")
        # Highlight best method for convergence (highest mean converged), allow CLI override
        best_conv = None
        if len(conv) > 0 and "convergence_rate" in conv.columns:
            try:
                best_conv = str(conv.loc[conv["convergence_rate"].idxmax(), "name"])
            except Exception:
                best_conv = None
        chosen = args.highlight_method if args.highlight_method else best_conv
        if chosen is not None:
            for lbl in ax.get_xticklabels():
                if lbl.get_text() == chosen:
                    lbl.set_fontweight("bold")
        plt.tight_layout()
        fname = os.path.join(plots_dir, "convergence_rate.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved\n {fname}")

    return df


def compute_zero_point_energy(df, out_dir, source_label, args):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xyz",
        default="ts1x-val.lmdb",
        help="input geometry in form of .xyz or folder of .xyz files or lmdb path",
        type=str,
        required=False,
    )
    ap.add_argument(
        "--coord",
        default="redund",
        choices=["cart", "redund", "dlc", "tric"],
        help="coordinate system",
        type=str,
        required=False,
    )
    ap.add_argument("--max_samples", type=int, default=15)
    ap.add_argument("--max_cycles", type=int, default=150)
    ap.add_argument("--debug", type=bool, default=False)
    ap.add_argument("--redo", type=bool, default=False)
    ap.add_argument("--verbose", type=bool, default=False)
    ap.add_argument("--thresh", type=str, default="gau")
    ap.add_argument(
        "--pddftonly",
        type=bool,
        default=False,
        help="only run optimizers when dft hessian is positive definite",
    )
    ap.add_argument(
        "--pdpredonly",
        type=bool,
        default=False,
        help="Stop optimization early when learned Hessian is not positive definite",
    )
    ap.add_argument(
        "--pdthresh",
        type=float,
        default=0,
        help="Threshold for positive definiteness of DFT Hessian",
    )
    ap.add_argument(
        "--noiserms",
        type=float,
        default=0.0,
        help="Per-atom RMS displacement (Å) added to geometry before Hessian; 0 disables noise",
    )
    ap.add_argument(
        "--highlight_method",
        type=str,
        default=None,
        help="If set, bold this method's x-label in plots",
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to ckpt file",
    )
    args = ap.parse_args()

    if args.ckpt_path is None:
        # ckpt_path = "/ssd/Code/ReactBench/ckpt/hesspred/alldatagputwoalphadrop0droppathrate0projdrop0-394770-20250806-133956.ckpt"
        args.ckpt_path = "/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746.ckpt"
        wandb_id = args.ckpt_path.split("/")[-1].split(".")[0].split("-")[1]

    # Determine source label for logging
    source_label = os.path.splitext(args.xyz.split("/")[-1])[0]
    out_dir = os.path.join(
        ROOT_DIR,
        "runs_relaxation",
        source_label
        + "_"
        + wandb_id
        + "_"
        + args.coord
        + "_"
        + args.thresh.replace("_", "")
        + "_"
        + str(args.max_samples)
        + "_pddft"
        + str(args.pddftonly)
        + "_pdpred"
        + str(args.pdpredonly)
        + "_pdthresh"
        + str(args.pdthresh),
    )

    df = do_relaxations(out_dir, source_label, args)
    plot_results(df, out_dir, args)
    compute_zero_point_energy(df, out_dir, source_label, args)


if __name__ == "__main__":
    main()
