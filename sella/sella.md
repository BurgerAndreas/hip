[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/zadorlab/sella)

# Sella

Sella is a utility for finding first order saddle points.

Here’s a high-level, implementation-focused overview of the Sella codebase, distilled from the paper so you can see how it all fits together:

1. Core algorithm modules

   * Internal-coordinate generation:
     Builds a *redundant* set of bond-stretch, angle-bend and dihedral coordinates (Appendix A).  It automatically detects near-linear angles, replaces them with impropers, and-even mid-optimization-drops and rebuilds the coordinate system as needed, inserting “dummy” atoms and associated constraints to keep everything well-defined .
   * Hessian diagonalization & update:
     Implements an iterative Rayleigh-Ritz (Olsen) solver in the *nonredundant* internal-coordinate basis to find the lowest-curvature mode without ever forming the full Hessian .  All curvature information from those Hessian-vector products is folded into a multi-secant TS-BFGS update of the approximate Hessian so subsequent steps stay accurate .
   * Geodesic stepping:
     Displacements in internal coordinates are realized by tracing geodesics on the curved manifold of valid geometries, yielding much more reliable steps (especially in highly redundant bases) than straight Newton updates .

2. Constrained saddle-point optimizer

   * Null-space SQP for constraint enforcement: constraints (distances, angles, even custom expressions) enter as Lagrange multipliers; at each iteration a small correction step brings you back toward the constraint manifold, then an SQP step (in the orthogonal subspace) optimizes the energy .
   * RS-PRFO for saddle points: splits the unconstrained space into maximization (reaction coordinate) and minimization subspaces, solving small eigenproblems to step uphill in one and downhill in the others-all within the SQP framework .
   * Trust-region control uses the infinity norm on the combined SQP step (so adding more redundant coordinates doesn’t shrink your allowed step size) and rescales via a simple Newton solve when needed .


An example script
```python
#!/usr/bin/env python3

from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT

from sella import Sella, Constraints

# Set up your system as an ASE atoms object
slab = fcc111('Cu', (5, 5, 6), vacuum=7.5)
add_adsorbate(slab, 'Cu', 2.0, 'bridge')

# Optionally, create and populate a Constraints object.
cons = Constraints(slab)
for atom in slab:
    if atom.position[2] < slab.cell[2, 2] / 2.:
        cons.fix_translation(atom.index)

# Set up your calculator
slab.calc = EMT()

# Set up a Sella Dynamics object
dyn = Sella(
    slab,
    constraints=cons,
    trajectory='test_emt.traj',
)

dyn.run(1e-3, 1000)
```




## How to cite

If you use our code in publications, please cite the revelant work(s). (1) is recommended when Sella is used for solids or in heterogeneous catalysis, (3) is recommended for molecular systems.

1. Hermes, E., Sargsyan, K., Najm, H. N., Zádor, J.: Accelerated saddle point refinement through full exploitation of partial Hessian diagonalization. Journal of Chemical Theory and Computation, 2019 15 6536-6549. https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b00869
2. Hermes, E. D., Sagsyan, K., Najm, H. N., Zádor, J.: A geodesic approach to internal coordinate optimization. The Journal of Chemical Physics, 2021 155 094105. https://aip.scitation.org/doi/10.1063/5.0060146
3. Hermes, E. D., Sagsyan, K., Najm, H. N., Zádor, J.: Sella, an open-source automation-friendly molecular saddle point optimizer. Journal of Chemical Theory and Computation, 2022 18 6974-6988. https://pubs.acs.org/doi/10.1021/acs.jctc.2c00395

