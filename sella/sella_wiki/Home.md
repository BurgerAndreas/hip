# Sella documentation

Sella is a utility primarily intended for refining approximate saddle point geometries.
It relies on ASE to communicate with electronic structure theory packages such as [NWChem](https://github.com/nwchemgit/nwchem) or [Quantum Espresso](https://github.com/QEF/q-e).

Before learning how to use Sella, you should familiarize yourself with [ASE](https://wiki.fysik.dtu.dk/ase/).
We suggest you follow the [ASE tutorial](https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html).

## Installing Sella

`pip install sella`

This requires Python 3.6+ and Numpy.
ASE, SciPy, JAX, jaxlib, and a newer version of Numpy will be installed automatically if necessary.

## Running Sella

Sella does not provide a shell utility or a graphical interface.
It must be invoked using Python scripts.
Here is an example script:

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

Sella provides an interface that is compatible with the [ASE Optimizer class](https://wiki.fysik.dtu.dk/ase/ase/optimize.html) interface.

***

For an overview of the hyperparameters available in Sella, see the [hyperparameters](https://github.com/zadorlab/sella/wiki/Hyperparameters) page.

To learn how to perform constrained saddle point refinement, see the [constraints](https://github.com/zadorlab/sella/wiki/Constraints) page.