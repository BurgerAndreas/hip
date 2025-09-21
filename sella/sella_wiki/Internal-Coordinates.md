# Internal Coordinates Overview

Sella supports optimization (both minimization and saddle point optimization) using a basis of internal coordinates (ICs). IC optimization can be extremely important when your system of interest is molecule-like, i.e. it contains covalent bonds that form a sparsely-connected network spanning all atoms in the system.

IC optimization works by defining a complete (usually over-complete) set of IC variables, which are a combination of bond distances, bending angles, and dihedral angles. These variables are then used as an auxiliary coordinates system for the optimization algorithm. This is beneficial because certain displacements (such as perturbing a dihedral angle in the middle of a long chain of atoms) are much "shorter" in IC-space than they are in Cartesian space, and so it is possible to take much larger steps using IC optimization.

## Using internal coordinates in Sella

Internal coordinate optimization can be enabled by passing the `internal=True` keyword argument to `Sella`:

```python
from sella import Sella

dyn = Sella(myatoms, internal=True)
```

By default, Sella will automatically determine all bond, angle, and dihedral ICs from the molecular geometry. It is also possible to manually specify some ICs:

```python
from sella import Sella, Internals

internals = Internals(myatoms)
internals.add_translation(...)
internals.add_bond(...)
internals.add_angle(...)
internals.add_dihedral(...)
internals.add_rotation(...)

dyn = Sella(myatoms, internal=internals)
```

### Translational coordinates

Translational internal coordinates can be added with the `add_translation` method. These coordinates use the barycenter of a cluster of atoms in a given Cartesian direction as an internal coordinate. It is possible to add a pure Cartesian direction as an internal coordinate by specifying only a single atom index.

Examples:

```python
# Add the x, y, and z Cartesian translations of atom 0 to the ICs
internals.add_translation(0)

# Add the x (but not y or z) Cartesian translation of atom 1 to the ICs
internals.add_translation(1, dim=0)

# Add the three Cartesian translations of the barycenter of the 2-3-4 cluster to the ICs
internals.add_translation((2, 3, 4))
```

### Bond coordinates

Bond coordinates can be added with the `add_bond` method.

Examples:

```python
# Adds the in-cell 0-1 bond to the ICs
internals.add_bond((0, 1))

# Adds the minimum-image convention 1-2 bond to the ICs
internals.add_bond((1, 2), mic=True)

# Adds the 2-3 bond in the (2, 0, -1) periodic cell direction to the ICs
# NOTE: the mic and ncvecs keyword arguments are incompatible, and cannot be used simultaneously.
internals.add_bond((2, 3), ncvecs=((2, 0, -1),))
```

### Angle coordinates

Bending angle coordinates can be added with the `add_angle` method.

Examples:

```python
# Adds the in-cell 0-1-2 bending angle to the ICs
internals.add_angle((0, 1, 2))

# Adds the minimum-image convention 1-2-3 bending angle to the ICs
internals.add_angle((1, 2, 3), mic=True)

# Adds the 2-3-4 bending angle to the ICs, where
# the 2-3 bond is in the (2, 0, -1) periodic cell direction,
# and the 3-4 bond is in the (0, 1, -1) periodic cell direction.
# NOTE: the mic and ncvecs keyword arguments are incompatible, and cannot be used simultaneously.
internals.add_angle((2, 3, 4), ncvecs=((2, 0, -1), (0, 1, -1)))
```

### Dihedral coordinates

Dihedral angle coordinates can be added with the `add_dihedral` method.

Examples:

```python
# Adds the in-cell 0-1-2-3 dihedral angle to the ICs
internals.add_dihedral((0, 1, 2, 3))

# Adds the minimum-image convention 1-2-3-4 dihedral angle to the ICs
internals.add_dihedral((1, 2, 3, 4), mic=True)

# Adds the 2-3-4-5 dihedral angle to the ICs, where
# the 2-3 bond is in the (2, 0, -1) periodic cell direction,
# the 3-4 bond is in the (0, 1, -1) periodic cell direction,
# and the 4-5 bond is in the (-1, 0, 2) periodic cell direction.
# NOTE: the mic and ncvecs keyword arguments are incompatible, and cannot be used simultaneously.
internals.add_dihedral((2, 3, 4, 5), ncvecs=((2, 0, -1), (0, 1, -1), (-1, 0, 2)))
```

### Rotational coordinates

The rotational coordinates of a cluster of atoms can be added with the `add_rotation` method.
NOTE: This is not yet fully tested.
NOTE: Rotational coordinates do not currently support periodic boundary conditions. All atoms in the cluster must be in the same unit cell.

Examples:

```python
# Add the rotational coordinates of the cluster consisting of atoms 0 and 1
internals.add_rotation((0, 1))

# Add the rotational coordinates of the entire system
internals.add_rotation()
```
