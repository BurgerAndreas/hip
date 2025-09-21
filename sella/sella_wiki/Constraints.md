# Constraints overview

Sella natively implements five types of constraints:
1. Fixing the barycenter of one or more atoms in one or more directions (`Translation`)
2. Fixing the distance between two atoms (`Bond`)
3. Fixing the bending angle defined by three atoms (`Angle`)
4. Fixing the dihedral angle defined by four atoms (`Dihedral`)
5. Fixing net rotation of two or more atoms (`Rotation`)

Additionally, Sella allows users to manually define their own constraints by defining a function which accepts as input the atomic positions and returns the residual of the constraint.

If compatible ASE Constraint objects are found attached to the `Atoms` object passed to `Sella`, Sella will convert those constraints into its native format. Note that if the ASE `Atoms` object has an attached constraint of a type which is not currently supported, Sella will raise a `RuntimeError`.

By default, all constraints are assumed to be strict equality constraints. It is also possible to define inequality constraints for `Bond`, `Angle`, `Dihedral`, and custom constraint types.

## Specifying constraints

These constraints can be passed to the `Sella` object:
```python
from sella import Sella, Constraints
...
cons = Constraints(myatoms)
cons.fix_translation(...)
cons.fix_bond(...)
cons.fix_angle(...)
cons.fix_dihedral(...)
cons.fix_rotation(...)
dyn = Sella(myatoms, constraints=cons)
```

If you are using internal coordinates and wish to manually specify some internal coordinates ([[see page on Internal Coordinates|Internal Coordinates]]), then the `Constraints` object must be passed to the `Internals` object constructor. In this case, it is not necessary to pass the `Constraints` object to `Sella`.

```python
from sella import Sella, Constraints, Internals

cons = Constraints(myatoms)
cons.fix_...
internals = Internals(myatoms, cons=cons)
internals.add_...
dyn = Sella(myatoms, internal=internals)
```

### Translation constraints

The `fix_translation` method has the following arguments:
```python
fix_translation(index=None, dim=None, target=None, replace_ok=True)
````
 - `index`: `int` or tuple of `int`s. The index or indices of atoms whose barycenter is to be fixed. Defaults to all atoms.
 - `dim`: `int` between 0 and 2 inclusive. The dimension in which the barycenter is to be fixed. Defaults to all dimensions (adds 3 constraints).
 - `target`: `float`. The position to which the barycenter is to be fixed. If no value is provided, the current barycenter will be used.
 - `replace_ok`: `bool`. If the provided coordinate has already been constrained, a value of `True` will result in the constraint being replaced, and a value of `False` will cause an exception to be raised.

The `fix_translation` method can be used to fix the barycenter of one or more atoms in one or more Cartesian directions. This can be used to fix individual atoms in place by passing only a single atom index.

Examples:

```python
# Fix atom 0 in the x, y, and z directions
cons.fix_translation(0)

# Fix atom 1 in only the x direction (note: 0, 1, and 2 correspond with x, y, and z respectively)
cons.fix_translation(1, dim=0)

# Fix barycenter of atoms 2, 3, and 4 in all three directions
cons.fix_translation((2, 3, 4))

# Fix net translation of the whole system in all three directions
cons.fix_translation()
```

### Bond constraints

The `fix_bond` method has the following arguments:
```python
fix_bond(indices, ncvecs=None, mic=None, target=None, comparator='eq', replace_ok=True)
```
 - `indices`: Tuple of 2 `int`s. Indices of the atoms between which the bond distance is to be fixed.
 - `ncvecs`: Tuple containing a 3-vector of integers. Indicates which periodic image of the unit cell the bond crosses into. Mutually exclusive with `mic`.
 - `mic`: `bool`. Whether to use the minimum image convention for systems with periodic boundary conditions. Disabled by default.
 - `target`: `float`. The value to which the bond distance will be fixed. Defaults to the current bond distance.
 - `comparator`: `str`. `'eq'` results in a strict equality constraint, `'lt'` will constrain to less than or equal to the target value, and `'gt'` will constrain to greater than or equal to the target value.
 - `replace_ok`: `bool`. If the provided coordinate has already been constrained, a value of `True` will result in the constraint being replaced, and a value of `False` will cause an exception to be raised.

The `fix_bond` method is used to constrain the distance between two atoms to a fixed distance.

Examples:

```python
# Fixes the in-cell distance between atoms 0 and 1 to the current distance.
cons.fix_bond((0, 1))

# Fixes the in-cell distance between atoms 1 and 2 to 1.5 Angstrom.
# NOTE: This doesn't immediately change the geometry!
cons.fix_bond((1, 2), target=1.5)

# Fixes the minimum image convention distance between atoms 2 and 3 to the current distance.
cons.fix_bond((2, 3), mic=True)

# Fixes the distance between atoms 3 and 4 through the (2, 0, -1) periodic cell direction to the current distance.
# NOTE: the mic and ncvecs keyword arguments are incompatible, and cannot be used simultaneously.
cons.fix_bond((3, 4), ncvecs=((2, 0, -1),))
```

### Angle constraints

The `fix_angle` method has the following arguments:
```python
fix_angle(indices, ncvecs=None, mic=None, target=None, comparator='eq', replace_ok=True)
```
 - `indices`: Tuple of 3 `int`s. Indices of the atoms that form the bending angle.
 - `ncvecs`: Tuple containing 2 3-vectors of integers. Indicates which periodic images of the unit cell the bonds that form the angle cross into. Mutually exclusive with `mic`.
 - `mic`: `bool`. Whether to use the minimum image convention for systems with periodic boundary conditions. Disabled by default.
 - `target`: `float`. The value to which the bending angle will be fixed. Defaults to the current bending angle.
 - `comparator`: `str`. `'eq'` results in a strict equality constraint, `'lt'` will constrain to less than or equal to the target value, and `'gt'` will constrain to greater than or equal to the target value.
 - `replace_ok`: `bool`. If the provided coordinate has already been constrained, a value of `True` will result in the constraint being replaced, and a value of `False` will cause an exception to be raised.

The `fix_angle` method is used to constrain the bending angle formed by three atoms to a specific value.

Examples:

```python
# Fix the in-cell 0-1-2 bending angle to its current value
cons.fix_angle((0, 1, 2))

# Fix the in-cell 1-2-3 bending angle to 120 degrees
# NOTE: This doesn't immediately change the geometry!
cons.fix_angle((1, 2, 3), target=120)

# Fix the minimum-image convention 2-3-4 bending angle to its current value
cons.fix_angle((2, 3, 4), mic=True)

# Fix the 3-4-5 bending angle to its current value,
# where the 3-4 bond is in the (2, 0, -1) cell direction,
# and the 4-5 bond is in the (0, 1, -1) cell direction
# NOTE: the mic and ncvecs keyword arguments are incompatible, and cannot be used simultaneously.
cons.fix_angle((3, 4, 5), ncvecs=((2, 0, -1), (0, 1, -1)))
```

### Dihedral constraints

The `fix_dihedral` method has the following arguments:
```python
fix_dihedral(indices, ncvecs=None, mic=None, target=None, comparator='eq', replace_ok=True)
```
 - `indices`: Tuple of 4 `int`s. Indices of the atoms that form the dihedral angle.
 - `ncvecs`: Tuple containing 3 3-vectors of integers. Indicates which periodic images of the unit cell the bonds that form the dihedral cross into. Mutually exclusive with `mic`.
 - `mic`: `bool`. Whether to use the minimum image convention for systems with periodic boundary conditions. Disabled by default.
 - `target`: `float`. The value to which the dihedral angle will be fixed. Defaults to the current dihedral angle.
 - `comparator`: `str`. `'eq'` results in a strict equality constraint, `'lt'` will constrain to less than or equal to the target value, and `'gt'` will constrain to greater than or equal to the target value.
 - `replace_ok`: `bool`. If the provided coordinate has already been constrained, a value of `True` will result in the constraint being replaced, and a value of `False` will cause an exception to be raised.

The `fix_dihedral` method is used to constrain the dihedral formed by four atoms to a specific value.

Examples:

```python
# Fix the in-cell 0-1-2-3 dihedral angle to its current value
cons.fix_dihedral((0, 1, 2, 3))

# Fix the in-cell 1-2-3-4 dihedral to 180 degrees
# NOTE: This doesn't immediately change the geometry!
cons.fix_dihedral((1, 2, 3, 4), target=180)

# Fix the minimum-image convention 2-3-4-5 dihedral to its current value
cons.fix_dihedral((2, 3, 4, 5), mic=True)

# Fix the 3-4-5-6 dihedral angle to its current value,
# where the 3-4 bond is in the (2, 0, -1) cell direction,
# the 4-5 bond is in the (0, 1, -1) cell direction,
# and the 5-6 bond is in the (-1, 0, 2) cell direction
# NOTE: the mic and ncvecs keyword arguments are incompatible, and cannot be used simultaneously.
cons.fix_dihedral((3, 4, 5, 6), ncvecs=((2, 0, -1), (0, 1, -1), (-1, 0, 2)))
```

### Rotational constraints

The `fix_rotation` method has the following arguments:
```python
fix_rotation(indices=None, axis=None)
````
 - `index`: Tuple of `int`s. The indices of atoms that form the cluster whose rotation is to be fixed.
 - `dim`: `int` between 0 and 2 inclusive. The index of the rotation axis to be constrained. Defaults to all axes (adds 3 constraints). Note that the rotational axis orientations are arbitrary, and therefore it is usually not meaningful to manually specify this value.

Unlike other constraints, it is not currently possible to specify a target value or replace rotational constraints.

The `fix_rotation` method is used to constrain the rotation of a cluster of atoms in one or more directions.

NOTE: While it is technically possible to constrain rotation in just one or two directions, it is currently difficult to control which rotational directions are actually being constrained. Users are advised only to use `fix_rotation` if they wish to remove all rotational degrees of freedom from a cluster of atoms.

NOTE: `fix_rotation` does not currently support periodic boundary conditions. It is currently only possible to constrain rotation of a cluster of atoms in the same periodic cell.

Examples:

```python
# Fix net rotation of atoms 0 and 1
cons.fix_rotation((0, 1))

# Fix net rotation of atoms 0, 1, and 2
cons.fix_rotation((0, 1, 2))

# Fix net rotation of the whole system
cons.fix_rotation()
```

### Custom constraints

Custom coordinates can be defined and constrained using the `make_internal` function and the `fix_custom` method of the `Constraints` object.

The `make_internal` function has the following arguments:
```python
def make_internal(name, fun, nindices, use_jit=True, jac=None, hess=None, **kwargs)
```
 - `name`: `str`. The name of your custom coordinate.
 - `fun`: A function with the signature `fun(pos, **kwargs)`, where `pos` is an `nindices x 3` array containing the positions of the atoms in the constraint which returns a `float`. See example below for more details.
 - `nindices`: `int`. The number of indices in the constraint type. For example, `Bond` has `nindices=2` and `Angle` has `nindices=3`.
 - `use_jit`: `bool`. Whether to optimize the function and its derivatives using the JAX JIT. Enabled by default. Only disable if you get a JIT error.
 - `jac`: A function with the signature `jac(pos, **kwargs)` which returns the Jacobian of `fun`. This is calculated automatically with JAX auto-differentiation if not provided. This is usually not necessary to provide, unless auto-differentiation fails for some reason.
 - `hess`: A function with the call signature `hess(pos, **kwargs)` which returns the Hessian of `fun`. This is calculated automatically with JAX auto-differentiation if not provided. This is usually not necessary to provide, unless auto-differentiation fails for some reason.
 - `**kwargs`: Any other keyword arguments that must be passed to `fun`, `jac`, and `hess`. Note that all instances of your custom class will have the same `kwargs`; if you need to use different `kwargs`, you will need to define a new class.

As an example, this is a function which can be used to constrain two bonds to be the same distance as each other:
```python
def bond_diff(pos):
    dx1 = pos[1] - pos[0]
    dx2 = pos[2] - pos[1]
    return dx1 @ dx1 - dx2 @ dx2

BondDiff = make_internal('BondDiff', bond_diff, 3)
```

Note that in order to use JAX auto-differentiation, care must be taken when defining the coordinate function. For example, NumPy methods (such as `np.sqrt`) should be replaced by the `jax.numpy` equivalent (i.e. `import jax.numpy as jnp` and `jnp.sqrt`). Additionally, `jax.numpy` arrays are immutable; this may affect how you implement your coordinate function. See [the JAX documentation](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html) for more information.

If you wish to specify an explicit Jacobian function, it must return the Jacobian in the same `nindices x 3` shape as the input positions.
For example:
```python
def bond_diff_jac(pos):
    return 2 * np.array([
        pos[0] - pos[1],
        pos[2] - pos[0],
        pos[1] - pos[2],
    ])
```

The explicit Hessian function must return the Hessian in a `nindices x 3 x nindices x 3` array.
For example (in this example, the Hessian is constant, though this is not generally the case):
```python
def bond_diff_hess(pos):
    hess = np.zeros((*pos.shape, *pos.shape), dtype=float)
    hess[0, :, 0] = 2 * np.eye(3)
    hess[2, :, 2] = -2 * np.eye(3)
    hess[0, :, 1] = hess[1, :, 0] = -2 * np.eye(3)
    hess[1, :, 2] = hess[2, :, 1] = 2 * np.eye(3)
    return hess
````

Note that since we are explicitly defining the Jacobian and Hessian in these examples, we may use standard `numpy` methods instead of the `jax.numpy` alternatives, though this may make it impossible to use the JIT (`use_jit` keyword).

Once a coordinate class has been created, a specific coordinate can be defined by creating an instance, for example
```python
my_coord = BondDiff((0, 1, 2))
```

This will create a coordinate which represents the difference between the `0-1` bond distance and the `1-2` bond distance.
You may check the accuracy of the Jacobian and Hessian with the `my_coord.check_gradient` and `my_coord.check_hessian` methods, respectively.
These methods accept an ASE `Atoms` object as an argument and will raise a warning and return a value of `False` if a discrepancy is found; otherwise, they will return `True`.

The constraint can be added to your Constraints instance using the `fix_other` method:
```python
def fix_other(coord, target=None, comparator='eq', replace_ok=True)
```
 - `coord`: An instance of your custom coordinate class.
 - `target`: `float`. The value to which the coordinate will be fixed. Defaults to the current value of that coordinate.
 - `comparator`: `str`. `'eq'` results in a strict equality constraint, `'lt'` will constrain to less than or equal to the target value, and `'gt'` will constrain to greater than or equal to the target value.
 - `replace_ok`: `bool`. If the provided coordinate has already been constrained, a value of `True` will result in the constraint being replaced, and a value of `False` will cause an exception to be raised.

For example, if we wish to constrain the `0-1` bond distance to be the same as the `1-2` bond distance, we may add that constraint in the following way:
```python
cons.fix_other(my_coord, target=0)
```
