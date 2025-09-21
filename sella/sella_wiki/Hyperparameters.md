# Hyperparameter overview

Sella has a number of hyperparameters that can potentially have a large impact on performance.
For more information, read our publication located [on ChemRxiv](https://chemrxiv.org/articles/Accelerated_Saddle_Point_Refinement_Through_Full_Exploitation_of_Partial_Hessian_Diagonalization/9750512).

```python
from sella import Sella
...
dyn = Sella(myatoms,
            ...
            eta=1e-4,        # Finite difference step size
            update='jd0',    # Hessian diagonalization method
            gamma=0.4,       # Convergence criterion for iterative diagonalization
            delta0=1.3e-3,   # Initial trust radius
            rho_inc=1.035,   # Threshold for increasing trust radius
            rho_dec=5.0,     # Threshold for decreasing trust radius
            sigma_inc=1.15,  # Trust radius increase factor
            sigma_dec=0.65)  # Trust radius decrease factor
```


## Iterative diagonalization

For saddle point refinement, it is necessary to identify the leftmost eigenvector of the true Hessian matrix.
Sella does this using an iterative diagonalization method.

### `eta`

`eta` controls the magnitude of the finite difference step that is used by the iterative diagonalization routine.
By default, this value is `1e-4` Angstrom.

If `eta` is chosen to be too large, then third- or higher-order effects will introduce error into the finite difference curvature estimates.
This may result in the iterative diagonalization failing to converge, which will increase the number of gradient evaluations needed drastically.

If `eta` is chosen to be too small, then the true change in the gradients may be smaller than the intrinsic error present in the gradients.
This is particularly true when using an electronic structure theory method like DFT that must be solved self-consistently.
When the gradients are analytical and exact, e.g. when using a molecular dynamics force field, `eta` can be reduced.

### `method`

`method` specifies which iterative diagonalization method used by Sella.
Valid values are `lanczos`, `gd` (Generalized Davidson), and `jd0` (Jacobi-Davidson without inner iterations).
The default is `jd0`.
Changing this default is not advised, as `jd0` is expected to have superior performance in all cases.

### `gamma`

`gamma` specifies the convergence criteria for the iterative eigensolver.
The approximate eigenvector (Ritz vector) is considered converged when:

![equation](http://latex.codecogs.com/svg.latex?%5C%7C%20%5Cmathbf%7Br%7D%5E%7B%28j%29%7D%20%5C%7C_2%20%5Cle%20%5Cgamma%20%5Cleft%7C%20%5Ctheta%5E%7B%28j%29%7D%20%5Cright%7C)

where **r**<sup>(_j_)</sup> is the residual vector of the _j_ th Ritz vector and θ<sup>(_j_)</sup> is the corresponding Ritz value.

Smaller values of `gamma` will result in more iterative diagonalization steps, but will also improve the quality of the approximate Hessian, which will reduce the number of geometry steps needed to converge to the saddle point.

## Trust radius

Sella uses a trust radius method to restrict step sizes.

### `delta0`

`delta0` specifies the initial trust radius in units of Angstrom per degree of freedom.
For example, a system of 5 atoms with no constraints and without periodic boundary conditions (see the [Constraints](https://github.com/zadorlab/sella/wiki/Constraints) page) will have `3 * 5 - 6 == 9` degrees of freedom, so the initial trust radius will be `1.3e-3 * 9 == 1.17e-2` Angstrom.

### `rho_inc`, `sigma_inc`, `rho_dec`, `sigma_dec`

After every geometry step, Sella evaluates the ratio of the true change in energy and the predicted change in energy:

![equation](https://latex.codecogs.com/svg.latex?\rho_k&space;=&space;\frac{\epsilon_{k&plus;1}&space;-&space;\epsilon_k}{\mathbf{g}_k^T&space;\mathbf{s}_k&space;&plus;&space;\frac{1}{2}&space;\mathbf{s}_k^T&space;\mathbf{B}_k&space;\mathbf{s}_k})

For the definition of symbols, see our paper linked at the top of this page.

When ρ<sub>k</sub> is between `rho_inc**-1` and `rho_inc`, the trust radius is set to the magnitude of the last step multiplied by `sigma_inc`; if this is smaller than the current trust radius, then the trust radius is not changed.

When ρ<sub>k</sub> is less than `rho_dec**-1` or greater than `rho_dec`, the trust radius is set to the magnitude of the last step multiplied by `sigma_dec`; if this is less than `eta` (see above), then the trust radius is instead set to `eta`.



