# IRC Overview

The IRC is the path which connects a first order saddle point to its adjacent minima. The trajectory is sketched by solving the first-order differential equation

![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%5Cmathbf%7Bq%7D%7D%7Bdt%7D%20%3D%20%5Cmathbf%7Bv%7D%28t%29)

where **v**(_t_) is the leftmost eigenvector of the Hessian for the initial structure (which is a first order saddle point) and the normalized gradient vector for all other points.

IRC calculations are very simple to run:
```python
from sella import IRC
from ase.io import read

my_atoms = read('converged_saddle_point_geometry.xyz')
opt = IRC(my_atoms, trajectory='irc.traj', dx=0.1, eta=1e-4, gamma=0.4)
opt.run(fmax=0.1, steps=1000, direction='forward')
opt.run(fmax=0.1, steps=1000, direction='reverse')
```

For standard IRC, `my_atoms` must be a first order saddle point. If `my_atoms` is not a first order saddle point, Sella's `IRC` class will print a warning and proceed anyway. The resulting trajectory is known as a pseudo-IRC.

The keywords `eta` and `gamma` are described on the [hyperparameters documentation page](https://github.com/zadorlab/sella/wiki/Hyperparameters). `dx` controls the distance between images along the IRC in units of `Angstrom * sqrt(amu)`. Larger values of `dx` will reduce the number of steps needed to construct the full IRC, but the resulting trajectory will deviate from the true IRC.

`IRC` inherits from the ASE `Optimizer` class, and largely follows its API. Unlike `Optimizer`, the `IRC.run()` method accepts the keyword `direction`, which specifies whether  IRC should be run in the "forward" or "reverse" directions.

Note that which direction is forward and which is reverse is completely arbitrary! However, if you are interested in the full IRC, you can call `opt.run` twice: once with `direction='forward'` and once with `direction='reverse'`. This guarantees that you will obtain the whole IRC, contained in the trajectory file (in this example, `irc.traj`).

If an IRC inner iteration fails, the IRC calculation stops by default. However, if the keyword argument `keep_going` is set as `keep_going=True`, the IRC optimization will print a warning that the trajectory is no longer accurate, but it will continue with the IRC. This helps when IRC is used as a way of finding connected minima, and the exactness of the entire IRC trajectory is of lesser relevance.