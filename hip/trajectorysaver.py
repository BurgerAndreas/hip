import contextlib
import io
import warnings
from typing import Tuple

import numpy as np

from ase.atoms import Atoms
from ase.parallel import world


class MyTrajectory:
    """ASE Trajectory lookalike that saves to memory instead of disk.

    Parameters:

    filename: str
        The name of the file.  Traditionally ends in .traj.
    mode: str
        The mode.  'r' is read mode, the file should already exist, and
        no atoms argument should be specified.
        'w' is write mode.  The atoms argument specifies the Atoms
        object to be written to the file, if not given it must instead
        be given as an argument to the write() method.
        'a' is append mode.  It acts as write mode, except that
        data is appended to a preexisting file.
    atoms: Atoms object
        The Atoms object to be written in write or append mode.
    properties: list of str
        If specified, these calculator properties are saved in the
        trajectory.  If not specified, all supported quantities are
        saved.  Possible values: energy, forces, stress, dipole,
        charges, magmom and magmoms.
    master: bool
        Controls which process does the actual writing. The
        default is that process number 0 does this.  If this
        argument is given, processes where it is True will write.
    comm: Communicator object
        Communicator to handle parallel file reading and writing.
    comm: MPI communicator
        MPI communicator for this trajectory writer, by default world.
        Passing a different communicator facilitates writing of
        different trajectories on different MPI ranks.
    save_full_atoms: bool
        If True, the full Atoms object is saved to the trajectory.
        If False, only the positions are saved.
    """

    def __init__(
        self,
        atoms,
        properties=None,
        master=None,
        comm=world,
        save_full_atoms=False,
    ):
        if master is None:
            master = comm.rank == 0

        self.filename = ""
        self.mode = "w"
        self.atoms = atoms
        self.properties = properties
        self.master = master
        self.comm = comm

        self.description = {}
        self.header_data = None
        self.multiple_headers = False

        # added
        self.trajectory = []
        self.trajectory_dummies = []
        self.trajectory_properties = []
        self.save_full_atoms = save_full_atoms

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def set_description(self, description):
        self.description.update(description)

    def _open(self, filename, mode):
        pass

    def write(self, atoms=None, dummies=None, **kwargs):
        """Write the atoms to self instead of the file.

        If the atoms argument is not given, the atoms object specified
        when creating the trajectory object is used.

        Use keyword arguments to add extra properties::

            writer.write(atoms, energy=117, dipole=[0, 0, 1.0])
        """
        if atoms is None:
            atoms = self.atoms

        _traj_step = []
        for image in atoms.iterimages():
            if self.save_full_atoms:
                _traj_step.append(atoms)
            else:
                _traj_step.append(atoms.get_positions())
        if len(_traj_step) == 1:
            _traj_step = _traj_step[0]
        self.trajectory.append(_traj_step)
        if dummies is not None:
            if self.save_full_atoms:
                self.trajectory_dummies.append(dummies)
            else:
                self.trajectory_dummies.append(dummies.get_positions())
        self.trajectory_properties.append(kwargs)
        return

    def close(self):
        pass

    def __len__(self):
        return len(self.trajectory)
