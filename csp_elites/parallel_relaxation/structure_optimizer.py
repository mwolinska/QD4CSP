from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import List

import numpy as np
from ase import Atoms
from chgnet.model import CHGNet
from chgnet.model.dynamics import TrajectoryObserver, CHGNetCalculator
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.parallel_relaxation.fire import OverridenFire
from csp_elites.parallel_relaxation.structure_to_use import atoms_to_test
from csp_elites.parallel_relaxation.unit_cell_filter import AtomsFilterForRelaxation


class MultiprocessOptimizer:
    def __init__(self ):
        self.overriden_optimizer = OverridenFire()
        self.atoms_filter = AtomsFilterForRelaxation()
        self.model = CHGNet.load()
        self.model.graph_converter.atom_graph_cutoff = 6

    def relax(self, list_of_atoms: List[Atoms], n_relaxation_steps: int, verbose: bool = False):

        forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
        original_cells = copy.deepcopy([atoms.cell for atoms in list_of_atoms])

        forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
            forces_from_chgnet=forces,
            stresses_from_chgnet=stresses,
            list_of_atoms=list_of_atoms,
            original_cells=original_cells,
            current_atom_cells=[atoms.cell for atoms in list_of_atoms],
            cell_factors=np.array([1] * len(list_of_atoms)),
        )
        v = None
        Nsteps=0
        dt = np.full((len(forces)), 0.1)
        a = np.full((len(forces)), 0.1)

        # trajectories = [TrajectoryObserver(atoms) for atoms in list_of_atoms]
        all_relaxed = False

        n_relax_steps = np.zeros(len(forces))
        # fmax_over_time = np.zeros((n_relaxation_steps, len(list_of_atoms)))
        fmax_over_time = []
        trajectories = defaultdict(list)
        while not all_relaxed:
            fmax = np.max((forces ** 2).sum(axis=2), axis=1) ** 0.5
            # fmax_over_time = fmax_over_time.at[Nsteps].set(fmax)
            fmax_over_time.append(fmax)
            if verbose:
                print(Nsteps, energies * 24, fmax)

            v, dt, n_relax_steps, a, dr = \
                self.overriden_optimizer.step_override(forces, v, dt, n_relax_steps, a)


            positions = self.atoms_filter.get_positions(
                original_cells,
                [atoms.cell for atoms in list_of_atoms],
                list_of_atoms,
                np.array([1] * len(list_of_atoms)),
            )

            list_of_atoms = self.atoms_filter.set_positions(original_cells,
                                                            list_of_atoms, np.array(positions + dr),
                                                            np.array([1] * len(list_of_atoms)))

            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
            forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
                forces_from_chgnet=forces,
                stresses_from_chgnet=stresses,
                list_of_atoms=list_of_atoms,
                original_cells=original_cells,
                current_atom_cells=[atoms.cell for atoms in list_of_atoms],
                cell_factors=np.array([1] * len(list_of_atoms)),
            )
            trajectories["forces"].append(forces)
            trajectories["energies"].append(energies)
            trajectories["stresses"].append(stresses)
            # trajectories = self._update_trajectories(trajectories, forces, energies, stresses)
            converged_mask = self.overriden_optimizer.converged(forces, fmax)
            Nsteps += 1
            all_relaxed = self._end_relaxation(Nsteps, n_relaxation_steps, converged_mask)

        final_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        reformated_output = []
        for i in range(len(final_structures)):
            reformated_output.append(
                {"final_structure": final_structures[i],
                 "trajectory": {
                     "energies":trajectories["energies"][-1][i],
                     "forces": trajectories["forces"][-1][i],
                     "stresses": trajectories["stresses"][-1][i],
                 }
                }
            )

        return reformated_output, list_of_atoms

    def _evaluate_list_of_atoms(self, list_of_atoms: List[Atoms]):
        if isinstance(list_of_atoms[0], Atoms):
            list_of_structures = [AseAtomsAdaptor.get_structure(list_of_atoms[i]) for i in range(len(list_of_atoms))]
        elif isinstance(list_of_atoms[0], Structure):
            list_of_structures = list_of_atoms

        predictions = self.model.predict_structure(list_of_structures, batch_size=5)
        if isinstance(predictions, dict):
            predictions = [predictions]

        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])
        return forces, energies, stresses

    def _update_trajectories(self, trajectories: List[TrajectoryObserver], forces, energies, stresses) -> List[TrajectoryObserver]:
        for i in range(len(trajectories)):
            trajectories[i].energies.append(energies[i])
            trajectories[i].forces.append(forces)
            trajectories[i].stresses.append(stresses)
        return trajectories


    def _end_relaxation(self, nsteps: int, max_steps: int, forces_mask:np.ndarray):
        return (nsteps >= max_steps) or forces_mask.all()


if __name__ == '__main__':

    optimizer = MultiprocessOptimizer()

    # optimizer_ref = StructOptimizer()
    # optimizer.relax(atoms_for_ref, fmax=0.1, steps=10)

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1840", final=True)
    atoms_2 = AseAtomsAdaptor.get_atoms(one_structure)
    atoms_2.calc = CHGNetCalculator()
    atoms_2.rattle(0.1)
    # atoms_2_copy = copy.deepcopy(atoms_2)
    # atoms_2_copy.calc = CHGNetCalculator()


    n_relax_steps = 100
    # optimizer_ref.relax(atoms_2_copy, fmax=0.2, steps=n_relax_steps)
    tic = time.time()
    relax_results, atoms_returned = optimizer.relax([atoms_to_test], n_relaxation_steps=n_relax_steps, verbose=True)
    print(time.time() - tic)
    #
    # tic = time.time()
    # predictions = optimizer.model.predict_structure([AseAtomsAdaptor.get_structure(atoms_to_test), AseAtomsAdaptor.get_structure(atoms_2)])
    # print(time.time() - tic)
    # print()