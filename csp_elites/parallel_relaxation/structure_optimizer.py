from __future__ import annotations

import contextlib
import copy
import io
import sys
from typing import List

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import ExpCellFilter
from ase.optimize.optimize import Optimizer
from chgnet.model import StructOptimizer, CHGNet
from chgnet.model.dynamics import TrajectoryObserver
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


from csp_elites.parallel_relaxation.fire import OverridenFire
from csp_elites.parallel_relaxation.unit_cell_filter import AtomsFilterForRelaxation


class MultiprocessOptimizer(StructOptimizer):
    def __init__(self, model: CHGNet | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
        atoms: Atoms = None # not optional
                 ):
        super().__init__(model, optimizer_class, use_device, stress_weight)
        self.overriden_optimizer = OverridenFire(atoms)
        self.atoms_filter = AtomsFilterForRelaxation()

    def relax(
            self,
            atoms: Structure | Atoms,
            fmax: float | None = 0.1,
            steps: int | None = 500,
            relax_cell: bool | None = False,
            save_path: str | None = None,
            trajectory_save_interval: int | None = 1,
            verbose: bool = True,
            **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.1
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            save_path (str | None): The path to save the trajectory.
                Default = None
            trajectory_save_interval (int | None): Trajectory save interval.
                Default = 1
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor.get_atoms(atoms)

        atoms.calc = self.calculator  # assign model used to predict forces

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=trajectory_save_interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)
        for key in struct.site_properties:
            struct.remove_site_property(property_name=key)
        struct.add_site_property(
            "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
        )
        return {"final_structure": struct, "trajectory": obs}


    def override_relax(self, list_of_atoms: List[Atoms], n_relaxation_steps: int):
        system_size = (len(list_of_atoms), (len(list_of_atoms[0])))
        forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
        original_cells = [atoms.cell for atoms in list_of_atoms]
        forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
            forces_from_chgnet=forces,
            stresses_from_chgnet=stresses,
            list_of_atoms=list_of_atoms,
            original_cells=original_cells,
            current_atom_cells=[atoms.cell for atoms in list_of_atoms],
            cell_factors=np.array([len(atoms) for atoms in list_of_atoms]),

        )

        velocity, e_last, r_last, v_last = None, None, None, None
        dt, Nsteps, a = 0.1, 0, 0.1

        trajectories = [TrajectoryObserver(atoms) for atoms in list_of_atoms]
        all_relaxed = False
        converged_atoms = []
        while not all_relaxed:
            v, e_last, r_last, v_last, dt, Nsteps, a, dr = \
                self.overriden_optimizer.step_override(system_size, forces, energies, velocity, e_last, r_last, v_last, dt,
                                             Nsteps, a)
            positions = self.atoms_filter.get_positions(
                original_cells,
                [atoms.cell for atoms in list_of_atoms],
                list_of_atoms,
                np.array([len(atoms) for atoms in list_of_atoms])
            )
            list_of_atoms = self.atoms_filter.set_positions(original_cells,
                                                            list_of_atoms, positions +dr,
                                                            np.array([len(atoms) for atoms in list_of_atoms]))

            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
            forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
                forces_from_chgnet=forces,
                stresses_from_chgnet=stresses,
                list_of_atoms=list_of_atoms,
                original_cells=original_cells,
                current_atom_cells=[atoms.cell for atoms in list_of_atoms],
                cell_factors=np.array([len(atoms) for atoms in list_of_atoms]),

            )
            trajectories = self._update_trajectories(trajectories, forces, energies, stresses)

            converged_mask = self.overriden_optimizer.converged(forces)
            Nsteps += 1
            all_relaxed = self._end_relaxation(Nsteps, n_relaxation_steps, converged_mask)

        final_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]
        reformated_output = []
        for i in range(len(final_structures)):
            reformated_output.append(
                {"final_structure": final_structures[i],
                 "trajectory": trajectories[i],
                }
            )

        return reformated_output

    def _evaluate_list_of_atoms(self, list_of_atoms: List[Atoms]):
        list_of_structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in list_of_atoms]

        predictions = chgnet.predict_structure(list_of_structures, batch_size=len(list_of_atoms))
        if isinstance(predictions, dict):
            predictions = [predictions]

        forces = np.array([pred["f"] for pred in predictions])
        energies = np.array([pred["e"] for pred in predictions])
        stresses = np.array([pred["s"] for pred in predictions])
        return forces, energies, stresses

    def _set_atom_calulators(self, list_of_atoms: List[Atoms],
        forces: np.ndarray,
        energies: np.ndarray,
        stresses: np.ndarray
    ):
        for i in range(len(list_of_atoms)):
            calculator = SinglePointCalculator(
                list_of_atoms[i], energy=energies[i], forces=forces[i], stress=stresses[i],
            )
            list_of_atoms[i].calc = calculator
        return atoms

    def _update_positions_post_relaxation_step(self, list_of_atoms: List[Atoms], position_change: np.ndarray):
        for i in range(len(list_of_atoms)):
            r = list_of_atoms[i].get_positions()
            list_of_atoms[i].set_positions(r + position_change[i])
        return list_of_atoms

    def _update_trajectories(self, trajectories: List[TrajectoryObserver], forces, energies, stresses) -> List[TrajectoryObserver]:
        for i in range(len(trajectories)):
            trajectories[i].energies.append(energies[i])
            trajectories[i].forces.append(forces)
            trajectories[i].stresses.append(stresses)
        return trajectories


    def _end_relaxation(self, nsteps: int, max_steps: int, forces_mask:np.ndarray):
        return (nsteps >= max_steps) or forces_mask.all()


if __name__ == '__main__':
    chgnet = CHGNet.load()
    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1341203", final=True)

    atoms = AseAtomsAdaptor.get_atoms(one_structure)

    atoms.rattle(0.7)
    atoms_2 = copy.deepcopy(atoms)

    optimizer = MultiprocessOptimizer()

    # optimizer.relax(atoms, fmax=0.1)

    optimizer.override_relax(
        [atoms_2, copy.deepcopy(atoms_2)],
        n_relaxation_steps=10
    )
