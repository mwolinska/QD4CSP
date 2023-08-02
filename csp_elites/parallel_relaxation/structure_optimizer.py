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
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from chgnet.model import StructOptimizer, CHGNet
from chgnet.model.dynamics import TrajectoryObserver, CHGNetCalculator
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


from csp_elites.parallel_relaxation.fire import OverridenFire
from csp_elites.parallel_relaxation.structure_to_use import atoms_to_test, atoms_for_ref
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
        self.model = model

    def relax(
            self,
            atoms: Structure | Atoms,
            fmax: float | None = 0.1,
            steps: int | None = 500,
            relax_cell: bool | None = True,
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

    def override_relax(self, list_of_atoms: List[Atoms], n_relaxation_steps: int, reference: Atoms):
        system_size = (len(list_of_atoms), (len(list_of_atoms[0])))
        forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
        original_cells = copy.deepcopy([atoms.cell for atoms in list_of_atoms])
        exp_filter = ExpCellFilter(reference)
        exp_filter.get_forces()
        forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
            forces_from_chgnet=forces,
            stresses_from_chgnet=stresses,
            list_of_atoms=list_of_atoms,
            original_cells=original_cells,
            current_atom_cells=[atoms.cell for atoms in list_of_atoms],
            cell_factors=np.array([1] * len(list_of_atoms)),

        )

        v, e_last, r_last, v_last = None, None, None, None
        Nsteps=0
        dt = np.full((len(forces)), 0.1)
        a = np.full((len(forces)), 0.1)

        trajectories = [TrajectoryObserver(atoms) for atoms in list_of_atoms]
        all_relaxed = False
        converged_atoms = []
        n_relax_steps = np.zeros(len(forces))
        while not all_relaxed:
            fmax = np.max((forces ** 2).sum(axis=2), axis=1) ** 0.5

            print(Nsteps, energies * 24, fmax)

            v, e_last, r_last, v_last, dt, n_relax_steps, a, dr = \
                self.overriden_optimizer.step_override(system_size, forces, energies, v, e_last, r_last, v_last, dt,
                                             n_relax_steps, a)
            positions = self.atoms_filter.get_positions(
                original_cells,
                [atoms.cell for atoms in list_of_atoms],
                list_of_atoms,
                np.array([len(atoms) for atoms in list_of_atoms])
            )

            exp_filter.set_positions(positions[0]+dr[0])

            list_of_atoms = self.atoms_filter.set_positions(original_cells,
                                                            list_of_atoms, positions + dr,
                                                            np.array([1] * len(list_of_atoms)))

            positions_to_check = self.atoms_filter.get_positions(
                original_cells,
                [atoms.cell for atoms in list_of_atoms],
                list_of_atoms,
                np.array([len(atoms) for atoms in list_of_atoms])
            )

            # print((exp_filter.get_positions() == positions_to_check).all())

            forces, energies, stresses = self._evaluate_list_of_atoms(list_of_atoms)
            forces, _ = self.atoms_filter.get_forces_exp_cell_filter(
                forces_from_chgnet=forces,
                stresses_from_chgnet=stresses,
                list_of_atoms=list_of_atoms,
                original_cells=original_cells,
                current_atom_cells=[atoms.cell for atoms in list_of_atoms],
                cell_factors=np.array([len(atoms) for atoms in list_of_atoms]),
            )
            # print((np.isclose(forces, exp_filter.get_forces())).all())

            trajectories = self._update_trajectories(trajectories, forces, energies, stresses)

            converged_mask = self.overriden_optimizer.converged(forces, fmax)
            Nsteps += 1
            all_relaxed = self._end_relaxation(Nsteps, n_relaxation_steps, converged_mask)
        # print(Nsteps)
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

        predictions = self.model.predict_structure(list_of_structures, batch_size=len(list_of_atoms))
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
        return list_of_atoms

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

    optimizer = MultiprocessOptimizer(CHGNet.load())
    # optimizer.relax(atoms_for_ref, fmax=0.1, steps=10)

    with MPRester(api_key="4nB757V2Puue49BqPnP3bjRPksr4J9y0") as mpr:
        one_structure = mpr.get_structure_by_material_id("mp-1840", final=True)
    atoms_2 = AseAtomsAdaptor.get_atoms(one_structure)
    atoms_2.rattle(0.1)

    # optimizer.relax(copy.deepcopy(atoms_2), fmax=0.1, steps=10)

    optimizer.override_relax(
        [
            # atoms_to_test,
            copy.deepcopy(atoms_to_test),
         # atoms_2,
            copy.deepcopy(atoms_2)
         ],
        n_relaxation_steps=10,
        reference=atoms_for_ref,
    )
    print()


# atoms_2

# *Force-consistent energies used in optimization.
# FIRE:    0 10:58:36     -221.365768*      27.4550
# FIRE:    1 10:58:36     -213.811432*      72.1598
# FIRE:    2 10:58:37     -220.890610*      46.9382
# FIRE:    3 10:58:37     -223.090599*      12.0744
# FIRE:    4 10:58:37     -222.509079*      25.3691
# FIRE:    5 10:58:37     -222.777260*      21.8202
# FIRE:    6 10:58:38     -223.167206*      14.9737
# FIRE:    7 10:58:38     -223.460083*       5.5494
# FIRE:    8 10:58:38     -223.496864*       7.0563
# FIRE:    9 10:58:39     -223.504440*       6.7553
# FIRE:   10 10:58:39     -223.518791*       6.1688

# 0 [-221.36577] [27.45499335]
# 1 [-185.19832] [108.21517887]
# 2 [-213.16785] [113.01248233]
# 3 [-223.0724] [19.99845704]
# 4 [-219.15082] [52.93323407]
# 5 [-220.3374] [46.67237474]
# 6 [-222.14378] [34.42728458]
# 7 [-223.63028] [16.03534909]
# 8 [-223.84814] [12.32245007]
# 9 [-223.86288] [11.81443116]

# forces sum
# -287.7707272001559
# 219.0701883158656
# 16.821549701150587
# -127.32726118629944
# -110.23114149438885
# -76.29279370455487
# -25.397218483294033
# 36.27582636388728
# 34.81758358025121

# atoms_to_test

#       Step     Time          Energy         fmax
# *Force-consistent energies used in optimization.
# FIRE:    0 10:19:17     -219.036552*      11.0884
# FIRE:    1 10:19:17     -217.946686*      55.1223
# FIRE:    2 10:19:17     -214.898552*      95.1003
# FIRE:    3 10:19:18     -221.270966*      10.8314
# FIRE:    4 10:19:18     -218.204613*      52.5300
# FIRE:    5 10:19:18     -219.091301*      45.5657
# FIRE:    6 10:19:18     -220.403915*      31.9720
# FIRE:    7 10:19:18     -221.413307*      12.5107
# FIRE:    8 10:19:19     -221.458809*      12.3343
# FIRE:    9 10:19:19     -221.474442*      11.7322
# FIRE:   10 10:19:19     -221.503052*      10.5659

# 0 [-219.03656] [11.08840322]
# 1 [-217.442] [59.6096384]
# 2 [-213.95834] [103.7137654]
# 3 [-221.35048] [10.66211866]
# 4 [-217.5156] [57.79763708]
# 5 [-218.58191] [50.34789201]
# 6 [-220.1701] [35.64318585]
# 7 [-221.40701] [14.49050357]
# 8 [-221.47302] [13.38759661]
# 9 [-221.49048] [12.73690958]
