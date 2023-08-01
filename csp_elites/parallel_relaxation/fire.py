from __future__ import annotations

import warnings
from typing import Tuple, List

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer


class OverridenFire:
    def __init__(self, atoms=None, restart=None, logfile='-', trajectory=None,
                 dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
                 finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
                 position_reset_callback=None, force_consistent=None):
        # Optimizer.__init__(self, atoms, restart, logfile, trajectory,
        #                master, force_consistent=force_consistent)

        self.dt = dt

        self.Nsteps = 0
        self.maxstep = 10 # todo: what is maxstep

        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.downhill_check = downhill_check
        self.position_reset_callback = position_reset_callback
        self.fmax = 0.1

    # def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
    #              dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
    #              finc=1.1, fdec=0.5,
    #              astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
    #              position_reset_callback=None, force_consistent=None):
    #     super().__init__(atoms, restart=None, logfile='-', trajectory=None,
    #              dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
    #              finc=1.1, fdec=0.5,
    #              astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
    #              position_reset_callback=None, force_consistent=None)

    # @jit(nopython=True)
    def step_override(
        self,
        system_size: Tuple[int],
        f: np.ndarray,
        e: np.ndarray,
        v: np.ndarray,
        e_last: np.ndarray,
        r_last: np.ndarray,
        v_last: np.ndarray,
        dt: np.ndarray,
        Nsteps: np.ndarray,
        a: np.ndarray,
    ):
        # atoms = self.atoms

        # if f is None:
        #     f = atoms.get_forces()

        if v is None:
            v = np.zeros((len(f), len(f[0]), 3))
            # if downhill_check:
            #     e_last = atoms.get_potential_energy(
            #         force_consistent=force_consistent)
            #     r_last = atoms.get_positions().copy()
            #     v_last = v.copy()
        else:
            is_uphill = False
            # if downhill_check:
                # e = atoms.get_potential_energy(
                #     force_consistent=force_consistent)
                # # Check if the energy actually decreased
                # if e > e_last:
                #     # If not, reset to old positions...
                #     if position_reset_callback is not None:
                #         position_reset_callback(atoms, r_last, e,
                #                                      e_last)
                #     atoms.set_positions(r_last)
                #     is_uphill = True
                # e_last = atoms.get_potential_energy(
                #     force_consistent=force_consistent)
                # r_last = atoms.get_positions().copy()
                # v_last = v.copy()

            vf = np.vdot(f, v)
            if vf > 0.0 and not is_uphill:
                v = (1.0 - a) * v + a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(v, v))
                if Nsteps > self.Nmin:
                    dt = min(dt * self.finc, self.dtmax)
                    a *= self.fa
                Nsteps += 1
            else:
                v[:] *= 0.0
                a = self.astart
                dt *= self.fdec
                Nsteps = 0

        v += dt * f
        dr = dt * v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr

        return v, e_last, r_last, v_last, dt, Nsteps, a, dr

    def step(self, atoms, f=None):
        # atoms = self.atoms
        #
        # if f is None:
        #     f = atoms.get_forces()

        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
            # if self.downhill_check:
            #     self.e_last = atoms.get_potential_energy(
            #         force_consistent=self.force_consistent)
            #     self.r_last = atoms.get_positions().copy()
            #     self.v_last = self.v.copy()
        else:
            is_uphill = False
            # if self.downhill_check:
            #     e = atoms.get_potential_energy(
            #         force_consistent=self.force_consistent)
            #     # Check if the energy actually decreased
            #     if e > self.e_last:
            #         # If not, reset to old positions...
            #         if self.position_reset_callback is not None:
            #             self.position_reset_callback(atoms, self.r_last, e,
            #                                          self.e_last)
            #         atoms.set_positions(self.r_last)
            #         is_uphill = True
            #     self.e_last = atoms.get_potential_energy(
            #         force_consistent=self.force_consistent)
            #     self.r_last = atoms.get_positions().copy()
            #     self.v_last = self.v.copy()

            vf = np.vdot(f, self.v)
            if vf > 0.0 and not is_uphill:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr
        r = atoms._get_positions_unit_cell_filter()
        atoms.set_position(r + dr)
        self.dump((self.v, self.dt))

    def converged(self, forces):
        """Did the optimization converge?"""
        return np.max((forces ** 2).sum(axis=2), axis=1)  < self.fmax ** 2

    def irun(self, list_of_atoms: List[Atoms]):
        converged_atoms = []
        forces, _, _ = None, None, None

        # run the algorithm until converged or max_steps reached
        while not self.converged(forces) and self.nsteps < self.max_steps:

            # compute the next step
            self.step()
            self.nsteps += 1

            # let the user inspect the step and change things before logging
            # and predicting the next step
            yield False

            # log the step
            self.log()
            self.call_observers()

        # finally check if algorithm was converged
        yield self.converged(forces)
