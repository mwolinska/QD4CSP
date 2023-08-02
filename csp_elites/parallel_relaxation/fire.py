from __future__ import annotations

import warnings
from typing import Tuple, List

import numpy as np
from ase import Atoms
from ase.optimize import FIRE
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

        if v is None:
            v = np.zeros((len(f), len(f[0]), 3))

        else:
            is_uphill = False

            # vf = np.vdot(f, v)
            print(f[1].sum())
            f_reshaped = f.reshape(len(f), -1)
            v_reshaped = v.reshape(len(v), -1)
            vf = np.diag(f_reshaped @ v_reshaped.T)

            vf_positive_mask = np.array(vf > 0.0, dtype=int).reshape((-1, 1, 1))
            vf_negative_mask = np.array(vf <= 0.0, dtype=int).reshape((-1, 1, 1))
            # update v

            vdot_ff = np.diag(f.reshape(len(f), -1) @ f.reshape(len(f), -1).T)
            vdot_vv = np.diag(v.reshape(len(f), -1) @ v.reshape(len(f), -1).T)
            v_positive = (1.0 - a).reshape((-1, 1, 1)) * (
                        v * vf_positive_mask) + a.reshape((-1, 1, 1)) * (
                                     f * vf_positive_mask) / np.sqrt(
                vdot_ff).reshape(-1, 1, 1) * np.sqrt(vdot_vv).reshape((-1, 1, 1))
            v_negative = v * vf_negative_mask * 0
            v = v_positive + v_negative

            Nsteps_bigger_than_n_min = np.array(Nsteps > self.Nmin, dtype=int)
            Nsteps_smaller_than_n_min = np.array(Nsteps <= self.Nmin, dtype=int)
            dt_1 = np.min(np.vstack([dt * vf_positive_mask.reshape(-1) * Nsteps_bigger_than_n_min * self.finc, [self.dtmax] * len(f)]), axis=0)
            # dt[vf_negative_mask.reshape(-1).astype(bool)] *= self.fdec
            dt_1b = dt * vf_positive_mask.reshape(-1) * Nsteps_smaller_than_n_min
            dt_2 = dt * vf_negative_mask.reshape(-1) * self.fdec
            dt = dt_1 + dt_1b + dt_2

            Nsteps[vf_positive_mask.reshape(-1).astype(bool)] += 1
            Nsteps_1 = Nsteps * vf_positive_mask.reshape(-1)
            Nsteps_2 = Nsteps * vf_negative_mask.reshape(-1) * 0
            Nsteps = Nsteps_1 + Nsteps_2

            # update a
            a_1 = a * vf_positive_mask.reshape(-1) * Nsteps_bigger_than_n_min * self.fa
            a_2 = a * vf_positive_mask.reshape(-1)
            a_3 = a * 0 + self.astart
            a = a_1 + a_2 + a_3
            # vf positive AND NSteps > NMIN

            # vf positive AND NSteps <= NMIN

        v += dt.reshape((-1, 1, 1)) * f
        dr = dt.reshape((-1, 1, 1)) * v
        # normdr = np.sqrt(np.vdot(dr, dr))
        normdr = np.sqrt(np.diag(dr.reshape(len(dr), -1) @ dr.reshape(len(dr), -1).T))

        update_dr_mask_positive = np.array(normdr > self.maxstep, dtype=int)
        update_dr_mask_negative = np.array(normdr <= self.maxstep, dtype=int)
        dr_1 = dr * update_dr_mask_negative.reshape((-1, 1, 1))
        if 0 in normdr:
            dr_2 = dr * update_dr_mask_positive.reshape((-1, 1, 1)) * self.maxstep
        else:
            dr_2 = dr * update_dr_mask_positive.reshape((-1, 1, 1)) * self.maxstep  / normdr.reshape((-1, 1, 1))
        dr = dr_1 + dr_2

        # if normdr > self.maxstep:
        #     dr = self.maxstep * dr / normdr

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

    def converged(self, forces, fmax):
        """Did the optimization converge?"""
        return np.max((forces ** 2).sum(axis=2), axis=1) < 0.2 ** 2
