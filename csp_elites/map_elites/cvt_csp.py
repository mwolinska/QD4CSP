# | This file is based on the implementation map-elites implementation pymap_elites repo by resibots team https://github.com/resibots/pymap_elites
# | Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
# | Eloise Dalin , eloise.dalin@inria.fr
# | Pierre Desreumaux , pierre.desreumaux@inria.fr
# | **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
# | mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.

import gc
import os
import pickle

import numpy as np
import psutil
from ase import Atoms
from chgnet.graph import CrystalGraphConverter
from matplotlib import pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.neighbors import KDTree
from tqdm import tqdm

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.map_elites.elites_utils import (
    cvt,
    save_archive,
    add_to_archive,
    write_centroids,
    make_experiment_folder,
    Species,
)
from csp_elites.utils.get_mpi_structures import get_all_materials_with_formula
from csp_elites.utils.plot import load_archive_from_pickle


class CVT:
    def __init__(
        self,
        number_of_bd_dimensions: int,
        crystal_system: CrystalSystem,
        crystal_evaluator: CrystalEvaluator,
    ):
        self.number_of_bd_dimensions = number_of_bd_dimensions
        self.crystal_system = crystal_system
        self.crystal_evaluator = crystal_evaluator
        self.graph_converter = CrystalGraphConverter()

    def initialise_run_parameters(
        self, number_of_niches, maximum_evaluations, run_parameters, experiment_label
    ):
        self.experiment_directory_path = make_experiment_folder(experiment_label)
        self.log_file = open(
            f"{self.experiment_directory_path}/{experiment_label}.dat", "w"
        )
        self.memory_log = open(f"{self.experiment_directory_path}/memory_log.dat", "w")
        with open(
            f"{self.experiment_directory_path}/experiment_parameters.pkl", "wb"
        ) as file:
            pickle.dump(run_parameters, file)

        self.archive = {}  # init archive (empty)
        self.n_evals = 0  # number of evaluations since the beginning
        self.b_evals = 0  # number evaluation since the last dump
        self.n_relaxation_steps = run_parameters["number_of_relaxation_steps"]
        self.configuration_counter = 0
        # create the CVT
        self.kdt = self._initialise_kdt_and_centroids(
            experiment_directory_path=self.experiment_directory_path,
            number_of_niches=number_of_niches,
            run_parameters=run_parameters,
        )
        self.number_of_niches = number_of_niches
        self.run_parameters = run_parameters
        self.maximum_evaluations = maximum_evaluations

        self.relax_every_n_generations = (
            run_parameters["relax_every_n_generations"]
            if "relax_every_n_generations" in run_parameters.keys()
            else 0
        )
        self.relax_archive_every_n_generations = (
            run_parameters["relax_archive_every_n_generations"]
            if "relax_archive_every_n_generations" in run_parameters.keys()
            else 0
        )
        self.generation_counter = 0
        self.run_parameters = run_parameters

    def batch_compute_with_list_of_atoms(
        self,
        number_of_niches,
        maximum_evaluations,
        run_parameters,
        experiment_label,
    ):
        # create the CVT
        self.initialise_run_parameters(
            number_of_niches, maximum_evaluations, run_parameters, experiment_label
        )

        ram_logging = []
        pbar = tqdm(desc="Number of evaluations", total=maximum_evaluations, position=2)
        while self.n_evals < maximum_evaluations:  ### NUMBER OF GENERATIONS
            ram_logging.append(psutil.virtual_memory()[3] / 1000000000)
            self.generation_counter += 1
            # random initialization
            population = []
            if len(self.archive) <= run_parameters["random_init"] * number_of_niches:
                individuals = self.crystal_system.create_n_individuals(
                    run_parameters["random_init_batch"]
                )
                if run_parameters["seed"]:
                    individuals = self.initialise_known_atoms()
                population += individuals

                with open(
                    f"{self.experiment_directory_path}/starting_population.pkl", "wb"
                ) as file:
                    pickle.dump(population, file)

            elif (
                (self.relax_archive_every_n_generations != 0)
                and (
                    self.generation_counter % self.relax_archive_every_n_generations
                    == 0
                )
                and (self.generation_counter != 0)
            ):
                population = [species.x for species in list(self.archive.values())]

            else:  # variation/selection loop
                mutated_individuals = self.mutate_individuals(
                    run_parameters["batch_size"]
                )
                population += mutated_individuals

            n_relaxation_steps = self.set_number_of_relaxation_steps()

            (
                population_as_atoms,
                population,
                fitness_scores,
                descriptors,
                kill_list,
                gradients,
            ) = self.crystal_evaluator.batch_compute_fitness_and_bd(
                list_of_atoms=population, n_relaxation_steps=n_relaxation_steps
            )

            if population is not None:
                self.crystal_system.update_operator_scaling_volumes(
                    population=population_as_atoms
                )
                del population_as_atoms

            self.update_archive(
                population, fitness_scores, descriptors, kill_list, gradients
            )
            pbar.update(len(population))
            del population
            del fitness_scores
            del descriptors
            del kill_list

        save_archive(self.archive, self.n_evals, self.experiment_directory_path)
        self.plot_memory(ram_logging)
        return self.experiment_directory_path, self.archive

    def initialise_known_atoms(self):
        _, known_atoms = get_all_materials_with_formula(
            self.crystal_system.compound_formula
        )
        individuals = []
        for atoms in known_atoms:
            if (
                len(atoms.get_atomic_numbers())
                == self.run_parameters["filter_starting_Structures"]
            ):
                atoms.rattle()
                atoms.info = None
                atoms = atoms.todict()
                individuals.append(atoms)
        del known_atoms
        return individuals

    def plot_memory(self, ram_logging):
        plt.plot(range(len(ram_logging)), ram_logging)
        plt.xlabel("Number of Times Evaluation Loop Was Ran")
        plt.ylabel("Amount of RAM Used")
        plt.title("RAM over time")
        plt.savefig(
            f"{self.experiment_directory_path}/memory_over_time.png", format="png"
        )

    def update_archive(
        self, population, fitness_scores, descriptors, kill_list, gradients
    ):
        s_list = self.crystal_evaluator.batch_create_species(
            population, fitness_scores, descriptors, kill_list, gradients
        )
        evaluations_performed = len(population)
        self.n_evals += evaluations_performed
        self.b_evals += evaluations_performed
        for s in s_list:
            if s is None:
                continue
            else:
                s.x["info"]["confid"] = self.configuration_counter
                self.configuration_counter += 1
                add_to_archive(s, s.desc, self.archive, self.kdt)
        if (
            self.b_evals >= self.run_parameters["dump_period"]
            and self.run_parameters["dump_period"] != -1
        ):
            print(
                "[{}/{}]".format(self.n_evals, int(self.maximum_evaluations)),
                end=" ",
                flush=True,
            )
            save_archive(self.archive, self.n_evals, self.experiment_directory_path)
            self.b_evals = 0
        # write log
        if self.log_file != None:
            fit_list = np.array([x.fitness for x in self.archive.values()])
            qd_score = np.sum(fit_list)
            coverage = 100 * len(fit_list) / self.number_of_niches

            self.log_file.write(
                "{} {} {} {} {} {} {} {} {}\n".format(
                    self.n_evals,
                    len(self.archive.keys()),
                    np.max(fit_list),
                    np.mean(fit_list),
                    np.median(fit_list),
                    np.percentile(fit_list, 5),
                    np.percentile(fit_list, 95),
                    coverage,
                    qd_score,
                )
            )
            self.log_file.flush()
        memory = psutil.virtual_memory()[3] / 1000000000
        self.memory_log.write("{} {}\n".format(self.n_evals, memory))
        self.memory_log.flush()
        gc.collect()

    def _initialise_kdt_and_centroids(
        self, experiment_directory_path, number_of_niches, run_parameters
    ):
        # create the CVT
        if run_parameters["normalise_bd"]:
            bd_minimum_values, bd_maximum_values = [0, 0], [1, 1]
        else:
            bd_minimum_values, bd_maximum_values = (
                run_parameters["bd_minimum_values"],
                run_parameters["bd_maximum_values"],
            )

        c = cvt(
            number_of_niches,
            self.number_of_bd_dimensions,
            run_parameters["cvt_samples"],
            bd_minimum_values,
            bd_maximum_values,
            experiment_directory_path,
            run_parameters["behavioural_descriptors"],
            run_parameters["cvt_use_cache"],
            formula=self.crystal_system.compound_formula,
        )
        kdt = KDTree(c, leaf_size=30, metric="euclidean")
        write_centroids(
            c,
            experiment_folder=experiment_directory_path,
            bd_names=run_parameters["behavioural_descriptors"],
            bd_minimum_values=run_parameters["bd_minimum_values"],
            bd_maximum_values=run_parameters["bd_maximum_values"],
            formula=self.crystal_system.compound_formula,
        )
        del c
        return kdt

    def mutate_individuals(self, batch_size):
        keys = list(self.archive.keys())
        # we select all the parents at the same time because randint is slow
        rand1 = np.random.randint(len(keys), size=batch_size)
        rand2 = np.random.randint(len(keys), size=batch_size)
        mutated_offsprings = []
        for n in range(0, batch_size):
            # parent selection
            x = self.archive[keys[rand1[n]]]
            y = self.archive[keys[rand2[n]]]
            # copy & add variation
            z = self.crystal_system.mutate([x, y])
            if z is None or (
                self.graph_converter(
                    AseAtomsAdaptor.get_structure(z), on_isolated_atoms="warn"
                )
                is None
            ):
                continue
            mutated_offsprings += [Atoms.todict(z)]
        return mutated_offsprings

    def set_number_of_relaxation_steps(self):
        if self.relax_every_n_generations != 0 and (
            self.relax_archive_every_n_generations == 0
        ):
            if self.generation_counter // self.relax_every_n_generations == 0:
                n_relaxation_steps = 100
            else:
                n_relaxation_steps = self.run_parameters["number_of_relaxation_steps"]
        elif (self.relax_archive_every_n_generations != 0) and (
            self.generation_counter % self.relax_archive_every_n_generations == 0
        ):
            n_relaxation_steps = (
                self.run_parameters[
                    "relax_archive_every_n_generations_n_relaxation_steps"
                ]
                if "relax_archive_every_n_generations_n_relaxation_steps"
                in self.run_parameters.keys()
                else 10
            )

        else:
            n_relaxation_steps = self.run_parameters["number_of_relaxation_steps"]

        return n_relaxation_steps

    def start_experiment_from_archive(
        self,
        experiment_to_load_directory_path: str,
        experiment_label: str,
        run_parameters,
        number_of_niches,
        maximum_evaluations,
    ):
        self.initialise_run_parameters(
            number_of_niches, maximum_evaluations, run_parameters, experiment_label
        )
        self.log_file = open(
            f"{experiment_to_load_directory_path}/{experiment_label}_continued.dat", "w"
        )
        last_archive = max(
            [
                int(name.lstrip("archive_").rstrip(".pkl"))
                for name in os.listdir(experiment_to_load_directory_path)
                if (
                    (not os.path.isdir(name))
                    and ("archive_" in name)
                    and (".pkl" in name)
                )
            ]
        )
        self.archive = self._convert_saved_archive_to_experiment_archive(
            experiment_directory_path=experiment_to_load_directory_path,
            archive_number=last_archive,
            experiment_label=experiment_label,
            kdt=self.kdt,
            archive=self.archive,
        )
        self.experiment_directory_path = experiment_to_load_directory_path
        self.n_evals = 6
        ram_logging = []
        pbar = tqdm(
            desc="Number of evaluations", total=self.maximum_evaluations, position=2
        )
        while self.n_evals < self.maximum_evaluations:  ### NUMBER OF GENERATIONS
            ram_logging.append(psutil.virtual_memory()[3] / 1000000000)
            self.generation_counter += 1
            # random initialization
            population = []
            if len(self.archive) <= run_parameters["random_init"] * number_of_niches:
                individuals = self.crystal_system.create_n_individuals(
                    run_parameters["random_init_batch"]
                )
                if run_parameters["seed"]:
                    individuals = self.initialise_known_atoms()
                population += individuals

                with open(
                    f"{self.experiment_directory_path}/starting_population.pkl", "wb"
                ) as file:
                    pickle.dump(population, file)

            elif (
                (self.relax_archive_every_n_generations != 0)
                and (
                    self.generation_counter % self.relax_archive_every_n_generations
                    == 0
                )
                and (self.generation_counter != 0)
            ):
                population = [species.x for species in list(self.archive.values())]

            else:  # variation/selection loop
                mutated_individuals = self.mutate_individuals(
                    run_parameters["batch_size"]
                )
                population += mutated_individuals

            n_relaxation_steps = self.set_number_of_relaxation_steps()

            (
                population_as_atoms,
                population,
                fitness_scores,
                descriptors,
                kill_list,
                gradients,
            ) = self.crystal_evaluator.batch_compute_fitness_and_bd(
                list_of_atoms=population, n_relaxation_steps=n_relaxation_steps
            )

            if population is not None:
                self.crystal_system.update_operator_scaling_volumes(
                    population=population_as_atoms
                )
                del population_as_atoms

            self.update_archive(
                population, fitness_scores, descriptors, kill_list, gradients
            )
            pbar.update(len(population))
            del population
            del fitness_scores
            del descriptors
            del kill_list

        save_archive(self.archive, self.n_evals, self.experiment_directory_path)
        self.plot_memory(ram_logging)
        return self.experiment_directory_path, self.archive

    def _convert_saved_archive_to_experiment_archive(
        self,
        experiment_directory_path,
        experiment_label,
        archive_number,
        kdt,
        archive,
        individual_type="atoms",
    ):
        fitnesses, centroids, descriptors, individuals = load_archive_from_pickle(
            filename=f"{experiment_directory_path}/archive_{archive_number}.pkl"
        )

        if isinstance(individuals[0], Atoms):
            species_list = [
                Species(
                    x=individuals[i].todict(),
                    desc=descriptors[i],
                    fitness=fitnesses[i],
                    centroid=None,
                )
                for i in range(len(individuals))
            ]
        elif isinstance(individuals[0], dict):
            species_list = [
                Species(
                    x=individuals[i],
                    desc=descriptors[i],
                    fitness=fitnesses[i],
                    centroid=None,
                )
                for i in range(len(individuals))
            ]
        for i in range(len(species_list)):
            add_to_archive(species_list[i], descriptors[i], archive=archive, kdt=kdt)

        return archive
