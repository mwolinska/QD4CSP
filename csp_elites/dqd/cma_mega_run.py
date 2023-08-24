from ase.ga.ofp_comparator import OFPComparator

from csp_elites.crystal.crystal_evaluator import CrystalEvaluator
from csp_elites.crystal.crystal_system import CrystalSystem
from csp_elites.dqd.cma_mega_loop import CMAMEGALOOP
from csp_elites.map_elites.elites_utils import make_current_time_string
from csp_elites.utils.experiment_parameters import ExperimentParameters



def main_cma(experiment_parameters: ExperimentParameters):
    current_time_label = make_current_time_string(with_time=True)
    experiment_label = \
        f"{current_time_label}_{experiment_parameters.system_name}_{experiment_parameters.experiment_tag}"
    alternative_operators = experiment_parameters.cvt_run_parameters[
        "alternative_operators"] if "alternative_operators" in experiment_parameters.cvt_run_parameters.keys() else None

    print(experiment_label)
    comparator = OFPComparator(n_top=len(experiment_parameters.blocks), dE=1.0,
                               cos_dist_max=1e-3, rcut=10., binwidth=0.05,
                               pbc=[True, True, True], sigma=0.05, nsigma=4,
                               recalculate=False)

    crystal_system = CrystalSystem(
        atom_numbers_to_optimise=experiment_parameters.blocks,
        volume=experiment_parameters.volume,
        ratio_of_covalent_radii=experiment_parameters.ratio_of_covalent_radii,
        splits=experiment_parameters.splits,
        operator_probabilities=experiment_parameters.operator_probabilities,
        start_generator=experiment_parameters.start_generator,
        alternative_operators=alternative_operators,
        learning_rate=None,
    )

    force_threshold = experiment_parameters.cvt_run_parameters[
        "force_threshold"] if "force_threshold" in experiment_parameters.cvt_run_parameters.keys() else False
    constrained_qd = experiment_parameters.cvt_run_parameters[
        "constrained_qd"] if "constrained_qd" in experiment_parameters.cvt_run_parameters.keys() else False
    fmax_threshold = experiment_parameters.cvt_run_parameters[
        "fmax_threshold"] if "fmax_threshold" in experiment_parameters.cvt_run_parameters.keys() else 0.2

    force_threshold_exp_fmax = experiment_parameters.cvt_run_parameters[
        "force_threshold_exp_fmax"] if "force_threshold_exp_fmax" in experiment_parameters.cvt_run_parameters.keys() else 1.0

    crystal_evaluator = CrystalEvaluator(
        comparator=comparator,
        with_force_threshold=force_threshold,
        constrained_qd=constrained_qd,
        fmax_relaxation_convergence=fmax_threshold,
        force_threshold_fmax=force_threshold_exp_fmax,
        compute_gradients=True,
    )

    cma = CMAMEGALOOP(
        number_of_bd_dimensions=2,
        crystal_system=crystal_system,
        crystal_evaluator=crystal_evaluator,
        step_size_gradient_optimizer_niu=experiment_parameters.cvt_run_parameters["cma_learning_rate"],
        initial_cmaes_step_size_sigma_g=experiment_parameters.cvt_run_parameters["cma_sigma_0"]
    )

    cma.compute(experiment_parameters.number_of_niches,
                experiment_parameters.maximum_evaluations,
                experiment_parameters.cvt_run_parameters,
                experiment_label=experiment_label
                )
