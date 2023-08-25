import json
import sys

print("cellbounds")
from ase.ga.utilities import CellBounds

print("data model")
from csp_elites.crystal.materials_data_model import MaterialProperties, StartGenerators
print("main cma")
from csp_elites.dqd.cma_mega_run import main_cma
from csp_elites.utils.experiment_parameters import ExperimentParameters

if __name__ == '__main__':
    file_location = "configs/0822/cma_5_relaxation_lr1_sigma_1.json"
    print(file_location)
    if file_location == "":
        file_location = sys.argv[1]
    with open(file_location, "r") as file:
        experiment_parameters = json.load(file)

    experiment_parameters = ExperimentParameters(**experiment_parameters)
    experiment_parameters.cellbounds = CellBounds(
            bounds={'phi': [20, 160], 'chi': [20, 160], 'psi': [20, 160], 'a': [2, 40], 'b': [2, 40],
                    'c': [2, 40]}),
    experiment_parameters.splits = {(2,): 1, (4,): 1}
    experiment_parameters.cvt_run_parameters["behavioural_descriptors"] = \
        [MaterialProperties(value) for value in experiment_parameters.cvt_run_parameters["behavioural_descriptors"]]

    experiment_parameters.start_generator = StartGenerators(experiment_parameters.start_generator)
    main_cma(experiment_parameters)
