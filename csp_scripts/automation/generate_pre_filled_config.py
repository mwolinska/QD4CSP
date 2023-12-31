import pathlib
import sys

from csp_elites.utils.experiment_parameters import ExperimentParameters

if __name__ == "__main__":
    # Update with your desired filename / location
    file_location = ""
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) == 3:
            config_subfolder = sys.argv[2]
        else:
            config_subfolder = ""
    else:
        filename = "demo_config"
        config_subfolder = "demo_folder"

    path_to_save_config = (
        pathlib.Path(__file__).parent.parent / "experiment_configs" / config_subfolder
    )
    path_to_save_config.mkdir(exist_ok=True)

    experiment_parameters = ExperimentParameters.generate_default_to_populate()
    experiment_parameters.save_as_json(
        experiment_directory_path=path_to_save_config, filename=filename
    )
