import os
import pathlib

from tqdm import tqdm

from retrieve_results.report_plot_generator import ReportPlotGenerator

if __name__ == "__main__":
    path_to_all_experiments = (
        pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync/report_data"
    )
    # all_experiments = [name for name in os.listdir(f"{path_to_all_experiments }")
    #                      if os.path.isdir(path_to_all_experiments / name) and (name != "all_plots")]

    all_experiments = ["7_benchmark_other_materials"]
    fitness_values = [[8.7, 9.5]]  # 6.5

    for i, experiment in tqdm(enumerate(all_experiments)):
        report_generator = ReportPlotGenerator(
            path_to_experiments=path_to_all_experiments / experiment,
            plot_labels=None,
            title_tag=None,
        )
        report_generator.plot_cvt_and_symmetry(
            force_replot=False,
            all_sub_experiments=True,
            plot_cvt=True,
            experiment_list=["dqd_other_materials"],
        )
