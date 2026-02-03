import os
import numpy as np
import matplotlib.pyplot as plt
from run_common import ResultSet

if not os.path.exists("./figure"):
    os.mkdir("./figure/")


def plot(resultset: dict[str, str], figname: str) -> None:
    plt.figure(figsize=(20, 10))
    cmap = plt.get_cmap("tab10")
    color = {3: cmap(0), 5: cmap(1), 7: cmap(2), 9: cmap(3)}
    linestyle = {0: "-", 1: "--"}
    for result_index, name in enumerate(resultset):
        plt.subplot(1, len(resultset), result_index + 1)
        filename = resultset[name]
        with open(filename, "r") as fin:
            json_str = fin.read()
        result_set = ResultSet.model_validate_json(json_str)

        plot_data: dict[int, dict[str, list]] = {}
        for result in result_set.result_list:
            key = (result.initial_state, result.code_distance)
            if key not in plot_data:
                plot_data[key] = {
                    "num_round": [],
                    "logical_error_rate": [],
                }
            plot_data[key]["num_round"].append(result.num_round)
            plot_data[key]["logical_error_rate"].append(
                result.failure_count / result.num_shot
            )

        key_list = sorted(list(plot_data.keys()))
        for key in key_list:
            initial_state, code_distance = key
            xy_data = list(
                zip(
                    plot_data[key]["num_round"],
                    plot_data[key]["logical_error_rate"],
                )
            )
            xy_data.sort()
            x = np.array(xy_data)[:, 0]
            y = np.array(xy_data)[:, 1]

            plt.plot(
                x,
                y,
                label=f"d={code_distance} init={initial_state}",
                color=color[code_distance],
                linestyle=linestyle[initial_state],
                marker="o",
            )

        plt.title(name)
        plt.legend(loc="upper left")
        plt.xlabel("num round")
        plt.ylabel("logical error rate")
        plt.xlim(0, 10)
        plt.ylim(0, 0.5)
        plt.grid(which="major", color="black", linestyle="-", alpha=0.2)
        plt.grid(which="minor", color="black", linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figname + ".png")
    plt.savefig(figname + ".pdf")


resultset = {
    "device noise model / experimental record": "./result/result_qec_device_experiment.json",
    "device noise model / simulation record": "./result/result_qec_device_simulation.json",
    "uniform noise model / experimental record": "./result/result_qec_uniform_experiment.json",
    "uniform noise model / simulation record": "./result/result_qec_uniform_simulation.json",
}
figname = "./figure/logical_error_rate_vs_num_round"
plot(resultset, figname)
