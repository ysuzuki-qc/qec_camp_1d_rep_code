import os
import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists("./figure"):
    os.mkdir("./figure/")


def _format(num_detector: int, num_round: int, code_distance: int) -> None:
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xticks(np.arange(-0.5, num_detector, num_round + 1), minor=True)
    plt.yticks(np.arange(-0.5, num_detector, num_round + 1), minor=True)
    ax = plt.gca()
    tick_pos = np.arange(-0.5 + (num_round + 1) / 2, num_detector, num_round + 1)
    tick_label = [f"Q{v}" for v in np.arange(1, code_distance * 2 - 1, 2)]
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_label)
    ax.set_yticklabels(tick_label)
    plt.xlabel("Qubit-Cycle")
    plt.ylabel("Qubit-Cycle")
    plt.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    plt.colorbar()


def plot_correlation(
    code_distance: int, num_round: int, initial_state: int, filename: str
):
    dataset_filename = f"./correlation/correlation_matrix_experiment_d{code_distance}_r{num_round}_i{initial_state}.npy"
    with open(dataset_filename, "rb") as fin:
        matrix_exp = np.load(fin)

    dataset_filename = f"./correlation/correlation_matrix_depolarizing_d{code_distance}_r{num_round}_i{initial_state}.npy"
    with open(dataset_filename, "rb") as fin:
        matrix_dep = np.load(fin)

    dataset_filename = f"./correlation/correlation_matrix_damping_d{code_distance}_r{num_round}_i{initial_state}.npy"
    with open(dataset_filename, "rb") as fin:
        matrix_damp = np.load(fin)

    limit = np.max(np.abs(matrix_exp))
    num_detector = len(matrix_exp)
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 3, 1)
    plt.title("Experiment")
    plt.imshow(matrix_exp, cmap="seismic", vmin=-limit, vmax=limit)
    _format(num_detector, num_round, code_distance)

    plt.subplot(1, 3, 2)
    plt.title("Depolarizing")
    plt.imshow(matrix_dep, cmap="seismic", vmin=-limit, vmax=limit)
    _format(num_detector, num_round, code_distance)

    plt.subplot(1, 3, 3)
    plt.title("Damping")
    plt.imshow(matrix_damp, cmap="seismic", vmin=-limit, vmax=limit)
    _format(num_detector, num_round, code_distance)

    plt.tight_layout()
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")


def run():
    code_distance_list = [3, 5, 7, 9]
    num_round_list = [4, 8]
    initial_state_list = [0, 1]
    for code_distance in code_distance_list:
        for num_round in num_round_list:
            for initial_state in initial_state_list:
                print(code_distance, num_round, initial_state)
                filename = f"./figure/correlation_matrix_damping_d{code_distance}_r{num_round}_i{initial_state}"
                plot_correlation(code_distance, num_round, initial_state, filename)


run()
