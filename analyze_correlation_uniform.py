import os
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.analysis import get_two_point_correlation
from lib_1drep.qec import sample_records
from lib_1drep.noise import create_1drep_uniform_noise_model

if not os.path.exists("./correlation"):
    os.mkdir("./correlation/")


def analyze_simulation_data(
    code_distance: int,
    num_round: int,
    initial_state: int,
    num_shot: int,
    error_rate: float,
    seed: int,
) -> np.ndarray:
    # create noise model
    noise_model = create_1drep_uniform_noise_model(error_rate, code_distance)

    # create record data
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    dataset = sample_records(circuit, num_shot, noise_model, seed)

    # analyze
    matrix = get_two_point_correlation(circuit, dataset, num_shot)
    return matrix


code_distance_list = [3, 5, 7, 9]
num_round_list = [2, 3, 4, 5, 6, 7, 8]
initial_state_list = [0, 1]
num_shot = 1000000
error_rate = 5e-2
seed = 42

for code_distance in code_distance_list:
    for num_round in num_round_list:
        for initial_state in initial_state_list:
            print(code_distance, num_round, initial_state)
            filename = f"./correlation/correlation_matrix_uniform_d{code_distance}_r{num_round}_i{initial_state}.npy"
            matrix = analyze_simulation_data(
                code_distance, num_round, initial_state, num_shot, error_rate, seed
            )
            with open(filename, "wb") as fout:
                np.save(fout, matrix)
