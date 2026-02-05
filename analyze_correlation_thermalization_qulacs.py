import os
import numpy as np
from lib_1drep.data import RecordDataset
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.analysis import get_two_point_correlation

if not os.path.exists("./correlation"):
    os.mkdir("./correlation/")


def analyze_damping_data(
    code_distance: int, num_round: int, initial_state: int, num_shot: int
) -> np.ndarray:
    # load record
    filename = f"./record_sim/record_thermalization_d{code_distance}_r{num_round}_i{initial_state}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    dataset = RecordDataset.model_validate_json(json_str)

    # evaluate
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    matrix = get_two_point_correlation(circuit, dataset, num_shot)
    return matrix


code_distance_list = [3, 5, 7]
num_round_list = [4, 8]
initial_state_list = [0, 1]
num_shot = 10000

for code_distance in code_distance_list:
    for num_round in num_round_list:
        for initial_state in initial_state_list:
            print(code_distance, num_round, initial_state)
            filename = f"./correlation/correlation_matrix_thermalization_d{code_distance}_r{num_round}_i{initial_state}.npy"
            matrix = analyze_damping_data(
                code_distance, num_round, initial_state, num_shot
            )
            with open(filename, "wb") as fout:
                np.save(fout, matrix)
