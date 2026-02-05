import os
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import evaluation
from lib_1drep.data import NoiseModel, RecordDataset
from run_common import ResultSet, Result

code_distance_list = [3, 5, 7]
num_round_list = [4, 8]
initial_state_list = [0, 1]
limit_shot = 10000


def run(
    code_distance: int, num_round: int, initial_state: int, noise_name: str, limit_shot: int = -1
) -> Result:
    # load noise model
    filename = f"./noise_model/noise_model_device_d{code_distance}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    noise_model = NoiseModel.model_validate_json(json_str)

    # load record
    folder = "record_sim" if not noise_name.startswith("dataset") else "record"
    filename = f"./{folder}/record_{noise_name}_d{code_distance}_r{num_round}_i{initial_state}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    dataset = RecordDataset.model_validate_json(json_str)

    # limit record
    if limit_shot <= 0:
        num_shot = len(dataset.record_dataset)
    else:
        num_shot = min(limit_shot, len(dataset.record_dataset))

    # evaluate
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    result = evaluation(circuit, dataset, noise_model, limit_shot=num_shot)
    failure_count = int(np.sum(result))

    result = Result(
        code_distance=code_distance,
        num_round=num_round,
        initial_state=initial_state,
        num_shot=num_shot,
        failure_count=failure_count,
    )
    return result


# main loop
result_set = ResultSet(
    method_noise_model="characterization", method_sampling="experiment"
)
for noise_name in ["amplitude_damping", "coherent_XX", "depolarizing", "dataset_device"]:
    for code_distance in code_distance_list:
        for initial_state in initial_state_list:
            for num_round in num_round_list:
                result = run(code_distance, num_round, initial_state, noise_name, limit_shot=limit_shot)
                print(noise_name, code_distance, initial_state, num_round, result.failure_count/result.num_shot)

