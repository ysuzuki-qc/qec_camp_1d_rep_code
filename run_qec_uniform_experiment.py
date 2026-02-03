import os
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import evaluation
from lib_1drep.data import RecordDataset
from lib_1drep.noise import create_1drep_uniform_noise_model
from run_common import ResultSet, Result

if not os.path.exists("./result"):
    os.mkdir("./result/")

filename = "./result/result_qec_uniform_experiment.json"
code_distance_list = [3, 5, 7, 9]
num_round_list = [2, 3, 4, 5, 6, 7, 8]
initial_state_list = [0, 1]
limit_shot = 10000
expected_uniform_error_rate = 1e-3


def run(
    code_distance: int,
    num_round: int,
    initial_state: int,
    expected_uniform_error_rate: float,
    limit_shot: int = -1,
) -> Result:
    # load noise model
    noise_model = create_1drep_uniform_noise_model(
        expected_uniform_error_rate, code_distance
    )

    # load record
    filename = f"./record/record_dataset_device_d{code_distance}_r{num_round}_i{initial_state}.json"
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
        expected_uniform_error_rate=expected_uniform_error_rate,
    )
    return result


# main loop
result_set = ResultSet(method_noise_model="uniform", method_sampling="experiment")
for code_distance in code_distance_list:
    for initial_state in initial_state_list:
        for num_round in num_round_list:
            result = run(
                code_distance,
                num_round,
                initial_state,
                limit_shot=limit_shot,
                expected_uniform_error_rate=expected_uniform_error_rate,
            )
            result_set.result_list.append(result)
            print(result)

# dump
json_str = result_set.model_dump_json(indent=4)
with open(filename, "w") as fout:
    fout.write(json_str)
