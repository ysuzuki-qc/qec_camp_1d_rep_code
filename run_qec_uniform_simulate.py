import os
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import evaluation, sample_records
from run_common import ResultSet, Result
from lib_1drep.noise import create_1drep_uniform_noise_model


if not os.path.exists("./result"):
    os.mkdir("./result/")

filename = "./result/result_qec_uniform_simulation.json"
code_distance_list = [3, 5, 7, 9]
num_round_list = [2, 3, 4, 5, 6, 7, 8, 10, 20, 30]
initial_state_list = [0, 1]
num_shot = 10000
seed = 42
error_rate = 5e-2


def run(
    code_distance: int,
    num_round: int,
    initial_state: int,
    num_shot: int,
    seed: int,
    error_rate: float,
) -> Result:
    # create noise model
    noise_model = create_1drep_uniform_noise_model(error_rate, code_distance)

    # sample record
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    dataset = sample_records(circuit, num_shot, noise_model, seed)

    # evaluate
    result = evaluation(circuit, dataset, noise_model, limit_shot=num_shot)
    failure_count = int(np.sum(result))

    result = Result(
        code_distance=code_distance,
        num_round=num_round,
        initial_state=initial_state,
        num_shot=num_shot,
        failure_count=failure_count,
        seed=seed,
        expected_uniform_error_rate=error_rate,
    )
    return result


# main loop
result_set = ResultSet(method_noise_model="uniform", method_sampling="simulation")
for code_distance in code_distance_list:
    for initial_state in initial_state_list:
        for num_round in num_round_list:
            result = run(
                code_distance,
                num_round,
                initial_state,
                num_shot=num_shot,
                seed=seed,
                error_rate=error_rate,
            )
            result_set.result_list.append(result)
            print(result)

# dump
json_str = result_set.model_dump_json(indent=4)
with open(filename, "w") as fout:
    fout.write(json_str)
