import os
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.data import NoiseModel, NoiseType
from lib_1drep.sampler_qulacs import sample_records_with_qulacs
from run_common import ResultSet, Result
from concurrent.futures import ProcessPoolExecutor

if not os.path.exists("./record_sim"):
    os.mkdir("./record_sim/")

code_distance_list = [3,5,7]
num_round_list = [4,8]
initial_state_list = [0, 1]
num_shot = 10000
seed = 42


def run(
    code_distance: int,
    num_round: int,
    initial_state: int,
    num_shot: int,
    seed: int,
) -> Result:
    record_filename = f"./record_sim/record_thermalization_d{code_distance}_r{num_round}_i{initial_state}.json"
    if os.path.exists(record_filename):
        return

    # load noise model
    filename = f"./noise_model/noise_model_device_d{code_distance}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    noise_model = NoiseModel.model_validate_json(json_str)

    sample_noise_model = NoiseModel()
    for noise_property in noise_model.noise_property_list:
        sample_noise_property = noise_property.model_copy()
        sample_noise_property.noise_type = NoiseType.THERMALIZATION
        sample_noise_model.noise_property_list.append(sample_noise_property)

    # sample record
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    dataset = sample_records_with_qulacs(circuit, num_shot, sample_noise_model)

    # dump
    json_str = dataset.model_dump_json(indent=4)
    with open(record_filename, "w") as fout:
        fout.write(json_str)
    print(record_filename)


# main loop
result_set = ResultSet(
    method_noise_model="characterization", method_sampling="simulation"
)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        tasks = []
        for code_distance in code_distance_list:
            for initial_state in initial_state_list:
                for num_round in num_round_list:
                    task = executor.submit(run,
                        code_distance,
                        num_round,
                        initial_state,
                        num_shot=num_shot,
                        seed=seed,
                    )
                    tasks.append(task)
        for task in tasks:
            print(task.result())

