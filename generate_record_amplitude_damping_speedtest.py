import os
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.data import NoiseModel, NoiseType
from lib_1drep.sampler_qulacs import sample_records_with_qulacs
from lib_1drep.noise import create_1drep_uniform_noise_model
from run_common import ResultSet, Result
from concurrent.futures import ProcessPoolExecutor

if not os.path.exists("./record_sim"):
    os.mkdir("./record_sim/")

code_distance_list = [3,5,7,9]
num_round_list = [8]
initial_state_list = [0]
num_shot = 100
seed = 42
num_worker=1

def run(
    code_distance: int,
    num_round: int,
    initial_state: int,
    num_shot: int,
    seed: int,
) -> Result:

    # load noise model
    noise_model = create_1drep_uniform_noise_model(1e-2, code_distance)

    sample_noise_model = NoiseModel()
    for noise_property in noise_model.noise_property_list:
        sample_noise_property = noise_property.model_copy()
        sample_noise_property.noise_type = NoiseType.AMPLITUDE_DAMPING
        sample_noise_model.noise_property_list.append(sample_noise_property)

    # sample record
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    import time
    start = time.time()
    dataset = sample_records_with_qulacs(circuit, num_shot, sample_noise_model)
    elapsed = time.time()-start
    print(code_distance, elapsed, num_shot, elapsed/num_shot)


# main loop
result_set = ResultSet(
    method_noise_model="characterization", method_sampling="simulation"
)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
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

