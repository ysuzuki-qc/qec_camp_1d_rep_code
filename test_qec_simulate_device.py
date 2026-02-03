import matplotlib.pyplot as plt
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import sample_records, evaluation
from lib_1drep.data import NoiseModel


num_round_list = list(np.arange(2, 10, 1)) + list(np.arange(20, 31, 10))
seed = 42
num_shot = 10000
initial_state = 1
code_distance_list = [3, 5, 7, 9]

for code_distance in code_distance_list:

    # load noise model
    filename = f"./noise_model/noise_model_device_d{code_distance}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    noise_model = NoiseModel.model_validate_json(json_str)

    logical_error_rate_list = []
    for num_round in num_round_list:
        circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
        record = sample_records(circuit, num_shot, noise_model, seed)
        result = evaluation(circuit, record, noise_model)
        logical_error_rate = np.sum(result) / num_shot
        logical_error_rate_list.append(logical_error_rate)
        print(code_distance, num_round, logical_error_rate)

    plt.plot(num_round_list, logical_error_rate_list, "*-", label=f"d={code_distance}")
plt.legend(loc="upper left")
plt.xlabel("num round")
plt.ylabel("logical error rate")
plt.xlim(0, max(num_round_list))
plt.ylim(0, 0.5)
plt.grid(which="major", color="black", linestyle="-", alpha=0.2)
plt.grid(which="minor", color="black", linestyle="-", alpha=0.2)
plt.title(f"num_round={num_round}")
plt.show()
