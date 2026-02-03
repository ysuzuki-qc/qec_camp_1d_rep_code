import matplotlib.pyplot as plt
import numpy as np
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import sample_records, evaluation
from lib_1drep.noise import create_1drep_uniform_noise_model

num_round = 8
seed = 42
num_shot = 10000
initial_state = 1
code_distance_list = [3, 5, 7, 9]
error_rate_list = np.logspace(np.log10(1e-3), np.log10(0.2), 20)

for code_distance in code_distance_list:
    num_round = code_distance*3
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    logical_error_rate_list = []
    for error_rate in error_rate_list:
        noise_model = create_1drep_uniform_noise_model(error_rate, code_distance)
        record = sample_records(circuit, num_shot, noise_model, seed)
        result = evaluation(circuit, record, noise_model)
        logical_error_rate = np.sum(result) / num_shot
        logical_error_rate_list.append(logical_error_rate)
        print(code_distance, error_rate, logical_error_rate)

    plt.plot(error_rate_list, logical_error_rate_list, "*-", label=f"d={code_distance}")
plt.legend(loc="upper left")
plt.xlabel("physical error rate")
plt.ylabel("logical error rate")
plt.xscale("log")
plt.yscale("log")
plt.grid(which="major", color="black", linestyle="-", alpha=0.2)
plt.grid(which="minor", color="black", linestyle="-", alpha=0.2)
plt.title(f"num_round={num_round}")
plt.show()
