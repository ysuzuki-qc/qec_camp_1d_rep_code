import os
import sys
import numpy as np
import copy
import scipy.optimize as optimize
import json
from lib_1drep.data import RecordDataset
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.analysis import get_two_point_correlation
from lib_1drep.data import NoiseModel
from lib_1drep.qec import sample_records

if not os.path.exists("./correlation"):
    os.mkdir("./correlation/")

result = {'fevs':[]}

def analyze_experimental_data(
    code_distance: int, num_round: int, initial_state: int, num_shot: int
) -> np.ndarray:
    # load record
    filename = f"./record/record_dataset_device_d{code_distance}_r{num_round}_i{initial_state}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    dataset = RecordDataset.model_validate_json(json_str)

    # evaluate
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    matrix = get_two_point_correlation(circuit, dataset, num_shot)
    return matrix

def analyze_simulation_data(
    code_distance: int,
    num_round: int,
    initial_state: int,
    num_shot: int,
    seed: int = 42,
) -> np.ndarray:
    # load noise model
    filename = f"./noise_model/noise_model_device_d{code_distance}.json"
    with open(filename, "r") as fin:
        json_str = fin.read()
    noise_model = NoiseModel.model_validate_json(json_str)

    # create record data
    circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
    dataset = sample_records(circuit, num_shot, noise_model, seed)

    # analyze
    matrix = get_two_point_correlation(circuit, dataset, num_shot)
    return matrix

def update_nm(n, x):
    for i,p in enumerate(n.noise_property_list):
        p.error_rate = x[i]

# target function for scipy.optimize.minimize
def eval_rmse(x, d, r, i, s, _n):

    seed = 42

    n = copy.deepcopy(_n)
    if not x[0] == -1:
        update_nm(n, x)

    circuit = create_1d_repetition_circuit(d, r, i)
    dataset = sample_records(circuit, s, n, seed)

    filename = f"./correlation/correlation_matrix_experiment_d{d}_r{r}_i{i}.npy"
    if os.path.exists(filename):
        with open(filename, "rb") as fin:
            matrix_e = np.load(fin)
    else:
        matrix_e = analyze_experimental_data(
            code_distance, num_round, initial_state, num_shot
        )
        with open(filename, "wb") as fout:
            np.save(fout, matrix_e)

    # try multiple times when the number of samples is not sufficient
    for num_shot in [s, s*10]:
        matrix_s = get_two_point_correlation(circuit, dataset, num_shot)
        matrix_d = matrix_e - matrix_s
        matrix_d = matrix_d ** 2
        rmse = np.sqrt(np.mean(matrix_d))
        if not np.isnan(rmse):
            break
    
    if np.isnan(rmse):
        rmse = np.sqrt(np.nanmean(matrix_d))

    result['fevs'].append(rmse)
    print(f"rmse={rmse}")
    return rmse

def printx(x):
    print(x)

if not os.path.exists("./correlation"):
    os.mkdir("./correlation/")

# obtain arguments of this script
code_distance = int(sys.argv[1])
num_round = int(sys.argv[2])
initial_state = int(sys.argv[3])
print(code_distance, num_round, initial_state)

num_shot = 1000000

filename = f"./correlation/optimized_errorrate_d{code_distance}_r{num_round}_i{initial_state}.npy"
if os.path.exists(filename):
    exit()

filename = f"./noise_model/noise_model_device_d{code_distance}.json"
with open(filename, "r") as fin:
    json_str = fin.read()
noise_model = NoiseModel.model_validate_json(json_str)

rmse_orig = eval_rmse([-1], code_distance, num_round, initial_state, num_shot, noise_model)
print(f"Original RMSE = {rmse_orig}")

filename = f"./correlation/correlation_matrix_experiment_d{code_distance}_r{num_round}_i{initial_state}.npy"
if os.path.exists(filename):
    with open(filename, "rb") as fin:
        matrix_e = np.load(fin)
else:
    matrix_e = analyze_experimental_data(
        code_distance, num_round, initial_state, num_shot
    )
    with open(filename, "wb") as fout:
        np.save(fout, matrix_e)

matrix_s = analyze_simulation_data(
    code_distance, num_round, initial_state, num_shot
)

x0 = [x.error_rate for x in noise_model.noise_property_list]

# oprimization configurations
b = ((x.error_rate/10, min(x.error_rate*10, 0.7)) for x in noise_model.noise_property_list)
_tol = 1e-2

res = optimize.minimize(
        eval_rmse,
        x0,
        args=(code_distance, num_round, initial_state, num_shot, noise_model),
        method='powell',
        #method='L-BFGS-B',
        #method='BFGS',
        #method='SLSQP',
        tol=_tol,
        bounds=b,
        options={"ftol": _tol, "xtol": _tol},
        callback=printx)

result['x0'] = x0
result['x'] = res.x.tolist()
result['success'] = res.success
result['message'] = res.message

# print optimization result
print(res)
print(x0)
print(res.x)

print(f"Original RMSE = {rmse_orig}")

rmse_opti = eval_rmse(res.x, code_distance, num_round, initial_state, num_shot, noise_model)
print(f"Optimized RMSE = {rmse_opti}")

# save correlation matrix of simulation with optimized error rate
filename = f"./correlation/correlation_matrix_optimize_d{code_distance}_r{num_round}_i{initial_state}.npy"
matrix_o = analyze_simulation_data(
    code_distance, num_round, initial_state, num_shot
)
with open(filename, "wb") as fout:
    np.save(fout, matrix_o)

# save diff of correlation matrices of experiment and simulation with optimized error rate
matrix_d = matrix_e - matrix_o
filename = f"./correlation/correlation_matrix_diff_d{code_distance}_r{num_round}_i{initial_state}.npy"
with open(filename, "wb") as fout:
    np.save(fout, matrix_d)

# dump optimization result 
filename = f"./correlation/correlation_matrix_optres_d{code_distance}_r{num_round}_i{initial_state}.json"
json_str = json.dumps(result, ensure_ascii=True, indent=4)
with open(filename, "wt") as fout:
    fout.write(json_str)
