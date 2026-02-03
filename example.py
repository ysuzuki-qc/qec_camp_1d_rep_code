import pprint
from lib_1drep.data import NoiseModel, RecordDataset
from lib_1drep.circuit import create_1d_repetition_circuit
from lib_1drep.qec import sample_records, _convert_circuit_to_stim
from lib_1drep.noise import create_1drep_uniform_noise_model

code_distance = 5
num_round = 4
initial_state = 0
num_shot = 10000
seed = 42
error_rate = 1e-2

# load device noise model
filename = f"./noise_model/noise_model_device_d{code_distance}.json"
with open(filename, "r") as fin:
    json_str = fin.read()
noise_model = NoiseModel.model_validate_json(json_str)
print("*"*100 + "\n"+"circuit")
pprint.pprint(noise_model.model_dump())

# load experimental record
filename = f"./record/record_dataset_device_d{code_distance}_r{num_round}_i{initial_state}.json"
with open(filename, "r") as fin:
    json_str = fin.read()
dataset = RecordDataset.model_validate_json(json_str)
dataset.record_dataset = dataset.record_dataset[:10,:]
print("*"*100 + "\n"+"circuit")
pprint.pprint(dataset.model_dump())

# create record data from simulation with noisy model
circuit = create_1d_repetition_circuit(code_distance, num_round, initial_state)
dataset = sample_records(circuit, num_shot, noise_model, seed)
print("*"*100 + "\n"+"circuit")
pprint.pprint(circuit.model_dump())

# create noise model with uniform error rate
noise_model = create_1drep_uniform_noise_model(error_rate, code_distance)

# create stim circuit from circuit and noise model
stim_circuit = _convert_circuit_to_stim(circuit, noise_model)
