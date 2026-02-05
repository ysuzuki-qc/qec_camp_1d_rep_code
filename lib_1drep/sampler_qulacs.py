import numpy as np
import tqdm
import qulacs
from lib_1drep.data import (
    Circuit, 
    NoiseModel, 
    NoiseProperty, 
    RecordDataset, 
    GateType, 
    NoiseType
)


def _get_noise_property(
    noise_model: NoiseModel, gate_type: GateType, target_qubit_list: list[int]
) -> NoiseProperty:
    if gate_type in [GateType.CLOCK]:
        return NoiseProperty(
            gate_type=GateType.UNKNOWN, target_qubit_list=[], error_rate=0.0
        )
    for noise_property in noise_model.noise_property_list:
        if (
            noise_property.gate_type == gate_type
            and noise_property.target_qubit_list == target_qubit_list
        ):
            return noise_property
    raise ValueError(
        f"noise information for gate_type={gate_type} target_qubit_list={target_qubit_list} not found"
    )


def process_noise(qulacs_circuit: qulacs.QuantumCircuit, noise_property: NoiseProperty) -> None:
    if noise_property.noise_type == NoiseType.UNIFORM_DEPOLARIZE:
        if len(noise_property.target_qubit_list) == 1:
            qulacs_circuit.add_gate(qulacs.gate.DepolarizingNoise(noise_property.target_qubit_list[0], noise_property.error_rate))
        elif len(noise_property.target_qubit_list) == 2:
            qulacs_circuit.add_gate(qulacs.gate.TwoQubitDepolarizingNoise(noise_property.target_qubit_list[0], noise_property.target_qubit_list[1], noise_property.error_rate))
        else:
            raise ValueError(f"Unsupported noise: {noise_property}")
    elif noise_property.noise_type == NoiseType.AMPLITUDE_DAMPING:
        for qubit_index in noise_property.target_qubit_list:
            qulacs_circuit.add_gate(qulacs.gate.AmplitudeDampingNoise(qubit_index, noise_property.error_rate))
    elif noise_property.noise_type == NoiseType.COHERENT_XX_ERROR:
        if len(noise_property.target_qubit_list) == 1:
            angle = np.sqrt(noise_property.error_rate)
            qulacs_circuit.add_gate(qulacs.gate.PauliRotation(noise_property.target_qubit_list, [1], angle))
        elif len(noise_property.target_qubit_list) == 2:
            angle = np.sqrt(noise_property.error_rate/3)
            qulacs_circuit.add_gate(qulacs.gate.PauliRotation(noise_property.target_qubit_list, [1, 1], angle))
            qulacs_circuit.add_gate(qulacs.gate.PauliRotation([noise_property.target_qubit_list[0]], [1,], angle))
            qulacs_circuit.add_gate(qulacs.gate.PauliRotation([noise_property.target_qubit_list[1]], [1,], angle))
        else:
            raise ValueError(f"Unsupported noise: {noise_property}")
    else:
        raise ValueError(f"Unsupported noise: {noise_property}")


def _convert_circuit_to_qulacs(
    circuit: Circuit, noise_model: NoiseModel
) -> qulacs.QuantumCircuit:
    qc = qulacs.QuantumCircuit(circuit.num_qubit)
    record_index = 0
    for moment in circuit.moment_list:
        for gate in moment:
            noise_property = _get_noise_property(
                noise_model, gate.gate_type, gate.target_qubit_list
            )
            if gate.gate_type == GateType.INIT0:
                process_noise(qc, noise_property)
            elif gate.gate_type == GateType.INIT1:
                process_noise(qc, noise_property)
                qc.add_gate(qulacs.gate.X(gate.target_qubit_list[0]))
            elif gate.gate_type == GateType.CNOT:
                qc.add_gate(qulacs.gate.CNOT(gate.target_qubit_list[0], gate.target_qubit_list[1]))
                process_noise(qc, noise_property)
            elif gate.gate_type == GateType.MEAS:
                qc.add_gate(qulacs.gate.Measurement(gate.target_qubit_list[0], record_index))
                record_index += 1
            elif gate.gate_type == GateType.IDLE_CNOT:
                process_noise(qc, noise_property)
            elif gate.gate_type == GateType.IDLE_MEAS:
                process_noise(qc, noise_property)
            elif gate.gate_type == GateType.CLOCK:
                pass
            else:
                raise ValueError(f"Unknown gate: {gate.gate_type}")
    return qc


def sample_records_with_qulacs(
    circuit: Circuit, num_shot: int, noise_model: NoiseModel
) -> RecordDataset:
    """Sampling records with circuit and noise model

    Args:
        circuit (Circuit): quantum circuit
        num_shot (int): number of shots
        noise_model (NoiseModel): noise model
        seed (int): seed for random sampling

    Returns:
        RecordDataset: record dataset
    """
    qulacs_circuit = _convert_circuit_to_qulacs(circuit, noise_model)
    state_vector = qulacs.StateVector(circuit.num_qubit)
    num_record = len(circuit.record_info_list)
    record_dataset = np.zeros(shape=(num_shot, num_record), dtype=np.int8)
    for shot_index in tqdm.tqdm(range(num_shot)):
        state_vector.set_zero_state()
        qulacs_circuit.update_quantum_state(state_vector)
        for record_index in range(num_record):
            record_dataset[shot_index, record_index] = state_vector.get_classical_value(record_index)

    dataset = RecordDataset(
        code_distance=circuit.code_distance,
        num_round=circuit.num_round,
        record_info_list=circuit.record_info_list.copy(),
        record_dataset=record_dataset,
    )
    return dataset
