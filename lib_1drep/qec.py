import numpy as np
import stim
import pymatching
from lib_1drep.data import Circuit, GateType, RecordDataset, NoiseModel, NoiseType


def _get_error_rate(
    noise_model: NoiseModel, gate_type: GateType, target_qubit_list: list[int]
) -> float:
    """Get error rate of a gate in quantum circuits

    Args:
        noise_model (NoiseModel): noise model
        gate_type (GateType): gate type
        target_qubit_list (list[int]): target qubit list

    Raises:
        ValueError: cannot find noise property in noise model

    Returns:
        float: error rates
    """
    if gate_type in [GateType.CLOCK]:
        return 0.0
    for noise_property in noise_model.noise_property_list:
        if (
            noise_property.gate_type == gate_type
            and noise_property.target_qubit_list == target_qubit_list
        ):
            if noise_property.noise_type != NoiseType.UNIFORM_DEPOLARIZE:
                raise ValueError(f"Unknown noise type: {noise_property.noise_type} provided")
            return noise_property.error_rate
    raise ValueError(
        f"noise information for gate_type={gate_type} target_qubit_list={target_qubit_list} not found"
    )


def _convert_circuit_to_stim(circuit: Circuit, noise_model: NoiseModel) -> stim.Circuit:
    """Convert circuit object to stim's circuit object

    Args:
        circuit (Circuit): circuit
        noise_model (NoiseModel): noise model

    Returns:
        stim.Circuit: stim's circuit object
    """
    stim_circuit = stim.Circuit()
    for moment in circuit.moment_list:
        for gate in moment:
            error_rate = _get_error_rate(
                noise_model, gate.gate_type, gate.target_qubit_list
            )
            if gate.gate_type == GateType.INIT0:
                stim_circuit.append("R", gate.target_qubit_list[0])
                stim_circuit.append(
                    "DEPOLARIZE1", gate.target_qubit_list[0], error_rate
                )
            elif gate.gate_type == GateType.INIT1:
                stim_circuit.append("R", gate.target_qubit_list[0])
                stim_circuit.append("X", gate.target_qubit_list[0])
                stim_circuit.append(
                    "DEPOLARIZE1", gate.target_qubit_list[0], error_rate
                )
            elif gate.gate_type == GateType.CNOT:
                stim_circuit.append(
                    "CX", [gate.target_qubit_list[0], gate.target_qubit_list[1]]
                )
                stim_circuit.append(
                    "DEPOLARIZE2",
                    [gate.target_qubit_list[0], gate.target_qubit_list[1]],
                    error_rate,
                )
            elif gate.gate_type == GateType.MEAS:
                stim_circuit.append(
                    "DEPOLARIZE1", gate.target_qubit_list[0], error_rate / 2
                )
                stim_circuit.append("M", [gate.target_qubit_list[0]], error_rate / 2)
            elif gate.gate_type == GateType.IDLE_CNOT:
                stim_circuit.append(
                    "DEPOLARIZE1", gate.target_qubit_list[0], error_rate
                )
            elif gate.gate_type == GateType.IDLE_MEAS:
                stim_circuit.append(
                    "DEPOLARIZE1", gate.target_qubit_list[0], error_rate
                )
            elif gate.gate_type == GateType.CLOCK:
                # stim_circuit.append("TICK")
                pass
            else:
                raise ValueError(f"Unknown gate: {gate.gate_type}")
        stim_circuit.append("TICK")

    num_record = len(circuit.record_info_list)
    for detector in circuit.detector_list:
        target_list = []
        for record_info in detector:
            record_index = circuit.record_info_list.index(record_info)
            record_index_inv = -num_record + record_index
            target_list.append(stim.target_rec(record_index_inv))
        stim_circuit.append("DETECTOR", target_list, (detector[0].qubit_index, 0))

    for observable_index, observable in enumerate(circuit.observable_list):
        target_list = []
        for record_info in observable:
            record_index = circuit.record_info_list.index(record_info)
            record_index_inv = -num_record + record_index
            target_list.append(stim.target_rec(record_index_inv))
        stim_circuit.append("OBSERVABLE_INCLUDE", target_list, observable_index)

    fault_distance = len(stim_circuit.shortest_graphlike_error())
    if fault_distance != circuit.code_distance:
        raise ValueError(
            f"fault distance is {fault_distance} while code distance is {circuit.code_distance}"
        )

    return stim_circuit


def sample_records(
    circuit: Circuit, num_shot: int, noise_model: NoiseModel, seed: int
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
    stim_circuit = _convert_circuit_to_stim(circuit, noise_model)
    sampler = stim_circuit.compile_sampler(seed=seed)
    record_dataset = sampler.sample(num_shot)
    dataset = RecordDataset(
        code_distance=circuit.code_distance,
        num_round=circuit.num_round,
        record_info_list=circuit.record_info_list.copy(),
        record_dataset=record_dataset,
    )
    return dataset




def evaluation(
    circuit: Circuit,
    dataset: RecordDataset,
    noise_model: NoiseModel,
    limit_shot: int = -1,
) -> np.ndarray:
    """Evaluate error rates for

    Args:
        circuit (Circuit): circuit
        dataset (RecordDataset): record dataset
        noise_model (NoiseModel): noise model
        limit_shot (int, optional): If provided, this function only evaluate limited shot in dataset. Defaults to -1.

    Returns:
        np.ndarray: list of observable errors after correction. Sum of this object gives the total number of error-estimation failure.
    """
    stim_circuit = _convert_circuit_to_stim(circuit, noise_model)

    # convert record to detector and observable
    m2d_converter = stim_circuit.compile_m2d_converter()

    if limit_shot == -1:
        record = dataset.record_dataset.astype(bool)
    else:
        limit_shot = min(limit_shot, len(dataset.record_dataset))
        record = dataset.record_dataset[:limit_shot, :].astype(bool)
    detector_list, observable_list = m2d_converter.convert(
        measurements=record, separate_observables=True
    )

    detector_error_model = stim_circuit.detector_error_model(decompose_errors=False)
    decoder = pymatching.Matching.from_detector_error_model(detector_error_model)
    prediction_list = decoder.decode_batch(detector_list)
    estimation_error_list = np.logical_xor(observable_list, prediction_list)
    return estimation_error_list
