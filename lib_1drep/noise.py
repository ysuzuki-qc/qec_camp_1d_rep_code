from lib_1drep.data import NoiseModel, NoiseProperty, GateType


def create_1drep_uniform_noise_model(
    error_rate: float, code_distance: int
) -> NoiseModel:
    """Create uniform noise model for 1d rep code

    Args:
        error_rate (float): error rates
        code_distance (int): code distance

    Returns:
        NoiseModel: uniform noise model
    """
    noise_model = NoiseModel()
    num_qubit = code_distance * 2 - 1
    gate_type_list_1Q = [
        GateType.INIT0,
        GateType.INIT1,
        GateType.MEAS,
        GateType.IDLE_CNOT,
        GateType.IDLE_MEAS,
    ]
    for qubit_index in range(num_qubit):
        for gate_type in gate_type_list_1Q:
            noise_model.noise_property_list.append(
                NoiseProperty(
                    gate_type=gate_type,
                    target_qubit_list=[qubit_index],
                    error_rate=error_rate,
                )
            )

    gate_type_list_2Q = [GateType.CNOT]
    for qubit_index in range(num_qubit - 1):
        for gate_type in gate_type_list_2Q:
            noise_model.noise_property_list.append(
                NoiseProperty(
                    gate_type=gate_type,
                    target_qubit_list=[qubit_index, qubit_index + 1],
                    error_rate=error_rate,
                )
            )
            noise_model.noise_property_list.append(
                NoiseProperty(
                    gate_type=gate_type,
                    target_qubit_list=[qubit_index + 1, qubit_index],
                    error_rate=error_rate,
                )
            )
    return noise_model
