from lib_1drep.data import Circuit, Gate, GateType, RecordInformation


def _initialize(circuit: Circuit, code_distance: int, initial_state: int) -> None:
    """Add initialization gates to quantum circuits

    Args:
        circuit (Circuit): circuit
        code_distance (int): code distance
        initial_state (int): initial state, which must be 0 or 1
    """
    num_qubit = 2 * code_distance - 1
    moment: list[Gate] = []
    if initial_state == 0:
        for qubit_index in range(num_qubit):
            moment.append(
                Gate(gate_type=GateType.INIT0, target_qubit_list=[qubit_index])
            )
    else:
        for qubit_index in range(num_qubit):
            moment.append(
                Gate(gate_type=GateType.INIT1, target_qubit_list=[qubit_index])
            )
    circuit.moment_list.append(moment)


def _round(circuit: Circuit, code_distance: int, round_index: int) -> None:
    """Add syndrome extraction round of 1d rep code to quantum circuits

    Args:
        circuit (Circuit): circuit
        code_distance (int): code distance
        round_index (int): index of current syndrome extraction round
    """
    num_qubit = 2 * code_distance - 1

    moment1: list[Gate] = []
    for qubit_index in range(0, num_qubit - 2, 2):
        moment1.append(
            Gate(
                gate_type=GateType.CNOT,
                target_qubit_list=[qubit_index, qubit_index + 1],
            )
        )
    moment1.append(
        Gate(gate_type=GateType.IDLE_CNOT, target_qubit_list=[num_qubit - 1])
    )
    circuit.moment_list.append(moment1)

    moment2: list[Gate] = []
    moment2.append(Gate(gate_type=GateType.IDLE_CNOT, target_qubit_list=[0]))
    for qubit_index in range(1, num_qubit - 1, 2):
        moment2.append(
            Gate(
                gate_type=GateType.CNOT,
                target_qubit_list=[qubit_index + 1, qubit_index],
            )
        )
    circuit.moment_list.append(moment2)

    moment3: list[Gate] = []
    for qubit_index in range(1, num_qubit, 2):
        moment3.append(Gate(gate_type=GateType.MEAS, target_qubit_list=[qubit_index]))
        circuit.record_info_list.append(
            RecordInformation(qubit_index=qubit_index, round_index=round_index)
        )
    for qubit_index in range(0, num_qubit, 2):
        moment3.append(
            Gate(gate_type=GateType.IDLE_MEAS, target_qubit_list=[qubit_index])
        )
    moment3.append(Gate(gate_type=GateType.CLOCK, target_qubit_list=[]))
    circuit.moment_list.append(moment3)


def _last_round(circuit: Circuit, code_distance: int, round_index: int) -> None:
    """Add last round of syndrome extraction of 1d rep code to quantum circuits

    Args:
        circuit (Circuit): circuit
        code_distance (int): code distance
        round_index (int): index of current syndrome extraction round
    """
    num_qubit = 2 * code_distance - 1

    moment1: list[Gate] = []
    for qubit_index in range(0, num_qubit - 2, 2):
        moment1.append(
            Gate(
                gate_type=GateType.CNOT,
                target_qubit_list=[qubit_index, qubit_index + 1],
            )
        )
    moment1.append(
        Gate(gate_type=GateType.IDLE_CNOT, target_qubit_list=[num_qubit - 1])
    )
    circuit.moment_list.append(moment1)

    moment2: list[Gate] = []
    moment2.append(Gate(gate_type=GateType.IDLE_CNOT, target_qubit_list=[0]))
    for qubit_index in range(1, num_qubit - 1, 2):
        moment2.append(
            Gate(
                gate_type=GateType.CNOT,
                target_qubit_list=[qubit_index + 1, qubit_index],
            )
        )
    circuit.moment_list.append(moment2)

    moment3: list[Gate] = []
    for qubit_index in range(1, num_qubit, 2):
        moment3.append(Gate(gate_type=GateType.MEAS, target_qubit_list=[qubit_index]))
        circuit.record_info_list.append(
            RecordInformation(qubit_index=qubit_index, round_index=round_index)
        )
    for qubit_index in range(0, num_qubit, 2):
        moment3.append(Gate(gate_type=GateType.MEAS, target_qubit_list=[qubit_index]))
        circuit.record_info_list.append(
            RecordInformation(qubit_index=qubit_index, round_index=round_index)
        )
    moment3.append(Gate(gate_type=GateType.CLOCK, target_qubit_list=[]))
    circuit.moment_list.append(moment3)


def _validate(circuit: Circuit, num_qubit: int) -> None:
    """Validate quantum circuit

    Args:
        circuit (Circuit): circuit
        num_qubit (int): number of qubits
    """
    for moment in circuit.moment_list:
        touched_qubit: list[int] = []
        for gate in moment:
            if gate.gate_type == GateType.CNOT:
                assert len(gate.target_qubit_list) == 2
            elif gate.gate_type == GateType.CLOCK:
                assert len(gate.target_qubit_list) == 0
            else:
                assert len(gate.target_qubit_list) == 1
            for qubit_index in gate.target_qubit_list:
                assert 0 <= qubit_index and qubit_index < num_qubit
            touched_qubit.extend(gate.target_qubit_list)
        touched_qubit.sort()
        assert touched_qubit == list(range(num_qubit))


def _create_record_to_detector(
    code_distance: int, num_round: int
) -> list[list[RecordInformation]]:
    """Create relation from record to detectors

    Args:
        code_distance (int): code distance
        num_round (int): number of rounds

    Returns:
        list[list[RecordInformation]]: list of detectors. Each element is a list of information of record locations
    """

    result: list[list[RecordInformation]] = []
    meas_qubit_index_list = list(range(1, 2 * code_distance - 1, 2))
    assert num_round >= 2

    for qubit_index in meas_qubit_index_list:
        for round_index in range(num_round):
            records: list[RecordInformation] = []
            records.append(
                RecordInformation(qubit_index=qubit_index, round_index=round_index)
            )
            if round_index >= 2:
                records.append(
                    RecordInformation(
                        qubit_index=qubit_index, round_index=round_index - 2
                    )
                )
            result.append(records)

            if round_index == num_round - 1:
                records: list[RecordInformation] = []
                records.append(
                    RecordInformation(qubit_index=qubit_index, round_index=round_index)
                )
                records.append(
                    RecordInformation(
                        qubit_index=qubit_index, round_index=round_index - 1
                    )
                )
                records.append(
                    RecordInformation(
                        qubit_index=qubit_index - 1, round_index=round_index
                    )
                )
                records.append(
                    RecordInformation(
                        qubit_index=qubit_index + 1, round_index=round_index
                    )
                )
                result.append(records)

    return result


def _create_record_to_observable(
    code_distance: int, num_round: int
) -> list[list[RecordInformation]]:
    """Create relation from record to observable

    Args:
        code_distance (int): code distance
        num_round (int): number of rounds

    Returns:
        list[list[RecordInformation]]: list of observables. Each element is a list of information of record locations
    """
    result: list[list[RecordInformation]] = []
    records: list[RecordInformation] = []
    for qubit_index in range(0, 2 * code_distance - 1, 2):
        records.append(
            RecordInformation(qubit_index=qubit_index, round_index=num_round - 1)
        )
    result.append(records)

    return result


def create_1d_repetition_circuit(
    code_distance: int, num_round: int, initial_state: int
) -> Circuit:
    """Create syndrome extraction circuits of 1d rep codes

    Args:
        code_distance (int): code distance
        num_round (int): number of syndrome extraction rounds
        initial_state (int): initial states

    Returns:
        Circuit: _description_
    """
    assert(initial_state in [0,1])
    if code_distance % 2 == 0:
        raise ValueError("code distance must be odd number")
    num_qubit = 2 * code_distance - 1
    circuit = Circuit(
        num_qubit=num_qubit, num_round=num_round, code_distance=code_distance
    )
    _initialize(circuit, code_distance, initial_state)
    for round_index in range(num_round - 1):
        _round(circuit, code_distance, round_index)
    _last_round(circuit, code_distance, num_round - 1)
    _validate(circuit, num_qubit)
    circuit.detector_list = _create_record_to_detector(
        circuit.code_distance, circuit.num_round
    )
    circuit.observable_list = _create_record_to_observable(
        circuit.code_distance, circuit.num_round
    )

    return circuit
