import base64
from enum import Enum
from dataclasses import field, dataclass
import numpy as np
from pydantic import BaseModel, field_validator, field_serializer, ConfigDict


@dataclass(frozen=True, slots=True)
class RawDataset:
    """Raw dataset taken from experiment

    classifier
      Rawdata for readout classifier.
      classifier[qubit_name] has a dictionary relating "0" and "1" to a single complex value.
      Each has the center IQ-point when measuring |0>- and |1>-state qubits.

    readout_raw_data
      Reference IQ points for qubit readout.
      readout_raw_data[qubit_name] has a dictionary relating "0" and "1" to many measurement values.
      Each has a collection of single-shot measurement data, which is represented as a complex value with integration.

    raw_data
      Main data for syndrome measurements.
      Each item corresponds to a shot of QEC circuit.
      For the i-th sample, a record taked at j-th round for qubit named Qxx can be taked by 'raw_data[i][j][Qxx]'
    """

    classifier: dict[str, dict[int, complex]]
    readout_raw_data: dict[str, dict[int, list[complex]]]
    raw_data: list


class RecordInformation(BaseModel):
    """Record information

    qubit_index: qubit index that is used for taking record. Index includes measurement qubits.
    round_index: index of syndrome extraction round
    """

    qubit_index: int
    round_index: int


class GateType(Enum):
    UNKNOWN = 0
    INIT0 = 10
    INIT1 = 11
    CNOT = 12
    IDLE_CNOT = 13
    IDLE_MEAS = 14
    MEAS = 15
    CLOCK = 16


class Gate(BaseModel):
    """Gate object

    gate_type: type of gates
    target_qubit_list: list of target qubits of gate
    """

    gate_type: GateType
    target_qubit_list: list[int]


class Circuit(BaseModel):
    """Quantum circuit

    num_qubit: number of qubits in this circuit
    num_round: number of syndrome extraction rounds of codes
    code_distance: code distance
    moment_list: list of moment. Each moment corresponds to a layer of quantum circuits, and moment consists of gates.
    record_info_list: list of record information taken in this circuit
    detector_list: list of detectors. The i-th element is a list of record, and XOR of them gives the i-th detector.
    observable_list: list of observables. The i-th element is a list of record, and XOR of them gives the i-th observable.
    """

    num_qubit: int
    num_round: int
    code_distance: int
    moment_list: list[list[Gate]] = field(default_factory=list)
    record_info_list: list[RecordInformation] = field(default_factory=list)
    detector_list: list[list[RecordInformation]] = field(default_factory=list)
    observable_list: list[list[RecordInformation]] = field(default_factory=list)


class NoiseType(Enum):
    UNKNOWN = 0
    UNIFORM_DEPOLARIZE = 10
    AMPLITUDE_DAMPING = 11
    COHERENT_XX_ERROR = 12



class NoiseProperty(BaseModel):
    """Noise property

    This class describes what kind of noise happens on quantum gates acting on a specific list of qubits.
    This class assumes noise depends on gates and qubits, and assumes noise map is depolarizing.

    gate_type: type of gate
    target_qubit_list: target qubit list
    error_rate: depolarizing error rate
    """

    gate_type: GateType
    target_qubit_list: list[int]
    error_rate: float
    noise_type: NoiseType = NoiseType.UNIFORM_DEPOLARIZE
    error_info: dict = field(default_factory=dict)

    def to_json(self):
        return {"gate_type": self.gate_type.name}

    @classmethod
    def from_json(cls, d):
        return cls(gate_type=GateType[d["gate_type"]])


class NoiseModel(BaseModel):
    """Noise model

    noise_property_list: list of noise property, each of them has a information of noise map for a gate type on certain qubits.
    """

    noise_property_list: list[NoiseProperty] = field(default_factory=list)


class RecordDataset(BaseModel):
    """Record dataset

    code_distance: code distance
    num_round: number of syndrome extraction rounds
    record_info_list: list of record information
    record_dataset: A matrix of bits. The record_dataset[i,j] corresponds to j-th record taken in the i-th shot of experiment. The information of the j-th record is provided in the j-th record_info_list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    code_distance: int
    num_round: int
    record_info_list: list[RecordInformation]
    record_dataset: np.ndarray

    @field_serializer("record_dataset")
    def serialize(self, v):
        return {
            "shape": v.shape,
            "dtype": str(v.dtype),
            "data": base64.b64encode(v.tobytes()).decode(),
        }

    @field_validator("record_dataset", mode="before")
    @classmethod
    def deserialize(cls, v):
        if isinstance(v, dict):
            data = base64.b64decode(v["data"])
            return np.frombuffer(data, dtype=v["dtype"]).reshape(v["shape"])
        return v
