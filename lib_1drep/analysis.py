import numpy as np
import stim
from lib_1drep.data import RecordDataset, Circuit
from lib_1drep.qec import _convert_circuit_to_stim
from lib_1drep.noise import create_1drep_uniform_noise_model


def get_two_point_correlation(circuit: Circuit, dataset: RecordDataset, limit_shot: int = -1) -> np.ndarray:
    """Generate matrix of two-point correlation of detector values.

    This function calculates the correlation matrix of two detectors.
    This can be used for checking the structure of detector graphs.

    Args:
        circuit (Circuit): circuit of syndrome extraction circuits
        dataset (RecordDataset): record dataset
        limit_shot (int, optional): If provided, limit the number of target data to this value. Defaults to -1.

    Returns:
        np.ndarray: two-point correlation matrix
    """
    noise_model = create_1drep_uniform_noise_model(1e-3, circuit.code_distance)
    stim_circuit: stim.Circuit = _convert_circuit_to_stim(circuit, noise_model)
    m2d_converter = stim_circuit.compile_m2d_converter()
    if limit_shot == -1:
        record = dataset.record_dataset.astype(bool)
    else:
        limit_shot = min(limit_shot, len(dataset.record_dataset))
        record = dataset.record_dataset[:limit_shot, :].astype(bool)
    detectors: np.ndarray = m2d_converter.convert(
        measurements=record, append_observables=False
    )

    num_detector = detectors.shape[1]
    correlation_matrix = np.zeros(shape=(num_detector, num_detector), dtype=float)
    for i in range(num_detector):
        xi = np.mean(detectors[:, i])
        for j in range(i + 1, num_detector):
            xj = np.mean(detectors[:, j])
            xij = np.mean(detectors[:, i] * detectors[:, j])
            value = 0.5 - 0.5 * np.sqrt(
                1 - 4 * (xij - xi * xj) / (1 - 2 * xi - 2 * xj + 4 * xij)
            )
            # value = (xij - xi * xj) / ((1 - 2 * xi)*(1 - 2 * xj))
            correlation_matrix[i, j] = value
            correlation_matrix[j, i] = value
    return correlation_matrix
