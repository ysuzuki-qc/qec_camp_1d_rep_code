import tqdm
import numpy as np
from lib_1drep.data import RecordInformation, RecordDataset
from lib_1drep.load import RawDataset


def _create_record_information(
    code_disntace: int, num_round: int
) -> list[RecordInformation]:
    """Create list of record information of raw dataset

    Args:
        code_disntace (int): code distance
        num_round (int): number of syndrome extraction rounds

    Returns:
        list[RecordInformation]: list of record infromation
    """
    result: list[RecordInformation] = []
    num_qubit = 2 * code_disntace - 1
    data_qubit_index_list = np.arange(0, num_qubit, 2)
    meas_qubit_index_list = np.arange(1, num_qubit, 2)

    for round_index in range(num_round):
        for qubit_index in meas_qubit_index_list:
            result.append(
                RecordInformation(qubit_index=qubit_index, round_index=round_index)
            )
        if round_index + 1 == num_round:
            for qubit_index in data_qubit_index_list:
                result.append(
                    RecordInformation(qubit_index=qubit_index, round_index=round_index)
                )
    return result


def convert_to_record_array(
    raw_dataset: RawDataset, code_distance: int, num_round: int, qubit_mapping: list[str]
) -> RecordDataset:
    """Convert raw dataset into record array

    Args:
        raw_dataset (RawDataset): raw dataset
        code_distance (int): code distance
        num_round (int): number of rounds
        qubit_mapping (list[str]): qubit mapping from chip-index to qubit index

    Returns:
        RecordDataset: record dataset
    """

    # create record list
    print("create record information")
    record_info_list = _create_record_information(code_distance, num_round)
    num_record = len(record_info_list)

    # format as IQ-record list
    print("convert raw data to record array")
    num_shot = len(raw_dataset.raw_data)
    IQ_record_dataset: np.ndarray = np.zeros(
        shape=(num_shot, num_record), dtype=complex
    )
    for shot_index in tqdm.tqdm(range(num_shot)):
        shot_data = raw_dataset.raw_data[shot_index]
        IQ_record_list: list[complex] = []
        for record_info in record_info_list:
            qubit_name = qubit_mapping[record_info.qubit_index]
            IQ_record_list.append(shot_data[record_info.round_index][qubit_name])
        IQ_record_dataset[shot_index, :] = IQ_record_list

    # convert IQ-complex to binary value
    print("convert IQ-value to binary value")
    record_dataset = np.zeros_like(IQ_record_dataset, dtype=np.int8)
    for record_index, record_info in enumerate(record_info_list):
        qubit_name = qubit_mapping[record_info.qubit_index]
        center_0 = raw_dataset.classifier[qubit_name][0]
        center_1 = raw_dataset.classifier[qubit_name][1]
        filtered_IQ = IQ_record_dataset[:, record_index]
        distance_0 = np.abs(center_0 - filtered_IQ) ** 2
        distance_1 = np.abs(center_1 - filtered_IQ) ** 2
        record_dataset[:, record_index] = distance_0 > distance_1

    dataset = RecordDataset(
        code_distance=code_distance,
        num_round=num_round,
        record_info_list=record_info_list,
        record_dataset=record_dataset,
    )
    return dataset
