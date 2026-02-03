import pickle
from lib_1drep.data import RawDataset


def load_dataset(dataset_path: str, num_round: int, initial_state: int) -> RawDataset:
    """Load dataset

    Args:
        num_round (int, optional): num rounds. Defaults to 8.
        initial_state (int, optional): initial logical state, which is either of 0 or 1. Defaults to 0.

    Returns:
        RawDataset: result
    """
    n_rounds_str = str(num_round)
    initial_state_str = str(initial_state)
    if initial_state not in [0, 1]:
        raise ValueError(f"initial state must be 0 or 1, but {initial_state} is given")

    if n_rounds_str != 1 and n_rounds_str != "1":
        dirname = dataset_path + str(n_rounds_str) + "_rounds/initial_" + str(initial_state_str)
    else:
        dirname = dataset_path + str(n_rounds_str) + "_round/initial_" + str(initial_state_str)

    print("load classifier")
    with open(dirname + "/classifiers.pickle", "rb") as f:
        classifiers = pickle.load(f)
    print("load raw_data")
    with open(dirname + "/raw_data.pickle", "rb") as f:
        raw_data = pickle.load(f)
    print("load readout_raw_data")
    with open(dirname + "/readout_raw_data/data.pickle", "rb") as f:
        readout_raw_data = pickle.load(f)

    raw_dataset = RawDataset(classifier=classifiers, raw_data=raw_data, readout_raw_data=readout_raw_data)
    return raw_dataset
