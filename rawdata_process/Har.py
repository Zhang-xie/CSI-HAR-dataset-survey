import zipfile
import numpy as np
from pathlib import Path
import h5py
import rarfile
import pandas as pd
import io
from tqdm import tqdm
import os

import helper


DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114

PHASE_MIN, PHASE_MAX = 3.1389, 3.1415
AMP_MIN, AMP_MAX = 0.0, 577.6582


def read_csi_data_from_csv(path_to_csv, is_five_hhz=False, antenna_pairs=4):
    """
    Read csi data(amplitude, phase) from .csv data
    :param path_to_csv: string
    :param is_five_hhz: boolean
    :param antenna_pairs: integer
    :return: (amplitudes, phases) => (np.array of shape(data len, num_of_subcarriers * antenna_pairs),
                                     np.array of shape(data len, num_of_subcarriers * antenna_pairs))
    """

    data = pd.read_csv(path_to_csv, header=None).values

    if is_five_hhz:
        subcarries_num = SUBCARRIES_NUM_FIVE_HHZ
    else:
        subcarries_num = SUBCARRIES_NUM_TWO_HHZ

    # 1 -> to skip subcarriers numbers in data
    amplitudes = data[:, subcarries_num * 1 : subcarries_num * (1 + antenna_pairs)]
    phases = data[
        :,
        subcarries_num * (1 + antenna_pairs) : subcarries_num * (1 + 2 * antenna_pairs),
    ]

    return amplitudes, phases


def read_labels_from_csv(path_to_csv):
    """
    Read labels(human activities) from csv file
    :param path_to_csv: string
    :return: labels, np.array of shape(data_len, 1)
    """

    data = pd.read_csv(path_to_csv, header=None).values
    labels = data[:, 1]

    return labels


def read_all_data_from_files(paths, is_five_hhz=True, antenna_pairs=4):
    """
    Read csi and labels data from all folders in the dataset
    :return: amplitudes, phases, labels all of shape (data len, num of subcarriers)
    """

    final_amplitudes, final_phases, final_labels = (
        np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)),
        np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)),
        np.empty((0)),
    )

    for index, path in enumerate(paths):
        amplitudes, phases = read_csi_data_from_csv(
            os.path.join(path, "data.csv"), is_five_hhz, antenna_pairs
        )
        labels = read_labels_from_csv(os.path.join(path, "label.csv"))

        amplitudes, phases = (
            amplitudes[:-1],
            phases[:-1],
        )  # fix the bug with the last element

        final_amplitudes = np.concatenate((final_amplitudes, amplitudes))
        final_phases = np.concatenate((final_phases, phases))
        final_labels = np.concatenate((final_labels, labels))

    return final_amplitudes, final_phases, final_labels


act_label_map = {
    "get_down": 0,
    "get_up": 1,
    "lying": 2,
    "no_person": 3,
    "sitting": 4,
    "standing": 5,
    "walking": 6,
}


def raw2zip(root, outzippath):
    root = Path(root) / "dataset_no_images"

    env_label = []
    env_label_idx = 0
    raw = []
    act_label = []
    for index, room in enumerate(DATA_ROOMS):
        for subroom in DATA_SUBROOMS[index]:
            # all_paths.append(os.path.join(root, room, subroom))
            p1 = os.path.join(root, room, subroom)
            amplitudes, phases, labels = read_all_data_from_files(
                [
                    p1,
                ]
            )
            env_label.extend([env_label_idx] * len(labels))
            env_label_idx += 1
            act_label.append(labels)
            raw.append(amplitudes * np.exp(1j * phases))
    env_label = np.array(env_label)
    act_label = np.concatenate(act_label)
    act_label = np.array([act_label_map[x] for x in act_label])
    raw = np.concatenate(raw)
    print(env_label.dtype, act_label.dtype, raw.dtype)
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        helper.write_array2zip("env_label", env_label, zip_out_f)
        helper.write_array2zip("act_label", act_label, zip_out_f)
        helper.write_array2zip("raw", raw, zip_out_f)


if __name__ == "__main__":
    raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/Har")
