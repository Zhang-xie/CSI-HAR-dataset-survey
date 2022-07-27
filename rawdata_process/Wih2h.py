import numpy as np
from pathlib import Path
import h5py
import rarfile
import zipfile
import pandas as pd
import re
import io
from tqdm import tqdm
from scipy.io import loadmat


def raw2zip(root, outzippath):
    from helper import numpyDict2BfResult, get_scaled_csi, write_array2zip

    root = Path(root)
    pattern = re.compile(r".*S(\d+)_S(\d+)_I(\d+)_T(\d+).mat")

    total_idx = 0

    user_label = []
    act_label = []

    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        for rarpath in tqdm(sorted(list(root.glob("*.rar"))), position=0):
            with rarfile.RarFile(rarpath, "r") as rar_f:
                for rar_info in tqdm(rar_f.infolist(), position=1, leave=False):
                    re_result = re.findall(pattern, rar_info.filename)
                    if len(re_result) == 0:
                        continue

                    (id0, id1, act, _) = map(int, re_result[0])
                    raw_arr = loadmat(rar_f.open(rar_info), simplify_cells=True)[
                        "Raw_Cell_Matrix"
                    ]

                    _csi = np.zeros((len(raw_arr), 2, 3, 30), dtype=np.complex128)
                    for i, sub_raw in enumerate(raw_arr):
                        _csi[i] = get_scaled_csi(numpyDict2BfResult(sub_raw))

                    write_array2zip(f"raw/{total_idx}", np.array(_csi), zip_out_f)
                    # g_raw.create_dataset(f"{total_idx}", data=np.array(_csi))
                    total_idx += 1

                    user_label.append(f"{id0}_{id1}")
                    act_label.append(act)
        outnmap = {}
        for i, n in enumerate(np.unique(user_label)):
            outnmap[n] = i
        print(outnmap)
        user_label = [outnmap[n] for n in user_label]
        print(user_label)

        write_array2zip(f"user_label", user_label, zip_out_f)
        write_array2zip(f"act_label", act_label, zip_out_f)
        # h5f.create_dataset("user_label", data=user_label)
        # h5f.create_dataset("act_label", data=act_label)


def testf(root, outzippath):
    root = Path(root)
    outzippath = Path(outzippath)
    pattern = re.compile(r".*S(\d+)_S(\d+)_I(\d+)_T(\d+).mat")

    total_idx = 0

    user_label = []
    act_label = []

    with zipfile.ZipFile(
        outzippath, "r", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        for rarpath in tqdm(sorted(list(root.glob("*.rar"))), position=0):
            with rarfile.RarFile(rarpath, "r") as rar_f:
                for rar_info in tqdm(rar_f.infolist(), position=1, leave=False):
                    re_result = re.findall(pattern, rar_info.filename)
                    if len(re_result) == 0:
                        continue

                    (id0, id1, act, _) = map(int, re_result[0])
                    # raw_arr = loadmat(rar_f.open(rar_info), simplify_cells=True)[
                    #     "Raw_Cell_Matrix"
                    # ]

                    # _csi = np.zeros((len(raw_arr), 2, 3, 30), dtype=np.complex128)
                    # for i, sub_raw in enumerate(raw_arr):
                    #     _csi[i] = get_scaled_csi(numpyDict2BfResult(sub_raw))

                    # write_array2zip(f"raw/{total_idx}", np.array(_csi), zip_out_f)
                    total_idx += 1

                    user_label.append(f"{id0}_{id1}")
                    act_label.append(act)
        outnmap = {}
        for i, n in enumerate(np.unique(user_label)):
            outnmap[n] = i
        print(outnmap)
        user_label = [outnmap[n] for n in user_label]
        print(user_label)
        # write_array2zip(f"user_label", user_label, zip_out_f)
        # write_array2zip(f"act_label", act_label, zip_out_f)


if __name__ == "__main__":
    testf(
        "/media/yk/Samsung_T5/CSI-HAR-Datasets/Wih2h",
        "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/Wih2h_data.zip",
    )
