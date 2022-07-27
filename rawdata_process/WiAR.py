import zipfile
import numpy as np
from pathlib import Path
import h5py
import rarfile
import pandas as pd
import re
import io
from tqdm import tqdm
from scipy.io import loadmat

import helper

user_map = {
    "volunteerD_1": 4,
    "volunteerD_2": 4,
    "volunteerE_1": 5,
    "volunteerE_2": 5,
    "volunteer_10": 10,
    "volunteer_9": 9,
    "volunteer_8": 8,
    "volunteer_7": 7,
    "volunteer_a_all_data": 1,
    "volunteer_b_all_data": 2,
    "volunteer_c_all_data": 3,
    "volunteer_f_all_data": 6,
}

pattern = re.compile(r".*csi_a(\d+)_(\d+)")


def raw2zip(root, outzippath):
    root = Path(root) / "data"

    act_label = []
    user_label = []
    total_idx = 0
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        for p1 in root.glob("*.rar"):
            with rarfile.RarFile(p1, "r") as rar_f:
                for rar_info in tqdm(rar_f.infolist()):
                    re_result = re.findall(pattern, rar_info.filename)
                    if len(re_result) == 0:
                        # print(rar_info.filename)
                        continue
                    (act, _) = map(int, re_result[0])

                    act_label.append(act)
                    user_label.append(user_map[p1.stem])

                    b_dat = rar_f.read(rar_info)
                    csi = helper.get_csi_from_bytes(b_dat)
                    helper.write_array2zip(f"raw/{total_idx}", csi, zip_out_f)
                    total_idx += 1
        helper.write_array2zip(f"user_label", user_label, zip_out_f)
        helper.write_array2zip(f"act_label", act_label, zip_out_f)
