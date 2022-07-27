import numpy as np
from pathlib import Path
import zipfile
import pandas as pd
import re
import io
from tqdm import tqdm
from scipy.io import loadmat

import helper

# dict_keys(['__header__', '__version__', '__globals__', 'csiu_lab', 'label_lab'])
# dict_keys(['__header__', '__version__', '__globals__', 'csid_lab', 'label_lab'])
# dict_keys(['__header__', '__version__', '__globals__', 'label', 'csi1', 'csi2', 'csi3', 'csi4', 'csi5'])
# dict_keys(['__header__', '__version__', '__globals__', 'csid_home', 'csiu_home', 'label_home'])


def raw2zip(root, outzippath):
    root = Path(root)

    act_label = []
    env_label = []
    user_label = []
    # env : lab 0  home 1
    raw = []

    # m1 = loadmat(root / "dataset_lab_276_ul.mat")
    m2 = loadmat(root / "dataset_lab_276_dl.mat")
    act_label.append(m2["label_lab"])
    env_label.append([0] * 5520)
    user_label.append([5] * 5520)
    raw.append(np.transpose(m2["csid_lab"], (3, 0, 2, 1)))

    m3 = loadmat(root / "dataset_lab_150.mat")
    act_label.append(m3["label"])
    env_label.append([0] * 7500)
    for i in range(5):
        raw.append(np.transpose(m3[f"csi{i+1}"], (3, 0, 2, 1)))
        user_label.append([i + 1] * 1500)

    m4 = loadmat(root / "dataset_home_276.mat")
    act_label.append(m4["label_home"])
    env_label.append([1] * 2760)
    user_label.append([5] * 2760)
    raw.append(np.transpose(m4["csid_home"], (3, 0, 2, 1)))

    act_label = np.concatenate(act_label).squeeze()
    env_label = np.concatenate(env_label).squeeze()
    user_label = np.concatenate(user_label).squeeze()
    raw = np.concatenate(raw).squeeze()

    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        helper.write_array2zip("env_label", env_label, zip_out_f)
        helper.write_array2zip("user_label", user_label, zip_out_f)
        helper.write_array2zip("act_label", act_label, zip_out_f)
        helper.write_array2zip("raw", raw, zip_out_f)
