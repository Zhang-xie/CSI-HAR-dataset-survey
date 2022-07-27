import numpy as np
from pathlib import Path
import zipfile
import pandas as pd
import re
import io
from tqdm import tqdm
from PIL import Image

import helper

label_map = {
    "walk": 1,
    "stand": 2,
    "sit": 3,
    "fall_right": 4,
    "fall_left": 5,
    "fall_forward": 6,
}


def raw2zip(root, outzippath):
    root = Path(root)

    act_label = []
    raw = []
    for lp in label_map.keys():
        for p1 in (root / lp).glob("*.jpg"):
            arr = np.array(Image.open(p1))
            raw.append(arr.T)
            act_label.append(label_map[lp])

    raw = np.stack(raw)
    act_label = np.array(act_label)
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        helper.write_array2zip("act_label", act_label, zip_out_f)
        helper.write_array2zip("raw", raw, zip_out_f)
