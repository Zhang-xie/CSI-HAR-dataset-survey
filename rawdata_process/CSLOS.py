import numpy as np
from pathlib import Path
import zipfile
import pandas as pd
import re
import io
from tqdm import tqdm

import helper


def raw2zip(root, outzippath):
    root = Path(root)

    total_idx = 0

    pattern = re.compile(r"E(\d+)_S(\d+)_C(\d+)_A(\d+)_T(\d+).csv")

    csi_t2 = np.zeros((3, 30), dtype=np.complex128)

    user_label = []
    act_label = []
    env_label = []
    experi_label = []
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        for p1 in tqdm(list(root.glob("*/*.zip")), position=0):
            with zipfile.ZipFile(p1, "r") as zip_f:
                for zip_info in tqdm(zip_f.infolist(), position=1, leave=False):
                    (env, user, experi, act, t) = map(
                        int, re.findall(pattern, zip_info.filename)[0]
                    )
                    df = pd.read_csv(io.BytesIO(zip_f.read(zip_info)))
                    arrays = df.to_records(index=False)
                    csi_t1 = np.zeros((len(arrays), 3, 30), dtype=np.complex128)
                    for k, arr in enumerate(arrays):
                        for i in range(3):
                            for j in range(30):
                                csi_t2[i, j] = complex(
                                    arr[f"csi_1_{i+1}_{j+1}"]
                                    .replace("i", "j")
                                    .replace("+-", "-")
                                )

                        csi_t1[k] = helper.get_scaled_csi(
                            helper.numpyrec2BfResult(arr, csi_t2)
                        )
                    helper.write_array2zip(f"raw/{total_idx}", csi_t1, zip_out_f)
                    total_idx += 1
                    env_label.append(env)
                    user_label.append(user)
                    act_label.append(act)
                    experi_label.append(experi)
        helper.write_array2zip("env_label", env_label, zip_out_f)
        helper.write_array2zip("user_label", user_label, zip_out_f)
        helper.write_array2zip("act_label", act_label, zip_out_f)
        helper.write_array2zip("experi_label", experi_label, zip_out_f)
