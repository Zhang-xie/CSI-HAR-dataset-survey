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
    gDict = {}

    act_label = []
    total_idx = 0
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        for r1 in (root / "Label_CsiAmplitudeCut").glob("*"):
            recs = pd.read_csv(r1).to_records(index=False)
            for rec in tqdm(recs):
                startT = rec.startPoint
                endT = rec.endPoint
                label = rec.ativityCategory

                n1 = rec.fileName
                j1, j2, j3 = n1.split("_")

                datname = list(
                    (root / "Data_RawCSIDat").glob(f"{j1}*/*_{j2}_{j3}.dat")
                )[0]
                if datname not in gDict:
                    del gDict
                    gDict = {}
                    with open(datname, "rb") as dat_f:
                        gDict[datname] = helper.get_csi_from_bytes(dat_f.read())
                csi = gDict[datname][startT:endT]
                # print(f"{total_idx}:--{csi.shape} {csi.dtype}")
                helper.write_array2zip(f"raw/{total_idx}", csi, zip_out_f)
                total_idx += 1
                act_label.append(label)
        helper.write_array2zip(f"act_label", act_label, zip_out_f)
