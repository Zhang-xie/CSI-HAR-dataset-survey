from pathlib import Path
import zipfile
import numpy as np, numpy
import csv
import glob
import os
import pandas as pd
import io
import helper

from tqdm import tqdm, trange

window_size = 1000
threshold = 60
slide_size = 200  # less than window_size!!!


def dataimport(path1, path2):
    # xx = np.empty([0, window_size, 90], float)
    yy = np.empty([0, 8], float)

    ###Input data###
    # data import from csv
    input_csv_files = sorted(glob.glob(path1))
    xx = []
    for f in tqdm(input_csv_files):
        # print("input_file_name=", f)
        tmp1 = pd.read_csv(f, header=None).to_numpy(dtype=np.float32)
        # data import by slide window
        k = 0
        while k <= (len(tmp1) + 1 - 2 * window_size):
            x = tmp1[k : k + window_size, 1:91]
            xx.append(x)
            k += slide_size

    xx = np.array(xx)
    xx = xx.reshape(len(xx), -1)

    ###Annotation data###
    # data import from csv
    annotation_csv_files = sorted(glob.glob(path2))
    for ff in tqdm(annotation_csv_files):
        # print("annotation_file_name=", ff)
        ano_data = [[str(elm) for elm in v] for v in csv.reader(open(ff, "r"))]
        tmp2 = np.array(ano_data)

        # data import by slide window
        y = np.zeros(((len(tmp2) + 1 - 2 * window_size) // slide_size + 1, 8))
        k = 0
        while k <= (len(tmp2) + 1 - 2 * window_size):
            y_pre = np.stack(np.array(tmp2[k : k + window_size]))
            bed = 0
            fall = 0
            walk = 0
            pickup = 0
            run = 0
            sitdown = 0
            standup = 0
            noactivity = 0
            for j in range(window_size):
                if y_pre[j] == "bed":
                    bed += 1
                elif y_pre[j] == "fall":
                    fall += 1
                elif y_pre[j] == "walk":
                    walk += 1
                elif y_pre[j] == "pickup":
                    pickup += 1
                elif y_pre[j] == "run":
                    run += 1
                elif y_pre[j] == "sitdown":
                    sitdown += 1
                elif y_pre[j] == "standup":
                    standup += 1
                else:
                    noactivity += 1

            if bed > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 1, 0, 0, 0, 0, 0, 0])
            elif fall > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            elif walk > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            elif pickup > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            elif run > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            elif sitdown > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 0, 0, 1, 0])
            elif standup > window_size * threshold / 100:
                y[k // slide_size, :] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
            else:
                y[k // slide_size, :] = np.array([2, 0, 0, 0, 0, 0, 0, 0])
            k += slide_size

        yy = np.concatenate((yy, y), axis=0)
    print(xx.shape, yy.shape)
    return (xx, yy)


def raw2zip(root, outzippath):
    root = Path(root)
    # h5f = h5py.File("csi.hdf5", "w")

    final_x = []
    final_y = []
    for i, label in enumerate(
        ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]
    ):
        filepath1 = (root / f"Dataset/Data/input_{label}*.csv").as_posix()
        filepath2 = (root / f"Dataset/Data/annotation_*{label}*.csv").as_posix()

        x, y = dataimport(filepath1, filepath2)
        print(label + "finish!")

        print("csv file importing...")

        SKIPROW = (
            1  # Skip every 2 rows -> overlap 800ms to 600ms  (To avoid memory error)
        )
        # num_lines = sum(1 for l in open("input_files/xx_1000_60_" + str(i) + ".csv"))
        num_lines = min(len(x), len(y))
        # skip_idx = [x for x in range(1, num_lines) if x % SKIPROW != 0]
        skip_idx = np.array([x2 for x2 in range(1, num_lines) if x2 % SKIPROW == 0])

        xx = x[skip_idx]
        yy = y[skip_idx]
        # xx = np.array(
        #     pd.read_csv(
        #         "input_files/xx_1000_60_" + str(i) + ".csv", header=None, skiprows=skip_idx
        #     )
        # )
        # yy = np.array(
        #     pd.read_csv(
        #         "input_files/yy_1000_60_" + str(i) + ".csv", header=None, skiprows=skip_idx
        #     )
        # )

        # eliminate the NoActivity Data
        rows, cols = np.where(yy > 0)
        xx = np.delete(xx, rows[np.where(cols == 0)], 0)
        yy = np.delete(yy, rows[np.where(cols == 0)], 0)

        xx = xx.reshape(len(xx), 1000, 90)

        # 1000 Hz to 500 Hz (To avoid memory error)
        # xx = xx[:, ::2, :90]

        # if i == 0:
        #     h5f.create_dataset("raw", data=xx)
        #     h5f.create_dataset("label", data=yy.shape[0] * [i])
        # else:
        #     h5f["raw"].append(xx)
        #     h5f["label"].append(yy.shape[0] * [i])
        print(i, "finished...", "xx=", xx.shape, "yy=", yy.shape)
        final_x.append(xx)
        final_y.extend(yy.shape[0] * [i])
        print(yy.sum(0))
    final_x = np.concatenate(final_x, 0)
    final_y = np.array(final_y)
    print(f"x:{final_x.shape} y:{final_y.shape}")
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        helper.write_array2zip("raw", final_x, zip_out_f)
        helper.write_array2zip("act_label", final_y, zip_out_f)
