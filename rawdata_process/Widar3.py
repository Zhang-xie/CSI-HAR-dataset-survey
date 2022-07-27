import itertools
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm
import h5py
import hdf5plugin

from helper import get_csi_from_bytes, get_csi_from_bytes_unscaled, write_array2zip

pattern = re.compile(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+).dat")
year_num_pattern = re.compile(r"(2018\d{4})")
widar3_env_label = {
    "20181109": 1,
    "20181112": 1,
    "20181115": 1,
    "20181116": 1,
    "20181117": 2,
    "20181118": 2,
    "20181121": 1,
    "20181127": 2,
    "20181128": 2,
    "20181130": 1,
    "20181204": 2,
    "20181205": 2,
    "20181208": 2,
    "20181209": 2,
    "20181211": 3,
}

widar3_gesture_mapping = {
    "20181109": [0, 1, 2, 3, 9, 10],
    "20181112": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "20181115": [0, 1, 2, 9, 10, 11],
    "20181116": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "20181117": [0, 1, 2, 9, 10, 11],
    "20181118": [0, 1, 2, 9, 10, 11],
    "20181121": [3, 4, 5, 6, 7, 8],
    "20181127": [3, 4, 5, 6, 7, 8],
    "20181128": [0, 1, 2, 4, 5, 8],
    "20181130": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "20181204": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "20181205": [3, 4, 5, 6, 7, 8],
    "20181208": [0, 1, 2, 3],
    "20181209": [0, 1, 2, 3, 5, 8],
    "20181211": [0, 1, 2, 3, 5, 8],
}
widar3_gesture_mapping = {k: np.array(v) + 1 for k, v in widar3_gesture_mapping.items()}


def widar3_infor(p):
    root_path = Path(p)
    for i, x1 in enumerate(
        itertools.chain.from_iterable(
            [
                root_path.glob("20181109.zip"),
                root_path.glob("20181115.zip"),
                root_path.glob("20181117.zip"),
                root_path.glob("20181118.zip"),
                root_path.glob("20181121.zip"),
                root_path.glob("20181127.zip"),
                root_path.glob("20181128.zip"),
                root_path.glob("20181130*.zip"),
                root_path.glob("20181204.zip"),
                root_path.glob("20181205.zip"),
                root_path.glob("20181208.zip"),
                root_path.glob("20181209.zip"),
                root_path.glob("20181211.zip"),
            ]
        )
    ):
        print(x1)
        archive = zipfile.ZipFile(x1, "r")
        l = []
        for x2 in archive.infolist():
            # print(x2.filename)
            if "yumeng" in x2.filename or "baidu" in x2.filename:
                print(x2)
                continue
            if not x2.is_dir():
                n = (x2.filename).split("/")[-1]
                try:
                    (
                        userid,
                        gesture,
                        location,
                        face_orientation,
                        sampleid,
                        receiverid,
                    ) = map(int, re.findall(pattern, n)[0])
                    l.append(
                        (
                            userid,
                            gesture,
                            location,
                            face_orientation,
                            sampleid,
                            receiverid,
                        )
                    )
                except BaseException as e:
                    print("Error:", x1, x2, n)
                    print(e)
        l = np.array(
            l,
            dtype=[
                ("userid", "i4"),
                ("gesture", "i4"),
                ("location", "i4"),
                ("face_orientation", "i4"),
                ("sampleid", "i4"),
                ("receiverid", "i4"),
            ],
        )
        print(np.unique(l["location"]))


def widar3_iter_process(extract_condition, in_path):
    root_path = Path(in_path)

    total_idx = 0

    mask_total = []

    multi_label_total = []
    with zipfile.ZipFile(
        "Widar3_data.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        for zip_idx, zip_name in enumerate(
            itertools.chain.from_iterable(
                [
                    # root_path.glob("20181109.zip"),
                    # root_path.glob("20181115.zip"),
                    # root_path.glob("20181117.zip"),
                    # root_path.glob("20181118.zip"),
                    # root_path.glob("20181121.zip"),
                    # root_path.glob("20181127.zip"),
                    # root_path.glob("20181128.zip"),
                    # root_path.glob("20181130*.zip"),
                    root_path.glob("20181204.zip"),  # 2.0G
                    root_path.glob("20181205.zip"),  # 2.7G
                    # root_path.glob("20181208.zip"),
                    # root_path.glob("20181209.zip"),
                    # root_path.glob("20181211.zip"),
                ]
            )
        ):
            zip_selected_info_list = []
            print(zip_name)
            print(f"len of data {sum(mask_total)}")
            year_num = re.findall(year_num_pattern, zip_name.name)[0]
            roomid = widar3_env_label[year_num]
            with zipfile.ZipFile(zip_name, "r") as zip_f:
                zip_file_info_list = zip_f.infolist()

                csi_info_list = []
                for zip_file_info in zip_file_info_list:
                    if (
                        "yumeng" in zip_file_info.filename
                        or "baidu" in zip_file_info.filename
                        or zip_file_info.is_dir()
                    ):
                        continue
                    file_name = (zip_file_info.filename).split("/")[-1]

                    (
                        userid,
                        gesture,
                        location,
                        face_orientation,
                        sampleid,
                        receiverid,
                    ) = map(int, re.findall(pattern, file_name)[0])
                    gesture = widar3_gesture_mapping[year_num][gesture - 1]

                    if (
                        True
                        # and gesture in extract_condition["gesture"]
                        # and receiverid in extract_condition["receiverid"]
                        # and face_orientation in extract_condition["face_orientation"]
                        # and location in extract_condition["location"]
                    ):
                        zip_selected_info_list.append(zip_file_info.filename)
                        csi_info_list.append(
                            (
                                roomid,
                                userid,
                                gesture,
                                location,
                                face_orientation,
                                sampleid,
                                receiverid,
                            )
                        )

                multi_label_total.extend(csi_info_list)
                mask = []
                for zip_file_info_selected in tqdm(zip_selected_info_list):
                    b_zip_dat = zip_f.read(zip_file_info_selected)
                    raw_csi = get_csi_from_bytes(b_zip_dat)
                    # raw_csi = get_csi_from_bytes_unscaled(b_zip_dat)
                    if len(raw_csi) < 900:
                        mask.append(False)
                        continue
                    else:
                        mask.append(True)

                    # raw_csi = apply_csi_filter(raw_csi, filter_fir)
                    # raw_csi, _ = conj_multi_csi(raw_csi)
                    # amp_csi, pha_csi = complex_to_amp_pha(raw_csi)
                    # array_l.append([raw_csi, amp_csi, pha_csi])
                    # array_l.append(raw_csi)
                    write_array2zip(f"raw/{total_idx}", raw_csi, zip_out_f)
                    total_idx += 1
                mask_total.extend(mask)
        # multi_label_total = np.array(
        #     multi_label_total,
        #     dtype=[
        #         ("roomid", "i4"),
        #         ("userid", "i4"),
        #         ("gesture", "i4"),
        #         ("location", "i4"),
        #         ("face_orientation", "i4"),
        #         ("sampleid", "i4"),
        #         ("receiverid", "i4"),
        #     ],
        # )
        # mask_total = np.array(mask_total)
        # process_idx_total = np.array(process_idx_total)
        # rm_info_all = multi_label_total[~mask_total]
        # multi_label_total = multi_label_total[mask_total]
        # np.save(out_path / "multi_label", multi_label_total)
        # np.save(out_path / "rm_info_all", rm_info_all)
        # np.save(out_path / "process_idx_total", process_idx_total)


def raw2zip(root, outzippath):
    extract_condition = {
        "receiverid": (3,),
        "face_orientation": (
            1,
            2,
            3,
        ),
        "location": (
            1,
            2,
            5,
        ),
        "gesture": (1, 2, 3, 4, 6, 9),
    }
    root = Path(root) / "CSI"
    widar3_iter_process(extract_condition, in_path=root)


# if __name__ == "__main__":
#     # test_widar_data()
#     extract_widar_data()
