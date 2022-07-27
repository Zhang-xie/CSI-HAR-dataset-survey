import zipfile
import numpy as np
from pathlib import Path
import h5py
import hdf5plugin

import helper


def raw2zip(root, outzippath):
    root = Path(root) / "data"
    amp_datatest_activity_label = np.load(root / "amp_test_activity_label.npy")
    amp_datatest_data = np.load(root / "amp_test_data.npy")
    amp_datatest_location_label = np.load(root / "amp_test_location_label.npy")
    amp_datatrain_activity_label = np.load(root / "amp_train_activity_label.npy")
    amp_datatrain_data = np.load(root / "amp_train_data.npy")
    amp_datatrain_location_label = np.load(root / "amp_train_location_label.npy")

    pha_datatest_data = np.load(root / "pha_test_data.npy")
    pha_datatrain_data = np.load(root / "pha_train_data.npy")

    act_label = np.squeeze(
        np.concatenate([amp_datatest_activity_label, amp_datatrain_activity_label], 0)
    )
    env_label = np.squeeze(
        np.concatenate([amp_datatest_location_label, amp_datatrain_location_label], 0)
    )
    data = np.squeeze(
        np.concatenate(
            [
                amp_datatest_data * np.exp(1j * pha_datatest_data),
                amp_datatrain_data * np.exp(1j * pha_datatrain_data),
            ],
            0,
        )
    )
    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:
        helper.write_array2zip("act_label", act_label, zip_out_f)
        helper.write_array2zip("env_label", env_label, zip_out_f)
        helper.write_array2zip("raw", data, zip_out_f)
