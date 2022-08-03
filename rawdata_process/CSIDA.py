from pathlib import Path
import zarr
from tqdm import tqdm
import zipfile
import helper


def raw2zip(root, outzippath):
    root = Path(root)
    group = zarr.open_group(root.as_posix(), mode="r")
    gesture = group.csi_label_act[:]  # gesture: 0~5
    room = group.csi_label_env[:]  # room: 0,1
    location = group.csi_label_loc[:]  # location: 0,1  0,1,2
    location = (
        location + room * 3
    )  # combine location  and room into five types:0,1,2,3,4
    user = group.csi_label_user[:]  # user: 0,1,2,3,4

    raw = group.csi_data_raw
    # print("csida loading is complete")

    with zipfile.ZipFile(
        outzippath, "w", compression=zipfile.ZIP_DEFLATED
    ) as zip_out_f:

        for i, x in tqdm(enumerate(raw), total=len(raw)):
            helper.write_array2zip(f"raw/{i}", x, zip_out_f)
        helper.write_array2zip("env_label", room, zip_out_f)
        helper.write_array2zip("user_label", user, zip_out_f)
        helper.write_array2zip("act_label", gesture, zip_out_f)
        helper.write_array2zip("loc_label", location, zip_out_f)
