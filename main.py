from rawdata_process import (
    Wih2h,
    Widar3,
    CSLOS,
    BehaviorCSI,
    ARIL,
    DeepSeg,
    Signfi,
    toifall,
    WiAR,
    Har,
    falldefi,
)
import numpy as np
from torchdataset import test_all_dataset


def generate_all_zip():
    # 参数1，输入目录。参数2，输出目录。
    Wih2h.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/Wih2h", "Wih2h_data.zip")

    CSLOS.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/CSLOS", "CSLOS_data.zip")
    BehaviorCSI.raw2zip(
        "/media/yk/Samsung_T5/CSI-HAR-Datasets/BehaviorCSI", "BehaviorCSI_data.zip"
    )
    ARIL.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/ARIL", "ARIL_data.zip")
    DeepSeg.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/DeepSeg", "DeepSeg_data.zip")
    Signfi.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/SignFi", "SignFi_data.zip")
    toifall.raw2zip(
        "/media/yk/Samsung_T5/CSI-HAR-Datasets/toifall_dataset", "toifall_data.zip"
    )
    WiAR.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/WiAR-master/", "WiAR_data.zip")

    Har.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/Har", "Har_data.zip")

    # Widar3.raw2zip(
    #     "/media/yk/Samsung_T5/CSI-HAR-Datasets/widar3.0/", "Widar3_data.zip"
    # )  # 没写好
    # falldefi.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/falldefi")  # 没写好


if __name__ == "__main__":
    # generate_all_zip()  # 重新运行会把已有的文件覆盖了，所以注释了
    Wih2h.raw2zip("/media/yk/Samsung_T5/CSI-HAR-Datasets/Wih2h", "Wih2h_data.zip")
    # test_all_dataset()
    # Widar3.raw2zip(
    #     "/media/yk/Samsung_T5/CSI-HAR-Datasets/widar3.0/", "Widar3_data.zip"
    # )  # 没写好
