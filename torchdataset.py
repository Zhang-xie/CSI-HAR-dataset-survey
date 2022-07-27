from pathlib import Path
import zipfile
import numpy as np
from tqdm import tqdm
import torch

from torch.utils.data import Dataset, DataLoader


class Base_Discrete_Dataset(Dataset):
    def __init__(self, zip_path):
        self.zip_f = zipfile.ZipFile(zip_path, "r")
        self.act_label = np.load(self.zip_f.open("act_label"))

    def __getitem__(self, idx):
        return {
            "act_label": self.act_label[idx],
            "raw": np.load(self.zip_f.open(f"raw/{idx}")),
        }

    def __len__(self):
        return len(self.act_label)


class Base_Continuous_Dataset(Dataset):
    def __init__(self, zip_path):

        self.zip_f = zipfile.ZipFile(zip_path, "r")
        self.act_label = np.load(self.zip_f.open("act_label"))
        self.raw = np.load(self.zip_f.open("raw"))

    def __getitem__(self, idx):
        return {
            "act_label": self.act_label[idx],
            "raw": self.raw[idx],
        }

    def __len__(self):
        return len(self.act_label)


class Wih2h_Dataset(Base_Discrete_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        self.user_label = np.load(self.zip_f.open("user_label"))

    def __getitem__(self, idx):
        _r = super().__getitem__(idx)
        _r["user_label"] = self.user_label[idx]
        return _r


class WiAR_Dataset(Base_Discrete_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        self.user_label = np.load(self.zip_f.open("user_label"))

    def __getitem__(self, idx):
        _r = super().__getitem__(idx)
        _r["user_label"] = self.user_label[idx]
        return _r


class toifall_Dataset(Base_Continuous_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)


class Signfi_Dataset(Base_Continuous_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        self.user_label = np.load(self.zip_f.open("user_label"))
        self.env_label = np.load(self.zip_f.open("env_label"))

    def __getitem__(self, idx):
        _r = super().__getitem__(idx)
        _r["user_label"] = self.user_label[idx]
        _r["env_label"] = self.env_label[idx]
        return _r


class DeepSeg_Dataset(Base_Discrete_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)


class CSLOS_Dataset(Base_Discrete_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        self.user_label = np.load(self.zip_f.open("user_label"))
        self.env_label = np.load(self.zip_f.open("env_label"))
        self.experi_label = np.load(self.zip_f.open("experi_label"))

    def __getitem__(self, idx):
        _r = super().__getitem__(idx)
        _r["user_label"] = self.user_label[idx]
        _r["env_label"] = self.env_label[idx]
        _r["experi_label"] = self.experi_label[idx]
        return _r


class BehaviourCSI_Dataset(Base_Continuous_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)


class ARIL_Dataset(Base_Continuous_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        self.env_label = np.load(self.zip_f.open("env_label"))

    def __getitem__(self, idx):
        _r = super().__getitem__(idx)
        _r["env_label"] = self.env_label[idx]
        return _r


class Har_Dataset(Base_Continuous_Dataset):
    def __init__(self, zip_path):
        super().__init__(zip_path)
        self.env_label = np.load(self.zip_f.open("env_label"))

    def __getitem__(self, idx):
        _r = super().__getitem__(idx)
        _r["env_label"] = self.env_label[idx]
        return _r


def test_all_dataset():
    testinfo = [
        (Wih2h_Dataset, "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/Wih2h_data.zip"),
        (WiAR_Dataset, "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/WiAR_data.zip"),
        (
            toifall_Dataset,
            "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/toifall_data.zip",
        ),
        (Signfi_Dataset, "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/SignFi_data.zip"),
        (
            DeepSeg_Dataset,
            "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/DeepSeg_data.zip",
        ),
        (CSLOS_Dataset, "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/CSLOS_data.zip"),
        (
            BehaviourCSI_Dataset,
            "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/BehaviorCSI_data.zip",
        ),
        (ARIL_Dataset, "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/ARIL_data.zip"),
        (Har_Dataset, "/media/yk/Samsung_T5/CSI-HAR-Datasets-Code/Har_data.zip"),
    ]

    for t in testinfo:
        dataset = t[0](t[1])
        for j in tqdm(dataset, desc=f"Test {Path(t[1]).stem}"):
            # for j in dataset:
            pass
