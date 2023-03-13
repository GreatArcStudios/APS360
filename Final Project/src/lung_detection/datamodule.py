import pathlib

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils import data


class HDF5SeqDataset(data.Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        self.length = len(h5py.File(self.data_path, "r")["sequences"])
        self.seqs = None
        self.tgts = None

    def _open_hdf5(self):
        data_file = h5py.File(self.data_path, "r")
        self.seqs = data_file["sequences"]
        self.tgts = data_file["targets"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.seqs is None:
            self._open_hdf5()
        seq = np.eye(4)[self.seqs[idx]].astype(np.float32)
        tgt = self.tgts[idx].astype(np.float32)
        return seq, tgt


class LungDetectionDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=2, num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = pathlib.Path(__file__).parents[2] / "data"
        self.train = HDF5SeqDataset(data_dir / "train.hdf5")
        self.valid = HDF5SeqDataset(data_dir / "valid.hdf5")
        self.test = HDF5SeqDataset(data_dir / "test.hdf5")

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
