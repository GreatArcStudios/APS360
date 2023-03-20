import torch
import numpy as np
from torchvision.datasets import DatasetFolder
from torch.utils import data
from lightning import LightningDataModule

class LungDetectionDataModule(LightningDataModule):

    def __init__(self, batch_size=2, num_workers=0, master_path=""):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_dir = master_path
        self.train = DatasetFolder(master_path + 'embeddingtrain2', 
                                   loader=torch.load, extensions=('.tensor'))
        self.valid = DatasetFolder(master_path + 'embeddingval2', 
                                   loader=torch.load, extensions=('.tensor'))
        self.test = DatasetFolder(master_path + 'embeddingval2', 
                                  loader=torch.load, extensions=('.tensor'))

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
