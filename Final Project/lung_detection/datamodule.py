import torch
import os
import zipfile
import numpy as np
import pandas as pd 
from torchvision.datasets import DatasetFolder
from torch.utils import data
from lightning import LightningDataModule
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from PIL import Image

class OversampledSubset(Dataset):
    def __init__(self, subset, transform=None):
        # we want to map from the indices of the original subset
        # to the indices of the over-sampled subset, which are really just pointers to objects in the original subset

        # the original subset
        self.subset = subset
        # the transform to apply to the images
        self.transform = transform
        # the indices of the original subset
        self.indices = subset.indices
        # the targets of the original subset
        # needs to be the targets given by self.indices
        self.targets = [subset.dataset.targets[i] for i in self.indices]
        
        ros = RandomOverSampler(sampling_strategy='auto', random_state=26)
        self.over_sampled_indices, self.over_sampled_targets = ros.fit_resample(np.array(self.indices).reshape(-1, 1), self.targets)

    def __getitem__(self, index):
        original_index = self.over_sampled_indices[index][0]
        image, label = self.subset.dataset[original_index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.over_sampled_indices)

class ImageDataset(Dataset):
    def __init__(self, dir, labels, transform=None):
        self.dir = dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.labels.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = torch.Tensor(np.array(self.labels.iloc[index, 3:]).astype(int)).float()
        return image, label

class LungDetectionDataModule(LightningDataModule):

    def __init__(self, batch_size=2, num_workers=0, master_path=""):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_dir = master_path
        # self.train = DatasetFolder(master_path + 'embeddingtrain' + self.reduced_path, 
        #                            loader=torch.load, extensions=('.tensor'))
        # self.valid = DatasetFolder(master_path + 'embeddingval' + self.reduced_path, 
        #                            loader=torch.load, extensions=('.tensor'))
        # self.test = DatasetFolder(master_path + 'embeddingval' + self.reduced_path, 
        #                           loader=torch.load, extensions=('.tensor'))

        initial_tranforms = [
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        datatransform = transforms.Compose(initial_tranforms)
        dataset = self._create_dataset(master_path, datatransform)
        train_len, val_len, test_len = int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)
        torch.manual_seed(26)
        train, valid, test = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
        )

        self.train = train
        self.valid = valid
        self.test = test

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        non_train_transforms_list = [
            transforms.Normalize(mean=mean, std=std)
        ]
        non_train_transforms = transforms.Compose(non_train_transforms_list)
        self.valid.transform = non_train_transforms
        self.test.transform = non_train_transforms

        # self.train = OversampledSubset(self.train)

        train_transforms_list = [
            transforms.RandomRotation((-15, 15)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std)
        ]
        train_transform = transforms.Compose(train_transforms_list)

        self.train.transform = train_transform
    
    def _create_dataset(self, path, transforms = None): 
        dataset_df = pd.read_csv(path + "/csv_mappings/nih_full.csv")
        dataset_df = dataset_df.iloc[:,:16]
        dataset_df[['Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
       'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
       'Pneumothorax']] = 0

        labels = dataset_df["Finding Labels"]
        for i in range(0,len(dataset_df)):
            string = labels[i]
            lst = string.split("|")
            for j in range(0,len(lst)):
                dataset_df.loc[i,lst[j]] = 1
            if i%1000 == 0:
                print(i)

        dataset_df = dataset_df.iloc[:,:16]

        with zipfile.ZipFile(f'{path}/data/nih_data.zip', 'r') as zip:
            # Get the list of file names in the zip file
            names = zip.namelist()
            
        # Print the list of file names
        names = names[6:]
        names = names[:-2]
        dataset_df['Paths'] = names
        dataset_df = dataset_df.reindex(columns=['Paths','Image Index', 'Finding Labels', 'Atelectasis', 'Cardiomegaly',
            'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
            'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
            'Pneumothorax'])
        data_path = os.path.join(path, "data", "nih_data")
        return ImageDataset(data_path, dataset_df, transforms)

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
