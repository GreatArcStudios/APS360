import torch
import os
import zipfile
import numpy as np
import pandas as pd 
from torchvision.datasets import DatasetFolder
from torch.utils import data
from lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class ImageDataset(Dataset):
    def __init__(self, dir, transform=None, load_first=True, testing=True, new_data=False):
        self.base_dir = dir
        self.nih_dir = dir + r"\data\nih_data"
        self.pneumonia_dir = dir + r"\data\pneumonia"
        self.pneumothorax_dir = dir + r"\data\pneumothorax"
        self.transform = transform
        self.load_first = load_first
        self.testing = testing
        self.new_data = new_data

        self.df = self._inital_processing(dir)
        
        if new_data:
            self.labels = torch.FloatTensor(self.df.iloc[:, 1:].values.astype(int)).float()
        else:
            self.labels = torch.FloatTensor(np.array(self.df.iloc[:, 4:]).astype(int)).float()

        if load_first:
            self.images = self._load_all_images_parallel()

        print(self.nih_dir)
        # print(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.load_first:
            # obtain image from self.images
            image = self.images[index]
            img_path = self.base_dir + fr"\{self.df.iloc[index, 0]}"
        else:
            if not self.new_data:
                if self.df.iloc[index,0] == 1:
                    img_path = self.nih_dir + fr"\{self.df.iloc[index, 1]}"

                #For Pneumothorax Images
                elif self.df.iloc[index,0] == 2:
                    img_path = self.pneumothorax_dir + fr"\{self.df.iloc[index, 1]}"

                #For Pneumonia Images
                elif self.df.iloc[index,0] == 3:
                    img_path = self.pneumonia_dir + fr"\{self.df.iloc[index, 1]}"
            else:
                img_path = os.path.join(self.base_dir, self.df.iloc[index, 0])
                print(img_path)
            # open then close the image to avoid too many open files error
            with Image.open(img_path) as image:
                image.load()

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        if self.testing:
            return image, label, index, img_path
        return image, label
    
    def _inital_processing(self, dir):
        file_name = "Allimages_onehot.csv" if not self.new_data else "newdata_onehot.csv"
        dataset_df = pd.read_csv(os.path.join(dir, "csv_mappings", file_name))

        if not self.new_data:
            # subset only nih data for now by column name
            # where only num column == 1
            # dataset_df = dataset_df[dataset_df['Num'] == 1]

            # undersample the no findings examples
            # they are the ones with all 0's as labels
            # randomly select 5000 of them
            no_findings = dataset_df[dataset_df.iloc[:,4:].sum(axis=1) == 0]
            no_findings = no_findings.sample(n=30000, random_state=26)
            dataset_df = dataset_df[dataset_df.iloc[:,4:].sum(axis=1) != 0]
            dataset_df = pd.concat([dataset_df, no_findings])

        return dataset_df

    def _load_all_images(self):
        images = []
        for index in range(len(self.df)):
            img_path = self.dir + self.df.iloc[index, 0]
            # open then close the image to avoid too many open files error
            with open(img_path, 'rb') as image:
                image = Image.open(image)
                image.load()
                images.append(image.convert('RGB'))
        return images
    
    def _load_image(self, start_index, end_index):
        images = []
        for index in range(start_index, end_index):
            if not self.new_data:
                if self.df.iloc[index,0] == 1:
                    img_path = self.nih_dir + fr"\{self.df.iloc[index, 1]}"
                elif self.df.iloc[index,0] == 2:
                    img_path = self.pneumothorax_dir + fr"\{self.df.iloc[index, 1]}"
                elif self.df.iloc[index,0] == 3:
                    img_path = self.pneumonia_dir + fr"\{self.df.iloc[index, 1]}"
            else: 
                print("File path:", self.df.iloc[index, 0])
                img_path = self.base_dir +  fr"\{self.df.iloc[index, 0]}"
            print(img_path)
            # open then close the image to avoid too many open files error
            with Image.open(img_path) as image:
                image.load()
                images.append(image)
        return images

    def _load_all_images_parallel(self):
        # load the images in parallel
        # maintain the order of the images
        # return a list of images
        images = []
        with ThreadPoolExecutor() as executor:
            # partition job into chunks
            chunk_size = 1000
            num_chunks = len(self.labels) // chunk_size
            for i in range(num_chunks):
                start_index = i * chunk_size
                end_index = start_index + chunk_size
                images += executor.submit(self._load_image, start_index, end_index).result()
            # process the remaining images
            start_index = num_chunks * chunk_size
            end_index = len(self.labels)
            images += executor.submit(self._load_image, start_index, end_index).result()

        return images

class LungDetectionDataModule(LightningDataModule):

    def __init__(self, batch_size=2, num_workers=0, master_path="", load_first = True, testing = False, new_data = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_dir = master_path

        initial_tranforms = [
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
        datatransform = transforms.Compose(initial_tranforms)
        dataset = self._create_dataset(master_path, datatransform, load_first, testing, new_data)
        train_len, val_len = int(len(dataset)*0.8), int(len(dataset)*0.1)
        test_len = len(dataset) - train_len - val_len
        train, valid, test = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
        )
        print("train: ", len(train), "valid: ", len(valid), "test: ", len(test))
        self.train = train
        self.valid = valid
        self.test = test

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        non_train_transforms_list = [
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(severity=6, mixture_width=6),
            transforms.TrivialAugmentWide(),
            transforms.Normalize(mean=mean, std=std),
        ]
        non_train_transforms = transforms.Compose(non_train_transforms_list)
        self.valid.transform = non_train_transforms
        self.test.transform = non_train_transforms

        train_transforms_list = [
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(severity=6, mixture_width=6),
            transforms.TrivialAugmentWide(),
            transforms.Normalize(mean=mean, std=std),
        ]
        train_transform = transforms.Compose(train_transforms_list)

        self.train.transform = train_transform
    
    def _create_dataset(self, path, transforms = None, load_first = True, testing = False, new_data = False): 
        print('new_data: ', new_data)
        return ImageDataset(path, transforms, load_first, testing, new_data)
    
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
