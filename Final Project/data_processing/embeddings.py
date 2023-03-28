import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  #for gradient descent
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from collections import Counter

#getting balanced dataset function
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import torchvision.models as models
import shutil

p = './data/images'
create_embeddings = False

#tensor transformation
initial_tranforms = [
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
]
datatransform = transforms.Compose(initial_tranforms)

dataset = torchvision.datasets.ImageFolder(p, transform=datatransform)

print(len(dataset))

#*
#MUST SET SEED TO 26
torch.manual_seed(26)
tr, val, te = torch.utils.data.random_split(
    dataset,
    [30325, 3790, 3790],
)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

#normalizing validataion and testing, but not normalizing training yet since it will be rotated etc first
non_train_transforms_list = [
    transforms.Normalize(mean=mean, std=std)
]
non_train_transforms = transforms.Compose(non_train_transforms_list)
val.transform = non_train_transforms
te.transform = non_train_transforms

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

train = OversampledSubset(tr)
print(f"Dataset Length {len(train.subset.dataset.targets)}, Dataset distribution {Counter(train.subset.dataset.targets)}")
print(f"Original indices size {len(train.indices)},  Original targets size {len(train.targets)}")
print(f"Original training set size: {len(tr)}, Original training set distribution: {Counter(train.targets)}")
print(f"Training set size: {len(train)}, Training set distribution: {Counter(train.over_sampled_targets)}")
print(f"Train indices size {len(train.over_sampled_indices)},  Train targets size {len(train.over_sampled_targets)}")

print(train.subset.dataset.targets[train.indices[0]], train.targets[0])
print(train[0][0].shape)

if create_embeddings:
    #transformation to do a rotation and flip and also normalize the training dataset
    #EVEN IF YOU DON"T WANT TO ROTATE OR FLIP MAKE SURE TO AT LEAST RUN THE NORMALIZE
    train_transforms_list = [
        transforms.RandomRotation((-15, 15)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean, std=std)
    ]
    train_transform = transforms.Compose(train_transforms_list)

    train.transform = train_transform

    print(train.transform)

    trloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    valloader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(te, batch_size=1, shuffle=True)

    #make sure to load the model.pth.tar checkpoint to use
    checkpoint = torch.load('./model.pth.tar', map_location=torch.device('cuda:0'))

    #loading the dictionary of the checkpoint and loading the densent model
    model = models.densenet121(pretrained=True).cuda()
    model_dict = model.state_dict()
    saved_state_dict = checkpoint['state_dict']

    # Modify the keys in the saved state dict to match the keys in your model
    newdict = {}
    for key, value in saved_state_dict.items():
        new_key = key.replace('densenet121.', '')
        new_key = new_key.replace('norm.', 'norm')
        new_key = new_key.replace('conv.', 'conv')
        new_key = new_key.replace('normr', 'norm.r')
        new_key = new_key.replace('normb', 'norm.b')
        new_key = new_key.replace('normw', 'norm.w')
        new_key = new_key.replace('convw', 'conv.w')
        newdict[new_key] = value

    #ignoring the model checkpoint's classifiers
    model_dict = model.state_dict()
    checkpoint_dict = {k: v for k, v in newdict.items() if k in model_dict}
    model_dict.update(checkpoint_dict)

    #loading in the model dictionary
    model.load_state_dict(model_dict)

    #* sanity check should get [1,3,224,224] and [1,1024,7,7]
    for imgs, labels in iter(valloader):
        print(imgs.size())
        f = model.features(imgs.cuda())
        print(f.size())
        break

    #creating directories
    folders = ['embeddingtrain', 'embeddingval', 'embeddingtest']
    directory = "./embeddings/"
    classes = list(dataset.class_to_idx.keys())

    #you need a separate folder for train, val and test with all 15 folders in each
    for i in range(0, len(folders)):
        d = directory + folders[i]
        #initial folder
        if not os.path.exists(d):
            os.makedirs(d)

        #class folders
        for i in range(0, len(classes)):
            fullpath = d + '/' + classes[i]

            if not os.path.exists(fullpath):
                os.makedirs(fullpath)

    #ensure this is run
    classes = list(dataset.class_to_idx.keys())
    folders = ['embeddingtrain', 'embeddingval', 'embeddingtest']


    #creating class folders

    def get_features(loader, classes, folder):
        with torch.no_grad():
            model.eval()
            path = './embeddings/' + folder
            n = 0
            class_dict = {}
            for _, batch in enumerate(loader):
                imgs, labels = batch
                imgs = imgs.cuda()
                label = classes[int(labels[0])]

                # add label to class dict
                if label not in class_dict:
                    class_dict[label] = 1
                else:
                    class_dict[label] += 1

                features = model.features(imgs)
                features_tensor = torch.from_numpy(features.cpu().numpy())
                torch.save(features_tensor.squeeze(0),
                        path + '/' + label + '/' + str(n) + '.tensor')
                n += 1
                if n % 1000 == 0:
                    print(n, class_dict)


    #get_features(trloader, classes, folders[0])
    get_features(valloader, classes, folders[1])
    #get_features(testloader, classes, folders[2])
