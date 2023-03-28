import torch
import wandb
import os
import csv
import numpy as np
import pandas as pd
from models import XRayLightning
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

master_path = r'C:\Users\ericz\Documents\Github\APS360\Final Project'

run = wandb.init(project="lung-xray", entity="ericzhu",)
artifact = run.use_artifact(
    'ericzhu/lung-xray/model-q41lvdyh:v7', type='model')
artifact_dir = artifact.download()
model_checkpoint = os.path.join(artifact_dir, "model.ckpt")

# initialize model and load checkpoint
model = XRayLightning.load_from_checkpoint(model_checkpoint)
model.freeze()

# load test predictions CSV 
test_predictions = pd.read_csv(os.path.join(master_path, "lung_detection", "test_predictions_final.csv"))

# filter for correct predictions
correct_predictions = test_predictions[test_predictions["is_correct"] == True]

print(correct_predictions.shape)

# select 10 random correct predictions
random_correct_predictions = correct_predictions.sample(n=10)

# run the model on the 10 random correct predictions
img_paths = random_correct_predictions["img_path"].tolist()
img_paths = [os.path.join(master_path, "data_processing", img_path) for img_path in img_paths]


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# run images through tranforms and save transformed images to "experiment_results" folder
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)
])

# don't run the model on the images, just save the transformed images
# do not undo the normalization
# file names should be "experiment_results/img_path_validation.png"
for img_path in img_paths:
    img = Image.open(img_path)
    img = transform(img)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = np.clip(img, 0, 1)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(os.path.join(master_path, "lung_detection", "experiment_results", img_path.split("\\")[-1], "validation.png"))

# further transform the images to be fed into the model
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.AugMix(severity=6, mixture_width=6),
    transforms.TrivialAugmentWide(),
    transforms.Normalize(mean=mean, std=std),
])

# save the transformed images to "experiment_results" folder, do not undo the normalization
# file names should be "experiment_results/img_path_train.png"
for img_path in img_paths:
    img = Image.open(img_path)
    img = transform(img)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = np.clip(img, 0, 1)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(os.path.join(master_path, "lung_detection", "experiment_results", img_path.split("\\")[-1], "train.png"))

# # run the model on the images and extract the activations
# # save the activations to "experiment_results" folder
# # file names should be "experiment_results/img_path_activations.png"
# for img_path in img_paths:
#     img = Image.open(img_path)
#     img = transform(img)
#     img = img.unsqueeze(0)
#     activations = model(img)
#     activations = activations[0]
#     activations = activations.permute(1, 2, 0)
#     activations = activations.numpy()
#     activations = np.clip(activations, 0, 1)
#     activations = Image.fromarray((activations * 255).astype(np.uint8))
#     activations.save(os.path.join(master_path, "lung_detection", "experiment_results", img_path.split("\\")[-1], "activations.png"))