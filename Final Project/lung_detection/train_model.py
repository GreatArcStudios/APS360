import wandb
import torch
import os
import csv
import numpy as np
import pandas as pd
from models import XRayLightning
from datamodule import LungDetectionDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.trainer import Trainer, seed_everything


def train_main(batch_size=128, num_workers=4, max_epochs=50,
               master_path="", use_inverse_weighting = False,
               pos_weight_multi=1.0, load_first = False, 
               fine_tune_epoch_start=20, checkpoint_path = None, 
               testing = False, new_data= False, **kwargs):
    # seed experiment
    seed_everything(seed=26)

    # construct datamodule
    datamodule = LungDetectionDataModule(batch_size=batch_size,
                                         num_workers=num_workers,
                                         master_path=master_path, 
                                         load_first=load_first, 
                                         testing=testing,
                                         new_data=new_data)
    data_size = len(datamodule.train)
    if use_inverse_weighting:
        targets = [target for _, target in datamodule.train]
        class_counts = np.bincount(targets)

        # Calculate the class frequencies
        class_freqs = class_counts / data_size

        # Calculate the inverse frequency weights
        weights = 1 / class_freqs
    else:
        weights = None

    # ratio of negative examples to positive examples 
    # for bce with logits loss 
    # consider the labels from datamodule.train 
    num_pos_labels_train = torch.sum(datamodule.train.dataset.labels, dim=0)
    num_neg_labels_train = len(datamodule.train) - num_pos_labels_train
    pos_weight_vec = num_neg_labels_train / num_pos_labels_train
    pos_weight_vec = pos_weight_vec * pos_weight_multi

    # construct model
    lit_model = XRayLightning(seed=123,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              data_size=data_size,
                              alpha=weights,
                              pos_weight_vec=pos_weight_vec,
                              fine_tune_epoch_start=fine_tune_epoch_start,
                              **kwargs)

    # logging
    logger = WandbLogger(project="lung-xray", entity="ericzhu",
                         log_model="all", save_dir="./wandb_saves")
    logger.experiment.config["train_set_len"] = len(datamodule.train)
    logger.experiment.config["val_set_len"] = len(datamodule.valid)
    logger.experiment.config["batch_size"] = batch_size

    # callbacks
    early_stopping = EarlyStopping(
        monitor="val_f1_score", mode="max", patience=100)
    checkpointing = ModelCheckpoint(
        monitor="val_f1_score", mode="max", save_top_k=5)
    stochastic_weighting = StochasticWeightAveraging(swa_epoch_start=0.75,
                                                     annealing_epochs=8,
                                                     swa_lrs=0.0003)
    model_sumary = ModelSummary(max_depth=4)
    learning_rate_montior = LearningRateMonitor(logging_interval="step")
    # training
    trainer = Trainer(
        callbacks=[early_stopping, checkpointing,
                   stochastic_weighting, model_sumary,
                   learning_rate_montior],
        devices="auto",
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        precision="bf16-mixed",
    )
    if not testing:
        trainer.fit(lit_model, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        test_results = trainer.test(lit_model, datamodule=datamodule, ckpt_path=checkpoint_path)
        wandb.finish()
        return test_results, lit_model

    wandb.finish()

    return lit_model


def compute_class_accuracies(use_cuda=True, model=None, train_features=None):
    nb_classes = 14

    correct_pred = [0]*nb_classes
    total_pred = [0]*nb_classes

    # data_features = torchvision.datasets.DatasetFolder(master_path, loader=torch.load, extensions=('.tensor'))
    data_loader = torch.utils.data.DataLoader(train_features, batch_size=2048)

    # delete
    n = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            if use_cuda and torch.cuda.is_available():
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")
            n += 1
            out = model(inputs)
            out = torch.nn.functional.softmax(out, dim=1)
            predicted = out.max(1)[1]

            for i in range(len(labels)):
                label = labels[i]
                correct_pred[label] += (predicted[i] == label).item()
                total_pred[label] += 1

    accuracy_per_class = np.array(correct_pred) / np.array(total_pred)

    print("Correct predictions: ", correct_pred)
    print("Total predictions: ", total_pred)
    print("Accuracy score per class: ", accuracy_per_class)
    print("Total Accuracy: ", np.sum(correct_pred) / np.sum(total_pred))
    return accuracy_per_class

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    master_path = r'C:\Users\ericz\Documents\Github\APS360\Final Project\data_processing'
    train = False
    checkpointed = True
    if train:
        train_configs = {
            "master_path": master_path,
            "batch_size": 128,
            "num_workers": 0,
            "max_epochs": 120,
            "lr": 0.00006,
            "weight_decay": 8e-8,
            "momentum": 0.98,
            "gamma": 2, 
            "use_inverse_weighting": False,
            "num_classes": 14,
            "fine_tune": True,
            "fine_tune_epoch_start": 40,
            "pos_weight_multi": 1.3,
            "train_from_scratch": False,
            "load_first": True,
            "checkpoint_path": r"C:\Users\ericz\Documents\GitHub\APS360\Final Project\lung_detection\wandb_saves\lung-xray\49vb85ek\checkpoints\epoch=49-step=27750.ckpt"
        }
        trunk_configs = {
            "trunk_input_channels": 1024,
            "trunk_mid_channels": 128,
            "trunk_out_channels": 64,
            "trunk_kernel_size": 7,
            "trunk_transpose_kernel": 21,
            "trunk_dropout": 0.15,
            "trunk_conv_layers": 2,
        }
        head_configs = {
            "use_vit": True,
            "head_n_layer": 52,
            "head_n_head": 4,
            "head_feature_map_dim": 16,
            "head_input_channels": 64,
            "head_mid_channels": 32,
            "head_output_channels": 16,
            "head_kernel_size": 4,
            "head_max_pool_kernel_size": 2,
            "head_conv_layers": 1,
            "head_classifier_input_features": 512,
            "head_hidden_size": 384,
            "head_dropout": 0.5,
        }
        # combine configs into train_configs
        train_configs.update(trunk_configs)
        train_configs.update(head_configs)
        dense_model = train_main(**train_configs)
    else:
        run = wandb.init(project="lung-xray", entity="ericzhu",)
        artifact = run.use_artifact(
            'ericzhu/lung-xray/model-q41lvdyh:v7', type='model')
        artifact_dir = artifact.download()
        model_checkpoint = os.path.join(artifact_dir, "model.ckpt")
        test_configs = {
            "master_path": master_path,
            "batch_size": 44,
            "num_workers": 0,
            "max_epochs": 120,
            "lr": 0.00006,
            "weight_decay": 8e-8,
            "momentum": 0.98,
            "gamma": 2, 
            "use_inverse_weighting": False,
            "num_classes": 14,
            "fine_tune": True,
            "fine_tune_epoch_start": 40,
            "pos_weight_multi": 1.3,
            "train_from_scratch": False,
            "load_first": True,
            "checkpoint_path": model_checkpoint,
            "testing": not train,
            "new_data": True,
        }
        trunk_configs = {
            "trunk_input_channels": 1024,
            "trunk_mid_channels": 128,
            "trunk_out_channels": 64,
            "trunk_kernel_size": 7,
            "trunk_transpose_kernel": 21,
            "trunk_dropout": 0.15,
            "trunk_conv_layers": 2,
        }
        head_configs = {
            "use_vit": True,
            "head_n_layer": 52,
            "head_n_head": 4,
            "head_feature_map_dim": 16,
            "head_input_channels": 64,
            "head_mid_channels": 32,
            "head_output_channels": 16,
            "head_kernel_size": 4,
            "head_max_pool_kernel_size": 2,
            "head_conv_layers": 1,
            "head_classifier_input_features": 512,
            "head_hidden_size": 384,
            "head_dropout": 0.5,
        }
        # combine configs into train_configs
        test_configs.update(trunk_configs)
        test_configs.update(head_configs)
        test_results, eval_model = train_main(**test_configs)
        print(test_results)

        # save test results to txt file
        with open("test_results_new_data.txt", "w") as f:
            f.write(str(test_results))

        # save the model predictions labels to a csv file
        # obtain the test results list from the eval_model
        test_results = eval_model.test_results

        # convert the list of dictionaries to a dataframe
        test_results_df = pd.DataFrame(test_results)
        # save the dataframe to a csv file
        test_results_df.to_csv("test_predictions_new_data.csv")