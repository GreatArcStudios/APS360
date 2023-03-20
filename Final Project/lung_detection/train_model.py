import wandb
import torch
import torchvision
import numpy as np
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
               master_path="", **kwargs):
    # seed experiment
    seed_everything(seed=123)

    # construct datamodule
    datamodule = LungDetectionDataModule(batch_size=batch_size,
                                         num_workers=num_workers,
                                         master_path=master_path)
    data_size = len(datamodule.train)

    # construct model
    lit_model = XRayLightning(seed=123,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              data_size=data_size,
                              **kwargs)

    # logging
    logger = WandbLogger(project="lung-xray", entity="ericzhu",
                         log_model="all", save_dir="./wandb_saves")
    logger.experiment.config["train_set_len"] = len(datamodule.train)
    logger.experiment.config["val_set_len"] = len(datamodule.valid)
    logger.experiment.config["batch_size"] = batch_size

    # callbacks
    early_stopping = EarlyStopping(
        monitor="val_f1_score", mode="max", patience=30)
    checkpointing = ModelCheckpoint(
        monitor="val_f1_score", mode="max", save_top_k=5)
    stochastic_weighting = StochasticWeightAveraging(swa_epoch_start=0.75,
                                                     annealing_epochs=5,
                                                     swa_lrs=3e-4)
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

    trainer.fit(lit_model, datamodule=datamodule)

    wandb.finish()

    return lit_model


def compute_class_accuracies(use_cuda=True, model=None, train_features=None):
    nb_classes = 15

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
    torch.set_float32_matmul_precision("high")
    master_path = '../data_processing/embeddings/'
    train = False
    if train:
        train_configs = {
            "master_path": master_path,
            "batch_size": 64,
            "num_workers": 0,
            "max_epochs": 50,
            "lr": 0.00008,
            "weight_decay": 1e-4,
            "momentum": 0.99,
        }
        trunk_configs = {
            "trunk_input_channels": 1024,
            "trunk_mid_channels": 512,
            "trunk_out_channels": 128,
            "trunk_kernel_size": 7,
            "trunk_transpose_kernel": 12,
            "trunk_dropout": 0.1,
            "trunk_conv_layers": 2,
        }
        head_configs = {
            "head_n_layer": 3,
            "head_n_head": 8,
            "head_feature_map_dim": 10,
            "head_input_channels": 128,
            "head_mid_channels": 256,
            "head_output_channels": 64,
            "head_kernel_size": 2,
            "head_max_pool_kernel_size": 2,
            "head_conv_layers": 1,
            "head_classifier_input_features": 1024,
            "head_hidden_size": 2048,
            "head_dropout": 0.2,
        }
        # combine configs into train_configs
        train_configs.update(trunk_configs)
        train_configs.update(head_configs)
        model = train_main(**train_configs)
    else:
        train_configs = {
            "master_path": master_path,
            "batch_size": 64,
            "num_workers": 0,
            "max_epochs": 50,
            "lr": 0.00008,
            "weight_decay": 1e-4,
            "momentum": 0.99,
        }
        trunk_configs = {
            "trunk_input_channels": 1024,
            "trunk_mid_channels": 512,
            "trunk_out_channels": 128,
            "trunk_kernel_size": 7,
            "trunk_transpose_kernel": 12,
            "trunk_dropout": 0.1,
            "trunk_conv_layers": 2,
        }
        head_configs = {
            "head_n_layer": 3,
            "head_n_head": 8,
            "head_feature_map_dim": 10,
            "head_input_channels": 128,
            "head_mid_channels": 256,
            "head_output_channels": 64,
            "head_kernel_size": 2,
            "head_max_pool_kernel_size": 2,
            "head_conv_layers": 1,
            "head_classifier_input_features": 1024,
            "head_hidden_size": 2048,
            "head_dropout": 0.2,
        }
        # combine configs into train_configs
        train_configs.update(trunk_configs)
        train_configs.update(head_configs)
        eval_model = XRayLightning(**train_configs)

        run = wandb.init(project="lung-xray", entity="ericzhu",)
        artifact = run.use_artifact(
            'ericzhu/lung-xray/model-nmy83p9y:v4', type='model')
        artifact_dir = artifact.download()
        model = eval_model.load_from_checkpoint(
            artifact_dir + '/model.ckpt')
        model.freeze()
        model.eval()
        model.to("cuda:0")
        
        val_features = torchvision.datasets.DatasetFolder(
            master_path + 'embeddingval2', loader=torch.load, extensions=('.tensor'))
        train_accuracy_per_class = compute_class_accuracies(
            use_cuda=True, model=model, train_features=val_features)
        np.savetxt("baseline_results.csv", train_accuracy_per_class, delimiter=",")

        val_features = torchvision.datasets.DatasetFolder(
            master_path + 'embeddingval2', loader=torch.load, extensions=('.tensor'))
        train_accuracy_per_class = compute_class_accuracies(
            use_cuda=True, model=model, train_features=val_features)
        np.savetxt("baseline_results.csv", train_accuracy_per_class, delimiter=",")