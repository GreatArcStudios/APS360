import argparse
import pathlib

import pytorch_lightning as pl
import torch.cuda
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor



def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser = LitContactPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    args.data_size = len(datamodule.train)

    # construct model
    lit_model = LitContactPredictor(**vars(args))

    # TODO: play with training, logging, callback, etc. parameters below
    # TODO: I wasn't sure how to get it to checkpoint in a nice directory

    # logging
    save_dir = pathlib.Path(__file__).parents[2]
    logger = WandbLogger(project="train_akita", log_model="all", save_dir=str(save_dir))
    logger.experiment.config["train_set_len"] = len(datamodule.train)
    logger.experiment.config["val_set_len"] = len(datamodule.valid)
    logger.experiment.config["batch_size"] = args.batch_size

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=40)
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=20)
    stochastic_weighting = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.75, annealing_epochs=5, swa_lrs=4.5e-4)
    lr_monitor = pl.callbacks.LearningRateMonitor("step", True)

    # training
    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpointing, stochastic_weighting, lr_monitor],
        deterministic=True,
        gpus=-1,
        gradient_clip_val=10.7,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        max_epochs=65,
        strategy = DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(lit_model, datamodule=datamodule)

    wandb.finish()

    return lit_model


if __name__ == "__main__":
    train_main()