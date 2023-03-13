import argparse

import pytorch_lightning as pl

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser = LitContactPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    # construct model
    lit_model = LitContactPredictor(True, 11, lr=1e-4, variational=False)

    # training
    trainer = pl.Trainer(
        deterministic=True,
        enable_checkpointing=False,
        enable_model_summary=True,
        enable_progress_bar=True,
        gradient_clip_val=10,
        logger=False,
        log_every_n_steps=1,
        fast_dev_run=True
    )

    trainer.fit(lit_model, datamodule=datamodule)

    return lit_model


if __name__ == "__main__":
    train_main()
