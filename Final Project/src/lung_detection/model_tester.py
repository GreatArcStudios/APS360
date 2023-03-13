import argparse

import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor, ContactPredictor


def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser = LitContactPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    # wandb checkpoint loading
    run = wandb.init()
    artifact = run.use_artifact('uoft-project-x/train_akita/model-22gl5iqa:v39', type='model')
    artifact_dir = artifact.download()
    lit_model = LitContactPredictor(
        augment_shift=args.augment_shift,
        augment_rc=args.augment_rc,
        lr=args.lr,
        variational=0
    )
    test_model = lit_model.load_from_checkpoint(artifact_dir + "\\model.ckpt")
    test_model.variational = 1

    # run the model on the test set
    trainer = pl.Trainer(
        deterministic=True,
        gpus=(1 if torch.cuda.is_available() else 0),
        gradient_clip_val=10,
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    trainer.test(test_model, datamodule=datamodule)


if __name__ == "__main__":
    test_main()
