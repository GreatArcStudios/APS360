import functools
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.akita.layers import (
    AverageTo2D,
    ConcatDist2D,
    Conv1dBlock,
    Conv2dBlock,
    DilatedResConv2dBlock,
    VariationalLayer,
    TransformerEncoder
)


# =============================================================================
# Data Transforms
# =============================================================================


def shift_pad(seqs, shift, pad=0.25):
    assert len(seqs.shape) == 3
    if shift == 0:
        return seqs
    seqs = torch.roll(seqs, shifts=shift, dims=1)
    if shift > 0:
        seqs[:, :shift, :] = pad
    else:
        seqs[:, shift:, :] = pad
    return seqs


def reverse_complement(seqs):
    assert len(seqs.shape) == 3
    # flipping dim=1 reverses
    # flipping dim=2 is a hack to perform complement
    return torch.flip(seqs, dims=[1, 2])


@functools.lru_cache()
def _reverse_triu_index_perm(width, offset):
    assert offset > 0
    triu_idxs = torch.triu_indices(width, width, offset)

    mat = torch.zeros(width, width, dtype=torch.long)
    mat[triu_idxs[0], triu_idxs[1]] = torch.arange(triu_idxs.shape[1])
    mat = mat + mat.T

    rev_mat = torch.flip(mat, dims=[0, 1])
    return rev_mat[triu_idxs[0], triu_idxs[1]]


def reverse_triu(trius, width, offset):
    assert offset > 0
    assert len(trius.shape) == 3

    perm = _reverse_triu_index_perm(width, offset)
    assert len(perm) == trius.shape[1]

    return trius[:, perm, :]


# =============================================================================
# Models
# =============================================================================


class Trunk(nn.Module):

    def __init__(self, n_layer, n_head, n_inner, dropout):
        super().__init__()

        transformer = TransformerEncoder(
            n_embd=96,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            dropout=dropout
        )

        modules = [Conv1dBlock(4, 96, 11, pool_size=2)]
        modules += [Conv1dBlock(96, 96, 5, pool_size=2) for _ in range(10)]
        modules += [transformer, Conv1dBlock(96, 64, 5)]
        self.trunk = nn.Sequential(*modules)

    def forward(self, input_seqs):
        input_seqs = input_seqs.transpose(1, 2)
        return self.trunk(input_seqs).transpose(1, 2)


class HeadHIC(nn.Module):

    def __init__(self):
        super().__init__()
        self.one_to_two = AverageTo2D()
        self.concat_dist = ConcatDist2D()

        modules = [Conv2dBlock(65, 48, 3, symmetrize=True)]
        dilation = 1.0
        for _ in range(6):
            modules.append(DilatedResConv2dBlock(48, 24, 48, 3, round(dilation), 0.1, True))
            dilation *= 1.75
        self.head = nn.Sequential(*modules)

    def forward(self, z):
        z = self.concat_dist(self.one_to_two(z))
        z = torch.permute(z, [0, 3, 1, 2])
        y = self.head(z)
        return torch.permute(y, [0, 2, 3, 1])



class ContactPredictor(nn.Module):

    def __init__(self, n_layer, n_head, n_inner, dropout, variational):
        super().__init__()
        self.variational = variational
        self.trunk = Trunk(n_layer, n_head, n_inner, dropout)
        self.head = HeadHIC()
        self.fc_out = nn.Linear(48, 5)
        self.vae_layer = VariationalLayer(64, 64) if variational else None

        self.target_width = 448
        self.diagonal_offset = 2
        self.triu_idxs = torch.triu_indices(448, 448, 2)

    def forward(self, input_seqs, flatten=False, only_decode=False):
        if only_decode:
            return self.fc_out(self.forward_pass_flatten(self.head(input_seqs), flatten))

        z = self.trunk(input_seqs)
        if self.variational:
            z, mu, logvar = self.vae_layer(z)
        else:
            mu = logvar = None

        y = self.forward_pass_flatten(self.head(z), flatten)

        return self.fc_out(y), mu, logvar

    def forward_pass_flatten(self, y, flatten):
        if flatten:
            y = y[:, 32:-32, 32:-32, :]
            y = y[:, self.triu_idxs[0], self.triu_idxs[1], :]
        return y

# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_layer', type=int, default=3)
        parser.add_argument('--n_head', type=int, default=6)
        parser.add_argument('--n_inner', type=int, default=256)
        parser.add_argument('--dropout', type=float, default=0.2)

        parser.add_argument('--augment_rc', type=int, default=1)
        parser.add_argument('--augment_shift', type=int, default=11)
        parser.add_argument('--lr', type=float, default=0.0025)
        parser.add_argument('--variational', type=int, default=1)
        return parser

    def __init__(
            self,
            n_layer, n_head, n_inner, dropout,
            augment_rc, augment_shift, lr, 
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = ContactPredictor(n_layer, n_head, n_inner, dropout, variational=bool(kwargs['variational']))
        self.augment_rc = augment_rc
        self.augment_shift = augment_shift
        self.lr = lr
        self.variational = bool(kwargs['variational'])

    def forward(self, input_seqs):
        return self.model(input_seqs, flatten=True)

    def training_step(self, batch, batch_idx):
        batch = self._stochastic_augment(batch)
        loss, batch_size = self._process_batch(batch)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self._process_batch(batch)
        self.log('val_loss', loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss, batch_size = self._process_batch(batch, test=True)
        self.log('test_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.975)
        return optimizer

    def _stochastic_augment(self, batch):
        take_rc = torch.rand(size=[]) > 0.5
        shift = random.randrange(-self.augment_shift, self.augment_shift + 1)

        seqs, tgts = batch
        if self.augment_rc and take_rc:
            seqs = reverse_complement(seqs)
            tgts = reverse_triu(tgts, self.model.target_width, self.model.diagonal_offset)
        seqs = shift_pad(seqs, shift, pad=0.25)
        return seqs, tgts

    def _process_batch(self, batch, test=False):
        seqs, tgts = batch
        batch_size = tgts.shape[0]
        preds, mu, logvar = self(seqs)

        if self.variational:
            if not test:
                return self._VAE_loss((preds, mu, logvar), tgts), batch_size
            else:
                preds = []
                ppd_predictions = []
                for sample in range(500):
                    posterior_sample = torch.distributions.normal.Normal(mu, torch.exp(logvar)).sample()
                    ppd_predictions.append(self.model(posterior_sample, flatten=True, only_decode=True))
                averaged_preds = torch.stack(ppd_predictions).mean(dim=0)
                preds = averaged_preds

        loss = F.mse_loss(preds, tgts)
        if not self.variational or test:
            return loss, batch_size

    def _VAE_loss(self, preds, tgts):
        # unpack preds; will be a tuple in the case of VAE
        y_hat, mu_q, log_var_q = preds

        MSELoss_criterion = nn.MSELoss()
        MSE_loss = MSELoss_criterion(y_hat, tgts)

        # Analytically derived loss function for the KL Divergence:
        # We want to calculate -D_KL[q(z|x) || p(z)]
        # KL divergence with a normal prior and normal posterior is given as:
        # log(sigma_q / sigma_p) - (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma^2_p) + 0.5
        # Equiv derivation here: https://jaketae.github.io/study/vae/
        # prior_mean, prior_variance = 0, 1
        #print("maximum and minimum", max(mu_q.flatten()), min(mu_q.flatten()))
        # KLDiv_loss = -torch.sum(0.5 * log_var_q - 0.5 * math.log(prior_variance) -
        #                        ((torch.exp(log_var_q) + (mu_q - prior_mean) ** 2) / (
        #                                    2 * prior_variance))
        #                        + 0.5, dim=(1, 2))
        # KLDiv_loss = torch.mean(KLDiv_loss, dim=0)
        kld_loss = -0.5 * torch.sum(1 + log_var_q - mu_q ** 2 - log_var_q.exp(), dim=(1, 2))
        kld_loss = torch.mean(kld_loss, dim=0)
        return MSE_loss + kld_loss
