import math

import torch
import torch.nn as nn


class AverageTo2D(nn.Module):

    def forward(self, z):
        b, w, d = z.shape
        z_2d = z.tile([1, w, 1])
        z_2d = z_2d.reshape(b, w, w, d)
        return (z_2d + z_2d.transpose(1, 2)) / 2


class ConcatDist2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist_matrix = torch.zeros(1, 1, 1, 1)

    def _cache_dist_matrix(self, width):
        if self.dist_matrix.shape[1] == width:
            return
        dists = torch.tensor([[[[abs(i - j)] for i in range(width)] for j in range(width)]])
        self.dist_matrix = dists.float() / (width - 1)

    def forward(self, z_2d):
        self._cache_dist_matrix(z_2d.shape[1])
        d = self.dist_matrix.tile([z_2d.shape[0], 1, 1, 1]).to(z_2d.device)
        return torch.cat([z_2d, d], dim=-1)


class Symmetrize2D(nn.Module):

    def forward(self, x):
        return (x + x.transpose(2, 3)) / 2


class Conv1dBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size,
            pool_size=1, activation=True
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2  # padding needed to maintain size

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels, momentum=0.0735)
        ]

        if pool_size > 1:
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
        if activation:
            layers.append(nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Conv2dBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size,
            dilation=1, dropout=0.0, symmetrize=False, activation=True
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # padding needed to maintain size

        # noinspection PyTypeChecker
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels, momentum=0.0735)
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        if symmetrize:
            layers.append(Symmetrize2D())
        if activation:
            layers.append(nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DilatedResConv2dBlock(nn.Module):

    def __init__(
            self, in_channels, mid_channels, out_channels, kernel_size,
            dilation=1, dropout=0.0, symmetrize=False
    ):
        super().__init__()
        self.symmetrize = symmetrize

        self.blocks = nn.Sequential(
            Conv2dBlock(in_channels, mid_channels, kernel_size, dilation=dilation),
            Conv2dBlock(mid_channels, out_channels, kernel_size, dropout=dropout, activation=False)
        )

        if symmetrize:
            self.activation = nn.Sequential(Symmetrize2D(), nn.GELU())
        else:
            self.activation = nn.GELU()

    def forward(self, x):
        x = x + self.blocks(x)  # residual connection
        return self.activation(x)


class VariationalLayer(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, z):
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        sample_z = self.reparameterize(mu, logvar)
        return sample_z, mu, logvar


# Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(512).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        pe = torch.zeros(512, 1, n_embd)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):

    def __init__(self, n_embd, n_layer, n_head, n_inner, dropout):
        super(TransformerEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_inner,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.pos_encoding = PositionalEncoding(n_embd, dropout)
        self.encoder = nn.TransformerEncoder(layer, n_layer)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(self.pos_encoding(x))
        return x.transpose(1, 2)


# ==================================================================================================
# Uncomment for later use
# ==================================================================================================
#
#
# class PoswiseFeedForwardNet(nn.Module):
#
#     def __init__(self, input_dim, output_dim, hidden_dim=192):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.activ = nn.GELU()
#
#     def forward(self, enc_inputs):
#         enc_outputs = self.fc1(enc_inputs)
#         enc_outputs = self.fc2(enc_outputs)
#         enc_outputs = self.activ(enc_outputs)
#         return enc_outputs
#
#
# class PoswiseConvNet(nn.Module):
#
#     def __init__(self, input_dim, output_dim, kernel_size=5):
#         super(PoswiseConvNet, self).__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding)
#         self.conv2 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding)
#         self.activ = nn.GELU()
#
#     def forward(self, enc_inputs):
#         enc_outputs = self.conv1(enc_inputs)
#         enc_outputs = self.conv2(enc_outputs)
#         enc_outputs = self.activ(enc_outputs)
#         return enc_outputs
#
#
# ########## Implementation with ConvNet ##############
# class EncoderLayer(nn.Module):
#    def __init__(self, embed_dim=96, num_heads=8):
#        super(EncoderLayer, self).__init__()
#        self.enc_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#        self.pos_conv = PoswiseConvNet(embed_dim, embed_dim)
#
#    def forward(self, enc_inputs):
#        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
#        enc_outputs = torch.transpose(enc_outputs, 1, 2)
#        enc_outputs = self.pos_conv(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
#        enc_outputs = torch.transpose(enc_outputs, 1, 2)
#        return enc_outputs, attn
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, embed_dim=96, num_heads=8):
#         super(EncoderLayer, self).__init__()
#         self.enc_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.pos_ffn = PoswiseFeedForwardNet(embed_dim, embed_dim)
#
#     def forward(self, enc_inputs):
#         enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
#         enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
#         return enc_outputs
