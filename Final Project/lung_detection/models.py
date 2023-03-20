import torch
import torch.cuda
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary
from self_attention_cv import AxialAttentionBlock
from layers import Conv2dBlock, TransposeConv2dBlock, DilatedResConv2dBlock, LinearBlock
from lightning import LightningModule

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None:
            assert len(self.alpha) == inputs.size(1), "Number of weights in alpha should match the number of classes"
            class_weights = torch.tensor(self.alpha, device=inputs.device, dtype=torch.float32)
        else:
            class_weights = None

        ce_loss = F.cross_entropy(inputs, targets, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassifierHead(nn.Module):

    def __init__(self, head_n_layer=3, head_n_head=8, head_feature_map_dim=31,
                 head_input_channels=96, head_mid_channels=24, head_output_channels=12,
                 head_kernel_size=4, head_max_pool_kernel_size=2, head_conv_layers=2,
                 head_dropout=0.01, head_classifier_input_features = 2048, head_hidden_size=1024, **kwargs):
        super().__init__()

        axial_attn_block = AxialAttentionBlock(
            head_input_channels, head_feature_map_dim, head_n_head)

        # build up the modules
        self.attention_layers = nn.Sequential(
            *[axial_attn_block for _ in range(head_n_layer)])
        modules = []
        for i in range(head_conv_layers):
            modules += [Conv2dBlock(head_input_channels,
                                    head_mid_channels, kernel_size=head_kernel_size)] # add back leaky relu here
            modules += [nn.BatchNorm2d(head_mid_channels, momentum=0.1)]
            modules += [nn.Mish()]
            modules += [Conv2dBlock(head_mid_channels,
                                    head_input_channels, kernel_size=head_kernel_size)]
            modules += [nn.BatchNorm2d(head_input_channels, momentum=0.1)]
            modules += [nn.MaxPool2d(kernel_size=head_max_pool_kernel_size)]
            modules += [nn.Mish()]

        modules += [Conv2dBlock(head_input_channels,
                                head_output_channels, kernel_size=1)]
        # unpack modules
        self.head = nn.Sequential(*modules)
        self.classifier_fc = nn.Sequential(
            LinearBlock(head_classifier_input_features,head_hidden_size, dropout=head_dropout),
            LinearBlock(head_hidden_size,head_hidden_size//2, dropout=head_dropout),
            LinearBlock(head_hidden_size//2,head_hidden_size//4, dropout=head_dropout),
            LinearBlock(head_hidden_size//4 ,head_hidden_size//8, dropout=head_dropout),
            LinearBlock(head_hidden_size//8,head_hidden_size//16, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//16, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//8, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//4, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//2, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//4, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//8, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//16, dropout=head_dropout),
            # LazyLinearBlock(head_hidden_size//16, dropout=head_dropout),
            # uncomment to use larger network
            # LinearBlock(head_hidden_size//16,head_hidden_size//32),
            # LinearBlock(head_hidden_size//32,head_hidden_size//64),
            # nn.Linear(head_hidden_size//64,15),
            # comment out to use larger network
            nn.Linear(head_hidden_size//16,15),
        )

    def forward(self, trunk_output):
        # z = self.attention_layers(trunk_output) + trunk_output
        z = self.attention_layers(trunk_output) 
        z = self.head(z)
        z = z.view(z.size(0), -1)
        return self.classifier_fc(z)


class ClassifierTrunk(nn.Module):

    def __init__(self, trunk_input_channels=1024, trunk_mid_channels=64,
                 trunk_out_channels=96, trunk_kernel_size=3, trunk_conv_layers=2,
                 trunk_transpose_kernel=12, trunk_dropout=0.04, **kwargs):
        super().__init__()

        # build up the modules
        modules = [Conv2dBlock(
            trunk_input_channels, trunk_out_channels, 
            kernel_size=1)] # add back leaky relu here
        dilation = 1.0
        for _ in range(trunk_conv_layers):
            layer_dilation = round(dilation)
            modules += [TransposeConv2dBlock(trunk_out_channels,
                                             trunk_out_channels, trunk_transpose_kernel, trunk_dropout)]
            modules += [nn.MaxPool2d(kernel_size=2)]
            modules += [DilatedResConv2dBlock(
                trunk_out_channels, trunk_mid_channels, trunk_out_channels, trunk_kernel_size, layer_dilation, trunk_dropout)]
            dilation *= 1.75

        # unpack modules
        self.trunk = nn.Sequential(*modules)

    def forward(self, input_embeds):
        return self.trunk(input_embeds)


class XRayPredictor(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.trunk = ClassifierTrunk(**kwargs)
        self.head = ClassifierHead(**kwargs)

    def forward(self, input_embeds):
        z = self.trunk(input_embeds)
        return self.head(z)


class XRayLightning(LightningModule):

    def __init__(self, lr, momentum, weight_decay, gamma = 2, alpha = None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = XRayPredictor(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.criterion = FocalLoss(alpha, gamma)
        self.disease_mapping = {
            0: "No Finding",
            1: "Atelectasis",
            2: "Cardiomegaly",
            3: "Consolidation",
            4: "Edema",
            5: "Effusion",
            6: "Emphysema",
            7: "Fibrosis",
            8: "Hernia",
            9: "Infiltration",
            10: "Mass",
            11: "Nodule",
            12: "Pleural_Thickening",
            13: "Pneumonia",
            14: "Pneumothorax",
        }

    def forward(self, chexnet_embeds):
        return self.model(chexnet_embeds)

    def training_step(self, batch, batch_idx):
        loss, batch_size, train_accuracy, f1_score, f1_score_avg, weighted_f1, auroc = self._process_batch(
            batch, True)
        self.log('train_loss', loss, batch_size=batch_size)
        self.log('train_accuracy', train_accuracy, batch_size=batch_size)
        self.log('train_f1_score', f1_score_avg, batch_size=batch_size)
        self.log('train_weighted_f1', weighted_f1, batch_size=batch_size)
        self._log_f1_score(f1_score, "train", batch_size)
        self._log_auroc(auroc, "train", batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_size, val_accuracy, f1_score, f1_score_avg, weighted_f1, auroc = self._process_batch(
            batch, True)
        self.log('val_loss', loss, batch_size=batch_size)
        self.log('val_accuracy', val_accuracy, batch_size=batch_size)
        self.log('val_f1_score', f1_score_avg, batch_size=batch_size)
        self.log('val_weighted_f1', weighted_f1, batch_size=batch_size)
        self._log_f1_score(f1_score, "val", batch_size)
        self._log_auroc(auroc, "val", batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        loss, batch_size, test_accuracy, f1_score, f1_score_avg, weighted_f1, auroc = self._process_batch(
            batch, True)
        self.log('test_loss', loss, batch_size=batch_size)
        self.log('test_accuracy', test_accuracy, batch_size=batch_size)
        self.log('test_f1_score', f1_score_avg, batch_size=batch_size)
        self.log('test_weighted_f1', weighted_f1, batch_size=batch_size)
        self._log_f1_score(f1_score, "test", batch_size)
        self._log_auroc(auroc, "test", batch_size)

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.lr, 
        #     weight_decay=self.wd, momentum=self.momentum, nesterov=True)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=4, eta_min=0.00002)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def _process_batch(self, batch, compute_accuracy=False):
        chexnet_embeds, labels = batch
        logits = self(chexnet_embeds)
        loss = self.criterion(logits, labels)

        if compute_accuracy:
            accuracy_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=15).cuda()
            accuracy = accuracy_metric(logits, labels)
            # pytorch f1 score
            f1_score_metric = torchmetrics.F1Score(task="multiclass",
                                                   num_classes=15, average='none').cuda()
            f1_score = f1_score_metric(logits, labels)
            f1_score_avg = f1_score.mean()

            weighted_f1_score_metric = torchmetrics.F1Score(task="multiclass",
                                                            num_classes=15, average='weighted').cuda()
            weighted_f1_score = weighted_f1_score_metric(logits, labels)

            # pytorch auroc scores per class
            auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=15,
                                       average="none").cuda()
            auroc = auroc_metric(logits, labels)
            return loss, len(labels), accuracy, f1_score, f1_score_avg, weighted_f1_score, auroc

        return loss, len(labels)

    def _log_f1_score(self, f1_score, step_type, batch_size):
        for i in range(len(f1_score)):
            self.log(f"{step_type}_f1_score_class_{self.disease_mapping[i]}", f1_score[i], batch_size=batch_size)

    def _log_auroc(self, auroc, step_type, batch_size):
        
        for i in range(len(auroc)):
            self.log(f"{step_type}_auroc_class_{self.disease_mapping[i]}", auroc[i], batch_size=batch_size)
