import torch
import torch.cuda
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils import data
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchinfo import summary
from layers import Conv2dBlock, TransposeConv2dBlock, DilatedResConv2dBlock, LinearBlock
from lightning import LightningModule
from sklearn.metrics import roc_auc_score
from collections import Counter
from torcheval.metrics.functional import multiclass_f1_score, multiclass_auroc
from vit_pytorch import ViT
from torchvision import models
from adamp import AdamP, SGDP


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', pos_weight_vec = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight_vec = pos_weight_vec
        print("pos_weight_vec:", pos_weight_vec)

    def forward(self, inputs, targets):
        if self.alpha is not None:
            assert len(self.alpha) == inputs.size(
                1
            ), "Number of weights in alpha should match the number of classes"
            class_weights = torch.tensor(self.alpha,
                                         device=inputs.device,
                                         dtype=torch.float32)
        else:
            class_weights = None

        ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                     targets,
                                                     weight=class_weights,
                                                     pos_weight=self.pos_weight_vec,
                                                     reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassifierHead(nn.Module):
    def __init__(self,
                 head_n_layer=3,
                 head_n_head=8,
                 head_feature_map_dim=31,
                 head_input_channels=96,
                 head_mid_channels=24,
                 head_output_channels=12,
                 head_kernel_size=4,
                 head_max_pool_kernel_size=2,
                 head_conv_layers=2,
                 head_dropout=0.01,
                 head_classifier_input_features=2048,
                 head_hidden_size=1024,
                 use_vit=False,
                 num_classes=14,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        print(self.num_classes, use_vit)
        if not use_vit:
            axial_attn_block = AxialAttentionBlock(head_input_channels,
                                                head_feature_map_dim,
                                                head_n_head)

            # build up the modules
            attention_blocks = [axial_attn_block for _ in range(head_n_layer)]
            self.attention_layers = nn.Sequential(*attention_blocks)
            modules = []
            for i in range(head_conv_layers):
                modules += [
                    Conv2dBlock(head_input_channels,
                                head_mid_channels,
                                kernel_size=head_kernel_size)
                ]  # add back leaky relu here
                modules += [nn.BatchNorm2d(head_mid_channels, momentum=0.1)]
                modules += [nn.Mish()]
                modules += [
                    Conv2dBlock(head_mid_channels,
                                head_input_channels,
                                kernel_size=head_kernel_size)
                ]
                modules += [nn.BatchNorm2d(head_input_channels, momentum=0.1)]
                modules += [nn.MaxPool2d(kernel_size=head_max_pool_kernel_size)]
                modules += [nn.Mish()]

            modules += [
                Conv2dBlock(head_input_channels,
                            head_output_channels,
                            kernel_size=1)
            ]
            # unpack modules
            self.head = nn.Sequential(*modules)
            self.classifier_fc = nn.Sequential(
                LinearBlock(head_classifier_input_features,
                            head_hidden_size,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size,
                            head_hidden_size // 2,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size // 2,
                            head_hidden_size // 4,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size // 4,
                            head_hidden_size // 8,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size // 8,
                            head_hidden_size // 16,
                            dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//16, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//8, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//4, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//2, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//4, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//8, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//16, dropout=head_dropout),
                # LazyLinearBlock(head_hidden_size//16, dropout=head_dropout),
                # uncomment to use larger network
                LinearBlock(head_hidden_size // 16, head_hidden_size // 32),
                LinearBlock(head_hidden_size // 32,
                            head_hidden_size // 64,
                            use_activation=False),
                nn.Linear(head_hidden_size // 64, self.num_classes),
                # comment out to use larger network
                # nn.Linear(head_hidden_size//16,15),
            )
        else:
            self.vit = ViT(image_size=head_feature_map_dim,
                           patch_size=4,
                           num_classes=head_hidden_size // 2,
                           dim=head_hidden_size,
                           channels=head_input_channels,
                           depth=head_n_layer,
                           heads=head_n_head,
                           mlp_dim=head_hidden_size,
                           dropout=head_dropout,
                           emb_dropout=0.1)
            self.classifier_fc = nn.Sequential(
                LinearBlock(head_hidden_size // 2,
                            head_hidden_size // 4,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size // 4,
                            head_hidden_size // 8,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size // 8,
                            head_hidden_size // 16,
                            dropout=head_dropout),
                LinearBlock(head_hidden_size // 16,
                            self.num_classes,
                            use_activation=False),
            )

    def forward(self, trunk_output):
        # residual connection
        # z = self.attention_layers(trunk_output) + trunk_output
        if hasattr(self, 'vit'):
            z = self.vit(trunk_output)
            out = self.classifier_fc(z)
        else:
            z = self.attention_layers(trunk_output)
            z = self.head(z)
            z = z.view(z.size(0), -1)
            out = self.classifier_fc(z)
        return out


class ClassifierTrunk(nn.Module):
    def __init__(self,
                 trunk_input_channels=1024,
                 trunk_mid_channels=64,
                 trunk_out_channels=96,
                 trunk_kernel_size=3,
                 trunk_conv_layers=2,
                 trunk_transpose_kernel=12,
                 trunk_dropout=0.04,
                 **kwargs):
        super().__init__()

        # build up the modules
        modules = [
            Conv2dBlock(trunk_input_channels,
                        trunk_out_channels,
                        kernel_size=1)
        ]  # add back leaky relu here
        dilation = 1.0
        for _ in range(trunk_conv_layers):
            layer_dilation = round(dilation)
            modules += [
                TransposeConv2dBlock(trunk_out_channels, trunk_out_channels,
                                     trunk_transpose_kernel, trunk_dropout)
            ]
            modules += [nn.MaxPool2d(kernel_size=2)]
            modules += [
                DilatedResConv2dBlock(trunk_out_channels, trunk_mid_channels,
                                      trunk_out_channels, trunk_kernel_size,
                                      layer_dilation, trunk_dropout)
            ]
            dilation *= 1.75

        # unpack modules
        self.trunk = nn.Sequential(*modules)

    def forward(self, input_embeds):
        return self.trunk(input_embeds)


class XRayPredictor(nn.Module):
    def __init__(self, fine_tune=False, train_from_scratch = False, **kwargs):
        super().__init__()
        self.trunk = ClassifierTrunk(**kwargs)
        self.head = ClassifierHead(**kwargs)

        if not train_from_scratch:
            if fine_tune:
                self.dense_model = self._init_dense_model(train_from_scratch)
                # don't initially allow the dense model to be trained
                for param in self.dense_model.parameters():
                    param.requires_grad = False
            else:
                # create dense net model but freeze the parameters
                self.dense_model = self._init_dense_model(train_from_scratch)
                for param in self.dense_model.parameters():
                    param.requires_grad = False
        else:
            self.dense_model = self._init_dense_model(train_from_scratch)

    def forward(self, input_embeds):
        if hasattr(self, 'dense_model'):
            input_embeds = self.dense_model(input_embeds)
        z = self.trunk(input_embeds)
        return self.head(z)

    def _init_dense_model(self, train_from_scratch):
        if not train_from_scratch:
            checkpoint = torch.load('../data_processing/model.pth.tar',
                                    map_location=torch.device('cuda:0'))
            #loading the dictionary of the checkpoint and loading the densent model
            dense_model = models.densenet121(pretrained=True, drop_rate=0.3).cuda()
            model_dict = dense_model.state_dict()
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
            model_dict = dense_model.state_dict()
            checkpoint_dict = {k: v for k, v in newdict.items() if k in model_dict}
            model_dict.update(checkpoint_dict)
            #loading in the model dictionary
            dense_model.load_state_dict(model_dict)
        else: 
            dense_model = models.densenet121(pretrained=False).cuda()
        return dense_model.features


class XRayLightning(LightningModule):
    def __init__(self,
                 lr,
                 momentum,
                 weight_decay,
                 gamma=2,
                 alpha=None,
                 pos_weight_vec = None,
                 fine_tune_epoch_start= 20, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = XRayPredictor(**kwargs)
        self.lr = lr
        self.num_classes = kwargs['num_classes']
        self.momentum = momentum
        self.wd = weight_decay
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec)
        self.fine_tune_epoch_start = fine_tune_epoch_start
        self.is_fine_tuning = False
        self.disease_mapping = {
            0: "Atelectasis",
            1: "Cardiomegaly",
            2: "Consolidation",
            3: "Edema",
            4: "Effusion",
            5: "Emphysema",
            6: "Fibrosis",
            7: "Hernia",
            8: "Infiltration",
            9: "Mass",
            10: "Nodule",
            11: "Pleural_Thickening",
            12: "Pneumonia",
            13: "Pneumothorax",
        }

        self.test_results = []

    def forward(self, chexnet_embeds):
        return self.model(chexnet_embeds)

    def training_step(self, batch, batch_idx):
        # unfreeze dense net after epoch
        if self.current_epoch >= self.fine_tune_epoch_start-1 and not self.is_fine_tuning:
            print(f"Unfreezing DenseNet after {self.fine_tune_epoch_start} epochs")
            for param in self.model.dense_model.parameters():
                param.requires_grad = True
            self.is_fine_tuning = True

        loss, batch_size, train_accuracy, f1_score, f1_score_avg, weighted_f1, auroc = self._process_batch(
            batch, True)
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True)
        self.log('train_accuracy',
                 train_accuracy,
                 batch_size=batch_size,
                 prog_bar=True)
        self.log('train_f1_score', f1_score_avg, batch_size=batch_size)
        self.log('train_weighted_f1', weighted_f1, batch_size=batch_size)
        self._log_f1_score(f1_score, "train", batch_size)
        self._log_auroc(auroc, "train", batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_size, val_accuracy, f1_score, f1_score_avg, weighted_f1, auroc = self._process_batch(
            batch, True)
        self.log('val_loss',
                 loss,
                 batch_size=batch_size,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log('val_accuracy',
                 val_accuracy,
                 batch_size=batch_size,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log('val_f1_score',
                 f1_score_avg,
                 batch_size=batch_size,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log('val_weighted_f1',
                 weighted_f1,
                 batch_size=batch_size,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self._log_f1_score(f1_score, "val", batch_size)
        self._log_auroc(auroc, "val", batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        loss, batch_size, test_accuracy, f1_score, f1_score_avg, weighted_f1, auroc, logits, labels, index, img_path = self._process_batch(
            batch, True, True)
        self.log('test_loss', loss, batch_size=batch_size, on_epoch=True)
        self.log('test_accuracy', test_accuracy, batch_size=batch_size, on_epoch=True)
        self.log('test_f1_score', f1_score_avg, batch_size=batch_size, on_epoch=True)
        self.log('test_weighted_f1', weighted_f1, batch_size=batch_size, on_epoch=True)
        self._log_f1_score(f1_score, "test", batch_size, on_epoch=True)
        self._log_auroc(auroc, "test", batch_size, on_epoch=True)

        probs = torch.sigmoid(logits).float().cpu().detach().numpy()
        preds = (probs > 0.5).astype(int)
        labels = labels.cpu().detach().numpy()
        is_correct = (preds == labels).all(axis=1)

        self.test_results.append({
            "probs": probs,
            "prediction": preds,
            "label": labels,
            "dataframe_index": index,
            "img_path": img_path,
            "is_correct": is_correct
        })

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.lr,
        #     weight_decay=self.wd, momentum=self.momentum, nesterov=True)
        # optimizer = SGDP(self.parameters(),
        #                 lr=self.lr,
        #                 weight_decay=self.wd,
        #                 momentum=self.momentum,
        #                 nesterov=True)
        
        # optimizer = torch.optim.Adam(self.parameters(),
        #                              lr=self.lr,
        #                              weight_decay=self.wd)
        optimizer = AdamP(self.parameters(), 
                          lr=self.lr, 
                          weight_decay=self.wd,
                          nesterov=True)
        
        cosine_anneal = CosineAnnealingLR(optimizer, T_max=6, eta_min=0.00015)
        reduce_lr = ReduceLROnPlateau(optimizer, min_lr=0.00001)

        return {
            "optimizer": optimizer,
            "lr_scheduler": cosine_anneal,
            "monitor": "val_loss"
        }

    def _process_batch(self, batch, compute_accuracy=False, return_logits = False):
        chexnet_embeds, labels, index, img_path = batch
        logits = self(chexnet_embeds)
        loss = self.criterion(logits, labels)
        if compute_accuracy:
            labels = labels.int()
            num_labels = int(self.num_classes)

            # debug code
            #print(F.sigmoid(logits).shape, labels.shape)
            # convert logit prob to binary
            #bin_preds = torch.round(F.sigmoid(logits)).int()
            #print("Preds: ", bin_preds, "labels:", labels)

            accuracy_metric = torchmetrics.Accuracy(task="multilabel",
                                                    num_labels=num_labels,
                                                    average="macro").cuda()
            accuracy = accuracy_metric(logits, labels)
            # pytorch f1 score
            f1_score_metric = torchmetrics.F1Score(task="multilabel",
                                                   num_labels=num_labels,
                                                   average='none').cuda()
            f1_score = f1_score_metric(logits, labels)
            # f1_score = multiclass_f1_score(logits, labels, average=None, num_classes=self.num_classes)
            f1_score_avg = f1_score.mean()

            weighted_f1_score_metric = torchmetrics.F1Score(
                task="multilabel", num_labels=num_labels,
                average='weighted').cuda()
            weighted_f1_score = weighted_f1_score_metric(logits, labels)

            # pytorch auroc scores per class
            auroc_metric = torchmetrics.AUROC(task="multilabel",
                                              num_labels=num_labels,
                                              average="none").cuda()
            auroc = auroc_metric(logits, labels)
            # auroc = multiclass_auroc(logits, labels, average=None, num_classes=self.num_classes)
            if not return_logits:
                return loss, len(
                    labels
                ), accuracy, f1_score, f1_score_avg, weighted_f1_score, auroc
            else:
                return loss, len(
                    labels
                ), accuracy, f1_score, f1_score_avg, weighted_f1_score, auroc, logits, labels, index, img_path

        return loss, len(labels)

    def _log_f1_score(self, f1_score, step_type, batch_size, on_epoch=False):
        for i in range(len(f1_score)):
            self.log(f"{step_type}_f1_score_class_{self.disease_mapping[i]}",
                     f1_score[i],
                     batch_size=batch_size, on_epoch=on_epoch)

    def _log_auroc(self, auroc, step_type, batch_size, on_epoch=False):
        for i in range(len(auroc)):
            self.log(f"{step_type}_auroc_class_{self.disease_mapping[i]}",
                     auroc[i],
                     batch_size=batch_size, on_epoch=on_epoch)
