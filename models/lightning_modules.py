
import torch
import torch.nn as nn
import wandb
import numpy as np
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from utils.utils import flatten_tensor_dicts
from collections import defaultdict
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class ImageClassifierLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, labels_text=None, **kwargs):
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()
        self.labels_text = labels_text
        if self.labels_text is None:
            self.labels_text = list(
                f'label {i}' for i in range(self.num_classes))

        self.metrics_epoch_end = defaultdict(list)

        self.save_hyperparameters()

    def forward(self, image):
        output = self.model(image)
        return output

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def training_epoch_end(self, outputs):
        self._common_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self._common_epoch_end(outputs, 'val')

    def _common_epoch_end(self, outputs, step):
        outputs = flatten_tensor_dicts(outputs)

        self.metrics_epoch_end[f'{step}_loss_epoch_end'].append(
            torch.mean(outputs['loss']).detach().item())
        self.metrics_epoch_end[f'{step}_accuracy_epoch_end'].append(accuracy(
            outputs['preds'], outputs['labels'], task='multiclass', num_classes=self.num_classes).detach().item())
        self.metrics_epoch_end[f'{step}_f1_macro_epoch_end'].append(f1_score(
            outputs['preds'], outputs['labels'], task='multiclass', num_classes=self.num_classes, average='macro').detach().item())
        self.metrics_epoch_end[f'{step}_f1_micro_epoch_end'].append(f1_score(
            outputs['preds'], outputs['labels'], task='multiclass', num_classes=self.num_classes, average='micro').detach().item())
        self.metrics_epoch_end[f'{step}_balanced_accuracy'].append(
            balanced_accuracy_score(outputs['labels'].cpu().numpy(), outputs['preds'].cpu().numpy()))
        self.metrics_epoch_end[f'{step}_matthews_corrcoef'].append(
            matthews_corrcoef(outputs['labels'].cpu().numpy(), outputs['preds'].cpu().numpy()))
        self.metrics_epoch_end[f'{step}_cohen_kappa_score'].append(
            cohen_kappa_score(outputs['labels'].cpu().numpy(), outputs['preds'].cpu().numpy()))

        for key in self.metrics_epoch_end:
            self.log(key, self.metrics_epoch_end[key][-1])
            self.log(f'{key}_max', np.max(self.metrics_epoch_end[key]))
            self.log(f'{key}_min', np.min(self.metrics_epoch_end[key]))

    def _common_step(self, batch, batch_idx, step):
        assert step in ('train', 'val')

        images, labels = batch['image'], batch['label']

        assert images.ndim == 4

        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss(logits, labels)

        self.log(f'{step}_loss', loss)
        self.log(f'{step}_accuracy', accuracy(preds, labels,
                 task='multiclass', num_classes=self.num_classes))
        self.log(f'{step}_f1_macro', f1_score(
            preds, labels, task='multiclass', num_classes=self.num_classes, average='macro'))
        self.log(f'{step}_f1_micro', f1_score(
            preds, labels, task='multiclass', num_classes=self.num_classes, average='micro'))

        return {'loss': loss, 'preds': preds, 'labels': labels}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate)


class SimCLRModel(pl.LightningModule):
    def __init__(self, max_epochs=1, imagenet_weights=True):
        super().__init__()

        # assinging passed attributes to the object
        self.max_epochs = max_epochs

        # TODO: make configurable
        # create a ResNet backbone and remove the classification head
        resnet = None
        if imagenet_weights:
            resnet = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = torchvision.models.resnet18()

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(
            hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)

        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss_ssl', loss)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )

        return [optim], [scheduler]
