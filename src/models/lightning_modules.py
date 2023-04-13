
import torch
import torch.nn as nn
import numpy as np
import torchvision
import lightning.pytorch as pl
from torchmetrics.functional import accuracy, f1_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from src.utils.utils import flatten_tensor_dicts, map_tensor_values
from collections import defaultdict
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss


class ImageClassifierLightningModule(pl.LightningModule):
    def __init__(self, model, num_labels, num_classes, label_to_class_mapping, class_to_label_mapping, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.label_to_class_mapping = label_to_class_mapping
        self.class_to_label_mapping = class_to_label_mapping
        self.loss = torch.nn.CrossEntropyLoss()

        self.step_outputs = {
            'train': [],
            'val': []
        }
        self.metrics_epoch_end = defaultdict(list)

        self.save_hyperparameters()

    def forward(self, image):
        output = self.model(image)
        return output

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def on_train_epoch_end(self):
        self._common_epoch_end('train')

    def on_validation_epoch_end(self):
        self._common_epoch_end('val')

    def _common_epoch_end(self, step):
        outputs = self.step_outputs[step]
        outputs = flatten_tensor_dicts(outputs)

        if step == 'train':
            self.metrics_epoch_end[f'{step}_loss_epoch_end'].append(
                torch.mean(outputs['loss']).detach().item())

        self.metrics_epoch_end[f'{step}_accuracy_epoch_end'].append(accuracy(
            outputs['pred_labels'], outputs['labels'], task='multiclass', num_classes=self.num_labels).detach().item())
        self.metrics_epoch_end[f'{step}_f1_macro_epoch_end'].append(f1_score(
            outputs['pred_labels'], outputs['labels'], task='multiclass', num_classes=self.num_labels, average='macro').detach().item())
        self.metrics_epoch_end[f'{step}_f1_micro_epoch_end'].append(f1_score(
            outputs['pred_labels'], outputs['labels'], task='multiclass', num_classes=self.num_labels, average='micro').detach().item())
        self.metrics_epoch_end[f'{step}_balanced_accuracy_epoch_end'].append(
            balanced_accuracy_score(outputs['labels'].cpu().numpy(), outputs['pred_labels'].cpu().numpy()))
        self.metrics_epoch_end[f'{step}_matthews_corrcoef_epoch_end'].append(
            matthews_corrcoef(outputs['labels'].cpu().numpy(), outputs['pred_labels'].cpu().numpy()))
        self.metrics_epoch_end[f'{step}_cohen_kappa_score_epoch_end'].append(
            cohen_kappa_score(outputs['labels'].cpu().numpy(), outputs['pred_labels'].cpu().numpy()))

        for key in self.metrics_epoch_end:
            # conversion needed due to float64 (double precision) not being supported in MPS
            values = np.array(self.metrics_epoch_end[key]).astype(np.float32)

            self.log(key, values[-1])
            self.log(f'{key}_max', np.max(values))
            self.log(f'{key}_min', np.min(values))

        self.step_outputs[step].clear()

    def _common_step(self, batch, batch_idx, step):
        assert step in ('train', 'val')

        images, classes, labels = batch['image'], batch['class'], batch['label']

        assert images.ndim == 4

        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        logits = self(images)
        pred_classes = torch.argmax(logits, dim=1)

        loss = None
        if step == 'train':
            loss = self.loss(logits, classes)

        # self.log(f'{step}_loss', loss)
        # self.log(f'{step}_accuracy', accuracy(preds, labels,
        #          task='multiclass', num_classes=self.num_classes))
        # self.log(f'{step}_f1_macro', f1_score(
        #     preds, labels, task='multiclass', num_classes=self.num_classes, average='macro'))
        # self.log(f'{step}_f1_micro', f1_score(
        #     preds, labels, task='multiclass', num_classes=self.num_classes, average='micro'))

        output = {
            'pred_classes': pred_classes,
            'pred_labels': map_tensor_values(pred_classes, self.class_to_label_mapping),
            'classes': classes,
            'labels': labels
        }
        if loss is not None:
            output['loss'] = loss
            
        self.step_outputs[step].append(output)

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.training.learning_rate)


class SimCLRModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        assert cfg.training.architecture in ('resnet18', 'resnet34')
        assert cfg.training.weights.type in ('imagenet', 'simclr', None)

        # create a ResNet backbone and remove the classification head
        resnet = None
        if cfg.training.architecture == 'resnet18':
            if cfg.training.weights.type == 'imagenet':
                resnet = torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = torchvision.models.resnet18()
        elif cfg.training.architecture == 'resnet34':
            if cfg.training.weights.type == 'imagenet':
                resnet = torchvision.models.resnet34(
                    weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                resnet = torchvision.models.resnet34()

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
        return self._common_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, mode='val')

    def _common_step(self, batch, mode='train'):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        self.log(f'{mode}_loss_ssl', loss)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            # TODO: connect learning rate to the config (might break the checkpointing)
            self.parameters(), lr=self.cfg.training.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.cfg.training.epochs
        )

        return [optim], [scheduler]
