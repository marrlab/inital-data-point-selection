
import torch
import torchvision
import torch.nn as nn
import lightning.pytorch as pl
from src.models.helpers import get_backbone
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

class SimCLR(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.backbone, out_dim = get_backbone(
            architecture=cfg.training.architecture, flatten=False)

        hidden_dim = out_dim
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
            self.parameters(), lr=self.cfg.training.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.cfg.training.epochs
        )

        return [optim], [scheduler]
