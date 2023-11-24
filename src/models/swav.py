
import torch
import torch.nn as nn
import lightning.pytorch as pl
from src.models.helpers import get_backbone
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


class SwaV(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.backbone, out_dim = get_backbone(
            architecture=cfg.training.architecture, flatten=False)

        hidden_dim = out_dim
        self.projection_head = SwaVProjectionHead(hidden_dim, hidden_dim, 128)
        # self.prototypes = SwaVPrototypes(128, n_prototypes=64)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)
        
        self.criterion = SwaVLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)

        return self.prototypes(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, mode='val')

    def _common_step(self, batch, mode='train'):
        self.prototypes.normalize()
        crops, _, _ = batch

        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)

        self.log(f'{mode}_loss_ssl', loss)

        return loss

    def configure_optimizers(self):
        # optim = torch.optim.Adam(self.parameters(), lr=self.cfg.training.learning_rate)
        # return optim

        optim = torch.optim.SGD(
            self.parameters(), lr=self.cfg.training.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.cfg.training.epochs
        )

        return [optim], [scheduler]

