
import copy

import lightning.pytorch as pl
import torch
from src.models.helpers import get_backbone

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class DINO(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        backbone, out_dim = get_backbone(
            architecture=cfg.training.architecture, flatten=False)
        input_dim = out_dim

        # student
        self.student_backbone = backbone
        # self.student_head = DINOProjectionHead(
        #     input_dim=input_dim, 
        #     hidden_dim=2048, 
        #     bottleneck_dim=256, 
        #     output_dim=2048,
        #     batch_norm=True
        # )
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048)

        # teacher
        self.backbone = copy.deepcopy(backbone)
        # self.head = DINOProjectionHead(
        #     input_dim=input_dim, 
        #     hidden_dim=2048, 
        #     bottleneck_dim=256, 
        #     output_dim=2048,
        #     batch_norm=True
        # )
        self.head = DINOProjectionHead(
            input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)

        return z

    def forward_teacher(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)

        return z

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, mode='val')

    def _common_step(self, batch, mode='train'):
        momentum = cosine_schedule(self.current_epoch, self.cfg.training.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_head, self.head, m=momentum)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self.log(f'{mode}_loss_ssl', loss)

        return {
            'loss': loss,
            'outputs': teacher_out[0]
        }

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

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
