
import torch
import wandb
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score
from utils.utils import flatten_tensor_dicts


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

        self.log(f'{step}_loss_epoch_end', torch.stack(outputs['loss']))
        self.log(f'{step}_accuracy_epoch_end', accuracy(torch.stack(outputs['preds']), torch.stack(outputs['labels']),
                 task='multiclass', num_classes=self.num_classes))
        self.log(f'{step}_f1_macro_epoch_end', f1_score(
            torch.stack(outputs['preds']), torch.stack(outputs['labels']), task='multiclass', num_classes=self.num_classes, average='macro'))

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

        return {'loss': loss, 'preds': preds, 'labels': labels}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate)
