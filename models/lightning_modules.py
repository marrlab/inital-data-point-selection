
import torch
import wandb
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, confusion_matrix


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
        images, labels = batch['image'], batch['label']

        assert images.ndim == 4

        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        logits = self(images)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss(logits, labels)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(preds, labels,
                 task='multiclass', num_classes=self.num_classes))
        wandb.log({'train_confusion_matrix': wandb.plot.confusion_matrix(probs=None,
                                                                         y_true=labels.cpu().numpy(), preds=preds.cpu().numpy(),
                                                                         class_names=self.labels_text)})
        # self.log('train_confusion_matrix', confusion_matrix(preds, label_ids, self.num_classes))

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        assert images.ndim == 4

        h, w = images.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        logits = self.forward(images)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss(logits, labels)

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(preds, labels,
                 task='multiclass', num_classes=self.num_classes))
        wandb.log({'val_confusion_matrix': wandb.plot.confusion_matrix(probs=None,
                                                                       y_true=labels.cpu().numpy(), preds=preds.cpu().numpy(),
                                                                       class_names=self.labels_text)})
        # self.log('val_confusion_matrix', confusion_matrix(preds, label_ids, self.num_classes))

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate)
