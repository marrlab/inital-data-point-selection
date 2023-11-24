
import torch
import numpy as np
import lightning.pytorch as pl
from torchmetrics.functional import accuracy, f1_score
from sklearn.metrics import \
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from src.utils.utils import flatten_tensor_dicts, map_tensor_values, map_tensor_probs
from collections import defaultdict
from lightly.utils.scheduler import CosineWarmupScheduler
from torchmetrics.classification import AveragePrecision, AUROC


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
            'val': [],
            'test': [],
        }
        self.metrics_epoch_end = defaultdict(list)

        self.save_hyperparameters()

    def forward(self, image):
        output = self.model(image)
        return output

    # regular step (per batch)
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')

    def _common_step(self, batch, batch_idx, step):
        images, classes, labels = batch['image'], batch['class'], batch['label']

        logits = self(images)
        prob_classes = torch.nn.functional.softmax(logits, dim=1)
        pred_classes = torch.argmax(logits, dim=1)

        loss = None
        if step == 'train':
            loss = self.loss(logits, classes)

        output = {
            'pred_classes': pred_classes,
            'prob_classes': prob_classes,
            'classes': classes,
            'pred_labels': map_tensor_values(pred_classes, self.class_to_label_mapping),
            'prob_labels': map_tensor_probs(prob_classes, self.class_to_label_mapping, new_classes=self.num_labels),
            'labels': labels
        }
        if loss is not None:
            output['loss'] = loss
            
        self.step_outputs[step].append(output)

        return output

    # epoch start
    def on_train_epoch_start(self):
        self._common_epoch_start('train')

    def on_validation_epoch_start(self):
        self._common_epoch_start('val')

    def on_test_epoch_start(self):
        self._common_epoch_start('test')

    def _common_epoch_start(self, step): 
        self.step_outputs[step].clear()

    # epoch end
    def on_train_epoch_end(self):
        self._common_epoch_end('train')

    def on_validation_epoch_end(self):
        self._common_epoch_end('val')

    def on_test_epoch_end(self):
        self._common_epoch_end('test')

    def _common_epoch_end(self, step):
        outputs = self.step_outputs[step]
        outputs = flatten_tensor_dicts(outputs)

        # moving everything to cpu
        for key in outputs:
            outputs[key] = outputs[key].detach().cpu()

        if step == 'train':
            self.metrics_epoch_end[f'{step}_loss_epoch_end'].append(
                torch.mean(outputs['loss']).item())

        self.metrics_epoch_end[f'{step}_accuracy_epoch_end'].append(accuracy(
            outputs['pred_labels'], outputs['labels'], task='multiclass', num_classes=self.num_labels).item())
        self.metrics_epoch_end[f'{step}_f1_macro_epoch_end'].append(f1_score(
            outputs['pred_labels'], outputs['labels'], task='multiclass', num_classes=self.num_labels, average='macro').item())
        self.metrics_epoch_end[f'{step}_f1_micro_epoch_end'].append(f1_score(
            outputs['pred_labels'], outputs['labels'], task='multiclass', num_classes=self.num_labels, average='micro').item())
        self.metrics_epoch_end[f'{step}_balanced_accuracy_epoch_end'].append(
            balanced_accuracy_score(outputs['labels'].numpy(), outputs['pred_labels'].numpy()))
        self.metrics_epoch_end[f'{step}_matthews_corrcoef_epoch_end'].append(
            matthews_corrcoef(outputs['labels'].numpy(), outputs['pred_labels'].numpy()))
        self.metrics_epoch_end[f'{step}_cohen_kappa_score_epoch_end'].append(
            cohen_kappa_score(outputs['labels'].numpy(), outputs['pred_labels'].numpy()))

        # so that we are sure that all the possible classes are contained here
        if step == 'test':
            # roc auc
            roc_auc = AUROC(task='multiclass', num_classes=self.num_labels, average='macro')(
                outputs['prob_labels'], outputs['labels'])
            self.metrics_epoch_end[f'{step}_roc_auc_curve_epoch_end'].append(roc_auc)
        
            # pr auc
            pr_auc = AveragePrecision(task='multiclass', num_classes=self.num_labels, average='macro')(
                outputs['prob_labels'], outputs['labels'])
            self.metrics_epoch_end[f'{step}_pr_auc_curve_epoch_end'].append(pr_auc)

        for key in self.metrics_epoch_end:
            # conversion needed due to float64 (double precision) not being supported in MPS
            values = np.array(self.metrics_epoch_end[key]).astype(np.float32)

            self.log(key, values[-1])
            self.log(f'{key}_max', np.max(values))
            self.log(f'{key}_min', np.min(values))

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.cfg.training.learning_rate)

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.cfg.training.epochs
        )

        return [optimizer], [scheduler]
