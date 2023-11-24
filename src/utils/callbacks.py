
import torch
import lightning.pytorch as pl
import numpy as np
from lightly.utils.debug import std_of_l2_normalized

class SSLModeCollapseCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_outputs = {
            'train': [],
            'val': []
        }

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx):
        self._common_batch_end(outputs, 'train')

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx):
        self._common_batch_end(outputs, 'val')

    def _common_batch_end(
            self, outputs, mode='train'):
        self.epoch_outputs[mode].append(outputs['outputs'])

    def on_train_epoch_end(self, trainer, pl_module):
        self._common_epoch_end(mode='train')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self._common_epoch_end(mode='val')

    def _common_epoch_end(self, mode='train'):
        self.epoch_outputs[mode] = torch.cat(self.epoch_outputs[mode], 0)
        val = std_of_l2_normalized(self.epoch_outputs[mode]).cpu().item()
        val_relative = val * np.sqrt(self.epoch_outputs[mode].shape[1])

        self.log(f'{mode}_std_of_l2_normalized', val)
        self.log(f'{mode}_std_of_l2_normalized_relative', val_relative)

        self.epoch_outputs[mode] = []
