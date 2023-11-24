
import torch
import torch.nn as nn
import lightning.pytorch as pl
from src.models.classifiers import get_ssl_model_class, Classifier
from src.models.build_data_path import get_model_path, get_scan_path
from src.models.helpers import get_backbone, ModuleWithFlatten
from src.utils.utils import to_best_available_device
import torch.nn.functional as F

class SCAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        assert cfg.training.weights.type == 'simclr', \
            'other ssl methods not yet supported'

        self.cfg = cfg

        self.backbone, out_dim = get_backbone(
            architecture=cfg.training.architecture, flatten=True)

        self.num_clusters = cfg.scan.num_clusters
        self.dropout = torch.nn.Dropout(p=cfg.scan.dropout)
        # self.fc = torch.nn.Linear(out_dim, self.num_clusters, bias=False)
        self.fc = torch.nn.Linear(out_dim, self.num_clusters, bias=True)
        
        self.criterion = SCANLoss(entropy_weight=cfg.scan.entropy_weight)

    def backbone_from_ssl(self):
        ssl_model_class = get_ssl_model_class(self.cfg)
        self.backbone = ssl_model_class(cfg=self.cfg)
        self.backbone = ssl_model_class.load_from_checkpoint(
            get_model_path(self.cfg, absolute=True),
            cfg=self.cfg
        ).backbone
        self.backbone = ModuleWithFlatten(self.backbone)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, mode='val')

    def _common_step(self, batch, mode='train'):
        anchor, neighbor = batch['anchor_image'], batch['neighbor_image']
            
        anchor_output = self.forward(anchor)
        neighbor_output = self.forward(neighbor)

        total_loss, consistency_loss, entropy_loss = self.criterion(
            anchor_output, neighbor_output)

        self.log(f'{mode}_scan_loss', total_loss)
        self.log(f'{mode}_consistency_loss', consistency_loss)
        self.log(f'{mode}_entropy_loss', entropy_loss)

        return total_loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=5e-4)
        return optim


class SCANLoss(torch.nn.Module):
    def __init__(self, entropy_weight=5.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 5.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss

EPS = 1e-8
def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))
    
def get_classifier_from_scan(cfg, num_classes: int):
    scan_model = SCAN.load_from_checkpoint(
        get_scan_path(cfg, absolute=True),
        cfg=cfg
    )
    backbone = scan_model.backbone
    backbone = to_best_available_device(backbone)

    output_dim = scan_model.fc.in_features

    classifier = Classifier(
        backbone=backbone, 
        backbone_output_dim=output_dim, 
        freeze_backbone=cfg.training.weights.freeze, 
        num_classes=num_classes
    )

    return classifier
