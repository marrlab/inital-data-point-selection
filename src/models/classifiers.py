
import os
import torch
import torchvision
from src.models.feature_extractors import get_feature_extractor_imagenet
from src.models.helpers import get_output_dim
from lightly.data import collate
from src.models.lightning_modules import SimCLRModel
from hydra.utils import get_original_cwd

def get_classifier_imagenet(architecture: str, num_classes: int) -> tuple:
    model, preprocess = get_feature_extractor_imagenet(architecture)

    output_dim = get_output_dim(model, preprocess)

    # defining the classifier with the appended linear layer
    class Classifier(torch.nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()

            self.model = model
            self.num_classes = num_classes
            self.fc = torch.nn.Linear(output_dim, self.num_classes, bias=False)
            
        def forward(self, x):
            return self.fc(self.model(x))

    return Classifier(), preprocess

def get_classifier_from_simclr(preprocess, cfg, num_classes: int):
    class ModuleWithFlatten(torch.nn.Module):
        def __init__(self, backbone):
            super(ModuleWithFlatten, self).__init__()

            self.backbone = backbone

        def forward(self, x):
            return self.backbone(x).flatten(start_dim=1)

    backbone = SimCLRModel.load_from_checkpoint(
        os.path.join(get_original_cwd(), cfg.training.weights.path),
        cfg=cfg
    ).backbone
    model = ModuleWithFlatten(backbone)
    if torch.cuda.is_available():
        model.to('cuda')

    output_dim = get_output_dim(model, preprocess)

    # defining the classifier with the appended linear layer
    class Classifier(torch.nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()

            self.model = model
            if cfg.training.weights.freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.num_classes = num_classes
            self.fc = torch.nn.Linear(output_dim, self.num_classes, bias=False)
            
        def forward(self, x):
            return self.fc(self.model(x))

    return Classifier()


def get_classifier_imagenet_preprocess_only(architecture: str):
    _, preprocess = get_feature_extractor_imagenet(architecture)

    return preprocess

def get_classifier_simclr_preprocess_only(input_size: int):
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])

    return preprocess
    