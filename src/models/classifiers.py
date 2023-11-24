
import os
import torch
import kornia
import torchvision
from src.models.feature_extractors import get_feature_extractor_imagenet
from src.models.helpers import get_output_dim, ModuleWithFlatten
from src.utils.utils import to_best_available_device
from lightly.data import collate
from lightly.transforms.swav_transform import SwaVViewTransform
from lightly.transforms.simclr_transform import SimCLRViewTransform
from lightly.transforms.dino_transform import DINOViewTransform
from src.models.simclr import SimCLR
from src.models.swav import SwaV
from src.models.dino import DINO
from src.utils.hydra import get_original_cwd_safe
from src.models.build_data_path import get_model_path

class Classifier(torch.nn.Module):
    def __init__(self, backbone, backbone_output_dim, freeze_backbone, num_classes):
        super(Classifier, self).__init__()

        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.num_classes = num_classes
        # self.fc = torch.nn.Linear(backbone_output_dim, self.num_classes, bias=False)
        self.fc = torch.nn.Linear(backbone_output_dim, self.num_classes, bias=True)
            
    def forward(self, x):
        return self.fc(self.backbone(x))

def get_classifier_imagenet(cfg, num_classes) -> tuple:
    backbone, preprocess = get_feature_extractor_imagenet(cfg.training.architecture)

    output_dim = get_output_dim(backbone, preprocess)

    classifier = Classifier(
        backbone=backbone, 
        backbone_output_dim=output_dim, 
        freeze_backbone=cfg.training.weights.freeze, 
        num_classes=num_classes
    )

    return classifier, preprocess

def get_classifier_from_simclr(preprocess, cfg, num_classes: int):
    backbone = SimCLR.load_from_checkpoint(
        os.path.join(get_original_cwd_safe(), cfg.training.weights.path),
        cfg=cfg
    ).backbone
    backbone = ModuleWithFlatten(backbone)
    backbone = to_best_available_device(backbone)

    output_dim = get_output_dim(backbone, preprocess)

    classifier = Classifier(
        backbone=backbone, 
        backbone_output_dim=output_dim, 
        freeze_backbone=cfg.training.weights.freeze, 
        num_classes=num_classes
    )

    return classifier


def get_ssl_model_class(cfg):
    ssl_model_class = None
    if cfg.training.weights.type == 'simclr':
        ssl_model_class = SimCLR
    elif cfg.training.weights.type == 'swav':
        ssl_model_class = SwaV
    elif cfg.training.weights.type == 'dino':
        ssl_model_class = DINO
    else:
        raise ValueError(f'ssl \'{cfg.training.weights.type}\' is not implemented')

    return ssl_model_class


def get_classifier_from_ssl(cfg, feature_dim: int, num_classes: int):
    ssl_model_class = get_ssl_model_class(cfg)

    ssl_model = ssl_model_class.load_from_checkpoint(
        get_model_path(cfg, absolute=True),
        cfg=cfg
    )
    backbone = ssl_model.backbone
    backbone = ModuleWithFlatten(backbone)
    backbone = to_best_available_device(backbone)

    # TODO: based on backbone
    output_dim = feature_dim

    classifier = Classifier(
        backbone=backbone, 
        backbone_output_dim=output_dim, 
        freeze_backbone=cfg.training.weights.freeze, 
        num_classes=num_classes
    )

    return classifier

def get_classifier_imagenet_preprocess_only(cfg):
    _, preprocess = get_feature_extractor_imagenet(cfg.training.architecture)

    return preprocess

def get_ssl_preprocess(cfg):
    assert cfg.training.weights.type in ('simclr', 'swav', 'dino')

    input_size = cfg.dataset.input_size

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size, antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])

    return preprocess

def get_ssl_transform(cfg):
    weights_type = cfg.training.weights.type
    weights_version = cfg.training.weights.version

    transform = None
    if (weights_type, weights_version) == ('swav', 'v1'):
        if cfg.dataset.name in ('isic', 'cifar10'):
            transform = SwaVViewTransform(
                sigmas=get_sigmas(cfg.dataset.input_size)
            ).transform
        else: # matek, retinopathy, jukrat
            transform = SwaVViewTransform(
                sigmas=get_sigmas(cfg.dataset.input_size),
                cj_strength=0.2,
            ).transform
    elif (weights_type, weights_version) == ('simclr', 'v1'):
        transform = SimCLRViewTransform(
            input_size=cfg.dataset.input_size,
            min_scale=0.25,
            sigmas=get_sigmas(cfg.dataset.input_size)
        ).transform
    elif (weights_type, weights_version) == ('dino', 'v1'):
        if cfg.dataset.name == 'cifar10':
            transform = DINOViewTransform(
                crop_size=cfg.dataset.input_size,
                gaussian_blur=0.0,
                solarization_prob=0.0
            ).transform
        else:
            transform = DINOViewTransform(
                crop_size=cfg.dataset.input_size,
                sigmas=get_sigmas(cfg.dataset.input_size),
                solarization_prob=0.0
            ).transform
    elif (weights_type, weights_version) == ('dino', 'v2'):
        if cfg.dataset.name == 'matek':
            transform = DINOViewTransform(
                crop_size=cfg.dataset.input_size,
                sigmas=get_sigmas(cfg.dataset.input_size),
                solarization_prob=0.0,
                cj_strength=0.3,
            )
        elif cfg.dataset.name == 'isic':
            transform = DINOViewTransform(
                crop_size=cfg.dataset.input_size,
                sigmas=get_sigmas(cfg.dataset.input_size),
            )
        elif cfg.dataset.name in ('jurkat', 'cifar10'):
            transform = DINOViewTransform(
                crop_size=cfg.dataset.input_size,
                gaussian_blur=0,
            )
        elif cfg.dataset.name == 'retinopathy':
            transform = DINOViewTransform(
                crop_size=cfg.dataset.input_size,
                sigmas=get_sigmas(cfg.dataset.input_size),
            )
        else:
            raise NotImplemented()
    else:
        raise ValueError(f'ssl transform for given {weights_type}/{weights_version} model not implemented')
    
    return transform

def get_classifier_simclr_preprocess_only(cfg):
    input_size = cfg.dataset.input_size

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size, antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])

    return preprocess
    
def get_sigmas(input_size):
    old_sigmas = (0.1, 2)
    coef = input_size / 224

    return (old_sigmas[0] * coef, old_sigmas[1] * coef)
