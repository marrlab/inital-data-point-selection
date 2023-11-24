
import csv
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from src.models.helpers import Identity
from torchvision import transforms
from ctypes import ArgumentError
from src.models.med_al_ssl_surgery.simclr_arch import SimCLRArch
from src.utils.utils import to_best_available_device

# source: https://pytorch.org/hub/facebookresearch_semi-supervised-ImageNet1K-models_resnext/
def get_feature_extractor_imagenet(architecture: str) -> tuple:
    model = None
    preprocess = None

    if architecture.startswith('resnet') or architecture.startswith('resnext'):
        assert architecture in (
        # SEMI-WEAKLY SUPERVISED MODELS PRETRAINED WITH 940 HASHTAGGED PUBLIC CONTENT #
            'resnet18_swsl',
            'resnet50_swsl',
            'resnext50_32x4d_swsl',
            'resnext101_32x4d_swsl',
    
            'resnext101_32x8d_swsl',
            'resnext101_32x16d_swsl',
        # SEMI-SUPERVISED MODELS PRETRAINED WITH YFCC100M #
            'resnet18_ssl',
            'resnet50_ssl',
            'resnext50_32x4d_ssl',
            'resnext101_32x4d_ssl',
            'resnext101_32x8d_ssl',
            'resnext101_32x16d_ssl'
        )

        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', architecture)
        model.fc = Identity()

        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif architecture.startswith('vgg'):
        assert architecture in (
        # VGG #
            'vgg11',
            'vgg11_bn',
            'vgg13',
            'vgg13_bn',
            'vgg16',
            'vgg16_bn',
            'vgg19',
            'vgg19_bn',
        )

        model = torch.hub.load('pytorch/vision:v0.10.0', architecture, pretrained=True)
        model.classifier[4] = Identity()
        model.classifier[5] = Identity()
        model.classifier[6] = Identity()

        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif architecture == 'efficientnet':
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1

        model = torchvision.models.efficientnet_b0(weights=weights)
        model.classifier = Identity()

        preprocess = weights.transforms()
    elif architecture == 'vit':
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1

        model = torchvision.models.vit_b_16(weights=weights)
        model.heads = Identity()

        preprocess = weights.transforms()
    elif architecture == 'inception_v3':
        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1

        model = torchvision.models.inception_v3(weights=weights)
        model.fc = Identity()

        preprocess = weights.transforms()


    if model is None or preprocess is None:
        raise ArgumentError('model or preprocessing pipeline could not be loaded')

    model = to_best_available_device(model)
    model.eval()

    return model, preprocess
