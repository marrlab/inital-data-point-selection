
import csv
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from models.helpers import Identity
from torchvision import transforms
from ctypes import ArgumentError

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
            transforms.Resize(256),
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
            transforms.Resize(256),
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

    if torch.cuda.is_available():
        model.to('cuda')

    model.eval()

    return model, preprocess


def compute_features(model, dataset, path, batch_size=32):
    model.eval()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    batches = int(np.ceil(len(dataset) / batch_size))

    data = {
        'features': [],
        'labels': [],
        'names': [],
    }

    print('started computing features')
    with tqdm(total=batches) as pbar:
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
              output = model(batch['image'])

            data['features'].extend(output.cpu().numpy())
            data['labels'].extend(batch['label'].cpu().numpy())
            data['names'].extend(batch['name'])

            if i == 0:
                print(f"feature vector length: {len(data['features'][0])}")

            pbar.update()

    print('creating csv file with features')
    with open(path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['name', 'label', 'feature'])
        for i in range(len(data['features'])):
            csv_writer.writerow([
                data['names'][i],
                data['labels'][i],
                data['features'][i].tolist()
            ])


def load_features_numpy(path):
    data = {
        'features': [],
        'labels': [],
        'names': [], 
    }

    with open(path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data['features'].append(np.array([float(el) for el in row['feature'][1:-1].split(', ')]))
            data['labels'].append(row['label'])
            data['names'].append(row['name'])

    return data
