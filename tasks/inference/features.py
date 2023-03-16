
import re
import csv
import sys
import torch
import logging
import torchvision
from models.lightning_modules import SimCLRModel
from utils.utils import load_yaml_as_obj
from datasets.datasets import get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate

def extract_features_simclr(config_path: str):
    # config init
    config = load_yaml_as_obj(config_path)

    # we create a torchvision transformation for embedding the dataset
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (config.input_size, config.input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])

    # creating datasets
    dataset_class = get_dataset_class_by_name(config.dataset)
    dataset = dataset_class('train')
    dataset_lightly = LightlyDataset(
        input_dir=dataset.images_dir,
        transform=transforms,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_lightly,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers
    )

    # loading the model and setting it to inference mode
    model = SimCLRModel.load_from_checkpoint(config.model_save_path)
    model.eval()

    features = []
    labels = []
    names = []
    with torch.no_grad():
        for image, label, name in data_loader:
            image = image.to(model.device)
            feature = model.backbone(image).flatten(start_dim=1)
            name = [
                re.search(r'[^/]+/(.+)', n).group(1)
                for n in name
            ]

            features.append(feature)
            labels.extend(label)
            names.extend(name)

    features = torch.cat(features, 0)

    # creating csv file with features
    with open(config.features_save_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['name', 'label', 'feature'])
        for i in range(len(features)):
            csv_writer.writerow([
                names[i],
                labels[i],
                features[i].tolist()
            ])

if __name__ == '__main__':
    task_name = sys.argv[1]
    config_path = sys.argv[2] 
    if task_name == 'extract_features_simclr':
        extract_features_simclr(config_path)
    else:
        raise ValueError(f'unknown task name: {task_name}')
