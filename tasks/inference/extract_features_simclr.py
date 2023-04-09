
import re
import os
import csv
import sys
import torch
import logging
import hydra
import torchvision
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from src.models.lightning_modules import SimCLRModel
from src.datasets.datasets import get_dataset_class_by_name
from lightly.data import LightlyDataset, SimCLRCollateFunction, collate

@hydra.main(version_base=None, config_path='../../conf', config_name='extract_features_simclr')
def main(cfg: DictConfig):
    # we create a torchvision transformation for embedding the dataset
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (cfg.training.input_size, cfg.training.input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=collate.imagenet_normalize['mean'],
            std=collate.imagenet_normalize['std'],
        )
    ])

    # creating datasets
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    dataset = dataset_class('train')
    dataset_lightly = LightlyDataset(
        input_dir=dataset.images_dir,
        transform=transforms,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_lightly,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers
    )

    # loading the model and setting it to inference mode
    model = SimCLRModel.load_from_checkpoint(os.path.join(get_original_cwd(), cfg.model_save_path), cfg=cfg)
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
    with open('features.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['name', 'label', 'feature'])
        for i in range(len(features)):
            csv_writer.writerow([
                names[i],
                labels[i],
                features[i].tolist()
            ])

if __name__ == '__main__':
    main()
