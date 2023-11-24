
import os
import re
import csv
import torch
import hydra
import torchvision
from tqdm import tqdm
from omegaconf import DictConfig
from src.models.scan import SCAN
from src.datasets.datasets import get_dataset_class_by_name
from lightly.data import LightlyDataset
from collections import defaultdict
from src.models.classifiers import get_ssl_preprocess, get_ssl_transform, get_ssl_model_class
from src.utils.utils import get_cpu_count, to_best_available_device
from src.models.build_data_path import get_model_path, get_scan_path, get_precomputed_features_folder_path

@hydra.main(version_base=None, config_path='../../conf', config_name='precompute_features')
def main(cfg: DictConfig):
    # loading the model and setting it to inference mode
    model = None
    if cfg.use_scan_weights:
        print('loading scan model')
        model = SCAN.load_from_checkpoint(get_scan_path(cfg), cfg=cfg)
        model = to_best_available_device(model)
    else:
        print('loading ssl weights')
        ssl_model_class = get_ssl_model_class(cfg)
        model = ssl_model_class.load_from_checkpoint(get_model_path(cfg), cfg=cfg)
    model = to_best_available_device(model)
    model.eval()

    # loading datasets
    preprocess = get_ssl_preprocess(cfg)    
    transform = get_ssl_transform(cfg)    

    dataset_class = get_dataset_class_by_name(cfg.dataset.name)

    dataset = dataset_class(
        'train', 
        transform=transform, 
        preprocess=torchvision.transforms.Resize(cfg.dataset.input_size, antialias=True)
    )
    compute_and_save(model, 'train', dataset, cfg.num_augmentations, cfg)

    dataset = dataset_class(
        'val', 
        preprocess=preprocess
    )
    compute_and_save(model, 'val', dataset, 1, cfg)

    dataset = dataset_class(
        'test', 
        preprocess=preprocess
    )
    compute_and_save(model, 'test', dataset, 1, cfg)


def compute_and_save(model, split, dataset, repetitions, cfg):
    print(f'computing features for {split} split with {repetitions} repetitions')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=get_cpu_count()
    )

    name_to_features = defaultdict(list)
    name_to_label_text = {}
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            for data_point in tqdm(data_loader, leave=False):
                images = data_point['image']
                labels = data_point['label']
                names = data_point['name']

                images = images.to(model.device)
                features = model.backbone(images)

                if len(features.shape) != 2:
                    features = features.flatten(start_dim=1)

                for i in range(len(names)):
                    name_to_features[names[i]].append(features[i])

                    label_text = dataset.labels_text[labels[i].item()]
                    name_to_label_text[names[i]] = label_text

    # TODO: add asserts
    assert len(name_to_features) == len(dataset)
    assert all(len(features) == repetitions for features in name_to_features.values())

    root = os.path.join(
        get_precomputed_features_folder_path(cfg, absolute=True),
        split
    )     
    for name in name_to_features:
        folder = os.path.join(root, name_to_label_text[name])
        os.makedirs(folder, exist_ok=True)

        path = os.path.join(folder, name)

        features = torch.vstack(name_to_features[name])
        torch.save(features, path)

    print()

if __name__ == '__main__':
    main()
