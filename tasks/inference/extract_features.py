
import re
import csv
import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from src.models.scan import SCAN
from src.datasets.datasets import get_dataset_class_by_name
from lightly.data import LightlyDataset
from src.models.classifiers import get_ssl_preprocess, get_ssl_model_class
from src.utils.utils import get_cpu_count, to_best_available_device
from src.models.build_data_path import get_model_path, get_features_path, get_scan_path, get_scan_features_path

@hydra.main(version_base=None, config_path='../../conf', config_name='extract_features')
def main(cfg: DictConfig):
    transforms = get_ssl_preprocess(cfg)

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
        num_workers=get_cpu_count()
    )

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

    features = []
    labels = []
    names = []
    with torch.no_grad():
        for image, label, name in tqdm(data_loader):
            image = image.to(model.device)
            feature = model.backbone(image)

            if not cfg.use_scan_weights:
                feature = feature.flatten(start_dim=1)

            name = [
                re.search(r'[^/]+/(.+)', n).group(1)
                for n in name
            ]

            features.append(feature)
            labels.extend(label)
            names.extend(name)

    features = torch.cat(features, 0)

    # creating csv file with features
    features_path = None
    if cfg.use_scan_weights:
        features_path = get_scan_features_path(cfg)
    else:
        features_path = get_features_path(cfg)
    print(f'saving feature to {features_path}')

    with open(features_path, 'w') as csv_file:
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
