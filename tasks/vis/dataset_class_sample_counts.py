
import glob
import os
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src.datasets.datasets import get_dataset_class_by_name
from src.datasets.datasets import get_dataset_class_by_name
from collections import Counter


@hydra.main(version_base=None, config_path='../../conf', config_name='dataset_class_sample_counts')
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f'saving everything to: {os.getcwd()}')

    dataset_names = ('matek', 'isic', 'retinopathy', 'jurkat', 'cifar10')
    for dataset_name in dataset_names:
        dataset_class = get_dataset_class_by_name(dataset_name)
        dataset = dataset_class(split='train')

        labels = dataset.images_data['labels']
        labels = sorted(labels)
        counter = Counter(labels)

        print(dataset_name)
        print(f'dataset size: {len(labels)}')
        print(f'min count: {min(counter.values())}, max count: {max(counter.values())}')
        print('label counts:')
        print(counter)
        print()

 
if __name__ == '__main__':
    main()
