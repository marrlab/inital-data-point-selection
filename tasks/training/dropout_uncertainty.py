
import hydra
import torch
import wandb
import torchvision
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.datasets.datasets import get_dataset_class_by_name
from src.models.classifiers import get_ssl_preprocess, get_ssl_transform
from src.utils.wandb import init_run, cast_dict_to_int
from src.utils.utils import get_cpu_count, to_best_available_device, argmax_top_n
from src.models.build_data_path import get_scan_path
from src.models.scan import SCAN, get_classifier_from_scan
from src.models.training import train_image_classifier
from src.datasets.subsets import get_by_indices


@hydra.main(version_base=None, config_path='../../conf', config_name='dropout_uncertainty')
def main(cfg: DictConfig):
    assert cfg.training.weights.type == 'simclr', \
        'other ssl methods not yet supported'

    init_run(cfg)

    # fetching the augmentations and preprocessing steps
    preprocess = get_ssl_preprocess(cfg)

    # creating datasets
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)

    train_dataset = dataset_class(
        split='train', 
        preprocess=preprocess
    )
    # val_dataset = dataset_class('val')

    # setting up the dataloader(s)
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.monte_carlo.batch_size, 
        num_workers=get_cpu_count()
    )

    # defining the model
    model = SCAN.load_from_checkpoint(get_scan_path(cfg), cfg=cfg)
    model = to_best_available_device(model)

    # monte carlo
    print('computing monte carlo to determine most uncertain data points')

    preds = []
    model.eval()
    with torch.no_grad():
        model.dropout.train()

        total_iterations = cfg.monte_carlo.epochs * len(train_data_loader)
        progress_bar = tqdm(total=total_iterations)

        for _ in range(cfg.monte_carlo.epochs):
            for batch in train_data_loader:
                image = batch['image']
                image = image.to(model.device)

                logits = model(image)
                logits = logits.cpu().numpy()

                preds.extend(np.argmax(logits, axis=1))

                progress_bar.update(1)
            
        progress_bar.close()

        assert len(preds) == len(train_dataset) * cfg.monte_carlo.epochs

    preds = np.array(preds)
    cluster_assignments = np.reshape(preds, (-1,len(train_dataset))).T

    entropies = []
    for a in cluster_assignments:
        prob = np.zeros((cfg.scan.num_clusters,))
        for i in a:
            prob[i] += 1

        prob /= np.sum(prob)
        entropies.append(entropy(prob))

    entropies = np.array(entropies)

    np.save('entropies.npy', entropies)
    np.save('cluster_assignments.npy', cluster_assignments)

    most_uncertain_indices = argmax_top_n(entropies, cfg.training.train_samples)
    train_subset = get_by_indices(train_dataset, most_uncertain_indices)

    # preparing for classifier training
    preprocess = get_ssl_preprocess(cfg)
    transform = get_ssl_transform(cfg)    

    train_subset.preprocess = torchvision.transforms.Resize(cfg.dataset.input_size, antialias=True)
    train_subset.transform = transform
    test_dataset = dataset_class('test', preprocess=preprocess)

    # reassigning classes based on those that were discovered 
    train_subset.reassign_classes()
    label_counts = defaultdict(int)
    for i in range(len(train_subset)):
        label_counts[train_subset[i]['label']] += 1
    wandb.config.labels = train_subset.get_number_of_labels()
    wandb.config.labels_text = train_subset.labels_text
    wandb.config.label_counts = cast_dict_to_int(label_counts)
    wandb.config.label_to_class_mapping = cast_dict_to_int(
        train_subset.label_to_class_mapping)
    wandb.config.class_to_label_mapping = cast_dict_to_int(
        train_subset.class_to_label_mapping)
    wandb.config.classes = train_subset.get_number_of_classes()

    # val_dataset.match_classes(train_subset)
    test_dataset.match_classes(train_subset)

    num_classes = train_subset.get_number_of_classes()

    # classifier training
    model = get_classifier_from_scan(cfg, num_classes)
    train_image_classifier(model, train_subset, None, test_dataset, cfg)

    wandb.finish()

if __name__ == '__main__':
    main()
