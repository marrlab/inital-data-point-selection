
import copy
import wandb
from datasets.datasets import MatekDataset
from models.classifiers import \
    get_classifier_imagenet, get_classifier_imagenet_preprocess_only
from datasets.subsets import get_n_random, get_n_kmeans
from tasks.training import train_image_classifier


def random_baseline():
    assert wandb.config.dataset in ('matek')

    preprocess = get_classifier_imagenet_preprocess_only(
        wandb.config.architecture)

    train_dataset, val_dataset = None, None
    if wandb.config.dataset == 'matek':
        train_dataset = MatekDataset('train', preprocess=preprocess)
        val_dataset = MatekDataset('test', preprocess=preprocess)

    train_subset = get_n_random(train_dataset, wandb.config.train_samples)
    train_subset.relabel()

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)
    val_subset = get_n_random(val_subset, wandb.config.val_samples)

    num_classes = len(train_subset.labels)
    model, _ = get_classifier_imagenet(wandb.config.architecture, num_classes)

    train_image_classifier(model, train_subset, val_subset)


# wandb.config requirements:
# .epochs
# .learning_rate
# .batch_size
# .architecture
# .dataset
# .train_samples
# .val_samples
# .features_path
# .mode
# .criterium
def badge_sampling():
    assert wandb.config.dataset in ('matek')

    preprocess = get_classifier_imagenet_preprocess_only(
        wandb.config.architecture)

    train_dataset, val_dataset = None, None
    if wandb.config.dataset == 'matek':
        train_dataset = MatekDataset(
            'train', preprocess=preprocess, features_path=wandb.config.features_path)
        val_dataset = MatekDataset('test', preprocess=preprocess)

    train_subset = get_n_kmeans(train_dataset, wandb.config.train_samples,
                                mode=wandb.config.mode, criterium=wandb.config.criterium)
    train_subset.relabel()

    val_subset = copy.deepcopy(val_dataset)
    val_subset.match_labels_and_filter(train_subset)
    val_subset = get_n_random(val_subset, wandb.config.val_samples)

    num_classes = len(train_subset.labels)
    model, _ = get_classifier_imagenet(wandb.config.architecture, num_classes)

    train_image_classifier(model, train_subset, val_subset)
