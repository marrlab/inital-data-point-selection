
import os
import csv
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hydra.utils import get_original_cwd


max_int = sys.maxsize
while True:
    # decrease the max_int value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)

# abstract class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            split, 
            dataset_root_dir=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        assert load_images or features_path is not None, 'unsupported combination of arguments'

        self.split = split
        self.dataset_root_dir = dataset_root_dir
        self.preprocess = preprocess
        if self.preprocess is None:
            self.preprocess = transforms.ToTensor()
        self.features_path = features_path
        if self.features_path is not None:
            self.features_path = os.path.join(get_original_cwd(), self.features_path)
        self.load_images = load_images

        self.images_dir = None
        self.labels_text = []
        self.labels = []
        self.classes = []
        self.labels_text_mapping = {}
        self.label_to_class_mapping = {}
        self.class_to_label_mapping = {}

        self.images_data = {
            'names': [],
            'labels_text': [],
            'labels': [],
            'paths': []
        }
        self.features = {}

        # mapping from image name to feature
        if self.features_path is not None:
            with open(self.features_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    self.features[row['name']] = np.array([float(el) for el in row['feature'][1:-1].split(', ')])

    def __getitem__(self, i):
        data_point = {
            'label': self.images_data['labels'][i],
            'class': self.label_to_class_mapping[self.images_data['labels'][i]],
            'name': self.images_data['names'][i],
        }

        if self.load_images:
            input_image = Image.open(self.images_data['paths'][i])
            input_image = input_image.convert('RGB')
            input_tensor = self.preprocess(input_image)

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_tensor = input_tensor.to('cuda')

            data_point['image'] = input_tensor

        if self.features_path is not None:
            data_point['feature'] = self.features[self.images_data['names'][i]]

        return data_point

    def __len__(self):
        return len(self.images_data['names'])

    # restructures label to class assignment, removing those for which there are no data points
    # e.g. possible lables [0, 1, 2, 3, 4], present labels [0, 3, 4], new classes [0, 1, 2]
    def reassign_classes(self):
        labels_remaining = sorted(list(set(self.images_data['labels'])))
        labels_missing = list(set(self.labels) - set(labels_remaining))

        self.classes = list(range(len(labels_remaining)))

        self.label_to_class_mapping = {
            label: label_index
            for label_index, label in enumerate(labels_remaining)
        }
        for label in labels_missing:
            self.label_to_class_mapping[label] = -1

        self.class_to_label_mapping = {
            _class: labels_remaining[_class]
            for _class in self.classes
        }

    # changes the label to class assignment so that it corresponds to the source dataset
    def match_classes_and_filter(self, source_dataset):
        assert self.labels == source_dataset.labels

        self.classes = source_dataset.classes.copy()
        self.label_to_class_mapping = source_dataset.label_to_class_mapping.copy()
        self.class_to_label_mapping = source_dataset.class_to_label_mapping.copy()

    def standard_scale_features(self):
        keys = list(self.features.keys())
        X = np.array([self.features[k] for k in keys])

        X_new = StandardScaler().fit_transform(X)

        for i, key in enumerate(keys):
            self.features[key] = X_new[i]
        
    def min_max_scale_features(self):
        keys = list(self.features.keys())
        X = np.array([self.features[k] for k in keys])

        X_new = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)

        for i, key in enumerate(keys):
            self.features[key] = X_new[i]

    def get_number_of_classes(self):
        return len(self.classes)

    def get_number_of_labels(self):
        return len(self.labels)


class ImageDatasetWithFolderStructure(ImageDataset):
    def __init__(
            self, 
            split, 
            dataset_root_dir,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDataset.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

        assert split in ('train', 'test')

        self.images_dir = os.path.join(get_original_cwd(), dataset_root_dir, split)
        self.labels_text = sorted([f.name for f in os.scandir(self.images_dir) if f.is_dir()])
        self.labels = list(range(len(self.labels_text)))
        self.classes = self.labels.copy()
        self.labels_text_mapping = {text: id for id, text in enumerate(self.labels_text)}
        self.label_to_class_mapping = {
            label: label
            for label in self.labels
        }
        self.class_to_label_mapping = {
            label: label
            for label in self.labels
        }
        
        for label_text in self.labels_text:
            label_dir = os.path.join(self.images_dir, label_text)
            for f in sorted(os.scandir(label_dir), key=lambda el: el.name):
                self.images_data['names'].append(f.name)
                self.images_data['labels_text'].append(label_text)
                self.images_data['labels'].append(self.labels_text_mapping[label_text])
                self.images_data['paths'].append(os.path.join(label_dir, f.name))


class MatekDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/matek',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)


class IsicDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

class IsicSmallDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic_small',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

class IsicSmallestDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic_smallest',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

class RetinopathyDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/retinopathy',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

# TODO
# class JurkatDataset(ImageDataset):

class Cifar10Dataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/cifar10',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

def get_dataset_class_by_name(dataset_name: str):
    if dataset_name == 'matek':
        return MatekDataset
    elif dataset_name == 'isic':
        return IsicDataset
    elif dataset_name == 'isic_small':
        return IsicSmallDataset
    elif dataset_name == 'isic_smallest':
        return IsicSmallestDataset
    elif dataset_name == 'retinopathy':
        return RetinopathyDataset
    else:
        raise ValueError(f'unknown dataset name: {dataset_name}')
