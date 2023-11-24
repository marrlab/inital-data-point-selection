
import os
import csv
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.utils import to_best_available_device
from src.utils.hydra import get_original_cwd_safe


max_int = sys.maxsize
while True:
    # decrease the max_int value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)

class NeighborsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, neighbors_path):
        self.dataset = dataset
        self.neighbors_path = os.path.join(get_original_cwd_safe(), neighbors_path)

        # precomputing the mapping from image name to index
        self.name_to_index = {}
        for i, name in enumerate(dataset.images_data['names']):
            self.name_to_index[name] = i

        self.neighbors = {}
        with open(self.neighbors_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                self.neighbors[row['name']] = np.array(row['neighbors'][1:-1].split(', '))

        assert len(self.neighbors) == len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        anchor = self.dataset.__getitem__(index)
        
        # randomly sampling one of the neighbors
        neighbor_name = np.random.choice(self.neighbors[anchor['name']], 1)[0]
        neighbor_index = self.name_to_index[neighbor_name]
        neighbor = self.dataset.__getitem__(neighbor_index)

        # possibly useful
        # output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        
        return {
            'anchor_image': anchor['image'],
            'neighbor_image': neighbor['image'],
            'anchor_name': anchor['name'],
            'neighbor_name': neighbor['name'],
        }

class ImageFeaturesDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split,
        dataset_root_dir
    ):
        assert split in ('train', 'val', 'test')

        self.split = split
        self.dataset_root_dir = dataset_root_dir

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

        # mapping from image name to feature
        self.features = None
        self.loaded = False

        # form image folder dataset
        self.features_dir = os.path.join(get_original_cwd_safe(), dataset_root_dir, split)
        self.labels_text = sorted([f.name for f in os.scandir(self.features_dir) if f.is_dir()])
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

    # names=None if to load all
    def load(self, names):
        assert not self.loaded, 'features already loaded'

        self.loaded = True

        self.features = []
        for label_text in self.labels_text:
            label_dir = os.path.join(self.features_dir, label_text)
            for f in sorted(os.scandir(label_dir), key=lambda el: el.name):
                if names is not None and f.name not in names:
                    continue

                self.images_data['names'].append(f.name)
                self.images_data['labels_text'].append(label_text)
                self.images_data['labels'].append(self.labels_text_mapping[label_text])
                self.images_data['paths'].append(os.path.join(label_dir, f.name))
                
                features_path = os.path.join(label_dir, f.name)
                self.features.append(torch.load(features_path))

        print(f"total of {len(self.images_data['names'])} loaded")
        if names is not None:
            assert len(self.features) == len(names)

        self.features = torch.stack(self.features).to(device='cpu')
        assert len(self.features.shape) == 3
        assert self.features.shape[0] == len(self.images_data['names'])

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

    def match_classes(self, source_dataset):
        assert self.labels == source_dataset.labels

        self.classes = source_dataset.classes.copy()
        self.label_to_class_mapping = source_dataset.label_to_class_mapping.copy()
        self.class_to_label_mapping = source_dataset.class_to_label_mapping.copy()

    def get_number_of_classes(self):
        return len(self.classes)

    def get_number_of_labels(self):
        return len(self.labels)

    def get_feature_dim(self):
        return self.features.shape[2]

    def __getitem__(self, i):
        data_point = {
            'label': self.images_data['labels'][i],
            'class': self.label_to_class_mapping[self.images_data['labels'][i]],
            'name': self.images_data['names'][i],
            'image': self.features[i, np.random.randint(0, self.features.shape[1])]
        }

        return data_point

    def __len__(self):
        return len(self.images_data['names'])
        

# abstract class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            split, 
            dataset_root_dir=None,
            transform=None,
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
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([])
        self.features_path = features_path
        if self.features_path is not None:
            self.features_path = os.path.join(get_original_cwd_safe(), self.features_path)
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

        # mapping from image name to feature
        self.features = {}

    def __getitem__(self, i):
        data_point = {
            'label': self.images_data['labels'][i],
            'class': self.label_to_class_mapping[self.images_data['labels'][i]],
            'name': self.images_data['names'][i],
        }

        if self.load_images:
            input_image = Image.open(self.images_data['paths'][i])
            input_image = input_image.convert('RGB')

            # debug
            input_image = self.transform(input_image)
            input_tensor = self.preprocess(input_image)

            # move the input and model to the best device for speed if available
            # input_tensor = to_best_available_device(input_tensor)

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
    def match_classes(self, source_dataset):
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

    def get_feature_dim(self):
        return len(self.features[next(iter(self.features.keys()))])


class ImageDatasetWithFolderStructure(ImageDataset):
    def __init__(
            self, 
            split, 
            dataset_root_dir,
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDataset.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

        assert split in ('train', 'val', 'test')

        self.images_dir = os.path.join(get_original_cwd_safe(), dataset_root_dir, split)
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

        if self.features_path is not None:
            with open(self.features_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    self.features[row['name']] = np.array([float(el) for el in row['feature'][1:-1].split(', ')])

class MatekDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/matek',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)


class IsicDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

class IsicSmallDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic_small',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

class IsicSmallestDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic_smallest',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

class IsicSmallestUnbalancedDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/isic_smallest_unbalanced',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

class RetinopathyDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/retinopathy',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

class JurkatDataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/jurkat',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

class Cifar10Dataset(ImageDatasetWithFolderStructure):
    def __init__(
            self, 
            split, 
            dataset_root_dir='src/datasets/data/cifar10',
            transform=None,
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDatasetWithFolderStructure.__init__(self, split, dataset_root_dir, transform, preprocess, features_path, load_images)

def get_dataset_class_by_name(dataset_name: str):
    if dataset_name == 'matek':
        return MatekDataset
    elif dataset_name == 'isic':
        return IsicDataset
    elif dataset_name == 'isic_small':
        return IsicSmallDataset
    elif dataset_name == 'isic_smallest':
        return IsicSmallestDataset
    elif dataset_name == 'isic_smallest_unbalanced':
        return IsicSmallestUnbalancedDataset
    elif dataset_name == 'retinopathy':
        return RetinopathyDataset
    elif dataset_name == 'jurkat':
        return JurkatDataset
    elif dataset_name == 'cifar10':
        return Cifar10Dataset
    else:
        raise ValueError(f'unknown dataset name: {dataset_name}')
