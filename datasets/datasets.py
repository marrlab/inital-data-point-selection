
import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.utils import delete_at_multiple_indices

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
        self.load_images = load_images

        self.images_dir = None
        self.labels_text = []
        self.labels = []
        self.labels_text_mapping = {}

        self.images_data = {
            'names': [],
            'labels_text': [],
            'labels': [],
            'paths': []
        }
        self.features = {}


    def __getitem__(self, i):
        data_point = {
            'label': self.images_data['labels'][i],
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

    # restructures labeling, removing those for which there are no data points
    # e.g. possible lables [0, 1, 2, 3, 4], present labels [0, 3, 4], new labels [0, 1, 2]
    def relabel_dataset(self):
        labels_unique = np.array(sorted(list(set(self.images_data['labels']))))

        self.labels_text = np.delete(self.labels_text, labels_unique)
        self.labels = list(range(len(labels_unique)))
        self.labels_text_mapping = {text: id for id, text in enumerate(self.labels_text)} 

        for i in range(len(self)):
            self.images_data['labels'][i] = self.labels_text_mapping[self.images_data['labels_text'][i]]

    # changes the labeling so that it corresponds to the source dataset
    # removes data points with labels (text) that don't appear in the source dataset
    def match_labels_and_filter(self, source_dataset):
        self.labels_text = source_dataset.labels_text.copy()
        self.labels = source_dataset.labels.copy()
        self.labels_text_mapping = source_dataset.labels_text_mapping.copy()

        for i in reversed(range(len(self))):
            if self.images_data['labels_text'][i] not in self.labels_text:
                del self.images_data['names'][i]
                del self.images_data['labels_text'][i]
                del self.images_data['labels'][i]
                del self.images_data['paths'][i]

            self.images_data['labels'][i] = self.labels_text_mapping[self.images_data['labels_text'][i]]

class MatekDataset(ImageDataset):
    def __init__(
            self, 
            split, 
            dataset_root_dir='/content/drive/MyDrive/master thesis/datasets/ssl_vs_al/data/matek',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        ImageDataset.__init__(self, split, dataset_root_dir, preprocess, features_path, load_images)

        assert split in ('train', 'test')

        self.images_dir = os.path.join(dataset_root_dir, split)
        self.labels_text = [f.name for f in os.scandir(self.images_dir) if f.is_dir()]
        self.labels = list(range(len(self.labels_text)))
        self.labels_text_mapping = {text: id for id, text in enumerate(self.labels_text)}
        
        for label_text in self.labels_text:
            label_dir = os.path.join(self.images_dir, label_text)
            for f in os.scandir(label_dir):
                self.images_data['names'].append(f.name)
                self.images_data['labels_text'].append(label_text)
                self.images_data['labels'].append(self.labels_text_mapping[label_text])
                self.images_data['paths'].append(os.path.join(label_dir, f.name))

        # mapping from image name to feature
        if features_path is not None:
            with open(features_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    self.features[row['name']] = np.array([float(el) for el in row['feature'][1:-1].split(', ')])

        self.load_images = load_images
    