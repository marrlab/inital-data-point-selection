
import os
import csv
import torch
import numpy as np
from PIL import Image

class MatekDataset(torch.utils.data.Dataset):
    """
    Args:
        images_dir (str): path to images folder
        segmentation_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            split, 
            dataset_root_dir='/content/drive/MyDrive/master thesis/datasets/ssl_vs_al/data/matek',
            preprocess=None,
            features_path=None,
            load_images=True,
    ):
        assert split in ('train', 'test')
        assert load_images or features_path is not None, 'unsupported combination of arguments'

        self.images_dir = os.path.join(dataset_root_dir, split)
        self.labels_text = [f.name for f in os.scandir(self.images_dir) if f.is_dir()]
        self.labels = list(range(len(self.labels_text)))
        self.labels_text_mapping = {text: id for id, text in enumerate(self.labels_text)}

        self.images_data = {
            'names': [],
            'labels_text': [],
            'labels': [],
            'paths': []
        }
        for label_text in self.labels_text:
            label_dir = os.path.join(self.images_dir, label_text)
            for f in os.scandir(label_dir):
                self.images_data['names'].append(f.name)
                self.images_data['labels_text'].append(label_text)
                self.images_data['labels'].append(self.labels_text_mapping[label_text])
                self.images_data['paths'].append(os.path.join(label_dir, f.name))

        self.preprocess = preprocess

        # mapping from image name to feature
        self.features = None
        if features_path is not None:
            self.features = {}
            with open(features_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    self.features[row['name']] = np.array([float(el) for el in row['feature'][1:-1].split(', ')])

        self.load_images = load_images
    
    def __getitem__(self, i):
        data_point = {
            'label': self.images_data['labels'][i],
            'name': self.images_data['names'][i],
            'image': None,
            'feature': None
        }

        if self.load_images:
            input_image = Image.open(self.images_data['paths'][i])
            input_image = input_image.convert('RGB')
            input_tensor = self.preprocess(input_image)

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_tensor = input_tensor.to('cuda')

            data_point['image'] = input_tensor

        if self.features is not None:
            data_point['feature'] = self.features[self.images_data['names'][i]]

        return data_point

    def __len__(self):
        return len(self.images_data['names'])