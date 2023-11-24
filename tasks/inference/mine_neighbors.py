
import re
import csv
import torch
import hydra
import faiss
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from src.models.simclr import SimCLR
from src.datasets.datasets import get_dataset_class_by_name
from lightly.data import LightlyDataset
from src.models.classifiers import get_ssl_preprocess, get_ssl_model_class
from src.utils.utils import get_cpu_count
from src.models.build_data_path import get_model_path, get_neighbors_path

@hydra.main(version_base=None, config_path='../../conf', config_name='mine_neighbors')
def main(cfg: DictConfig):
    assert cfg.training.weights.type == 'simclr', \
        'other ssl methods not yet supported; neighborhood mining needs to be determined first'

    print(f'starting to mine neighbors for {cfg.dataset.name}')
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
    ssl_model_class = get_ssl_model_class(cfg)
    model = ssl_model_class.load_from_checkpoint(get_model_path(cfg), cfg=cfg)
    model.eval()

    embeddings = []
    labels = []
    names = []
    with torch.no_grad():
        for image, label, name in tqdm(data_loader):
            image = image.to(model.device)
            embedding = model(image)
            name = [
                re.search(r'[^/]+/(.+)', n).group(1)
                for n in name
            ]

            embeddings.append(embedding)
            labels.extend(label)
            names.extend(name)

    embeddings = torch.cat(embeddings, 0)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # computing the neighbors
    embeddings = embeddings.cpu().numpy()
    _, dim = embeddings.shape[0], embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # requires gpu support (not really needed in our case)
    # index = faiss.index_cpu_to_all_gpus(index)
    index.add(embeddings)

    print('computing nearest neighbors')
    distances, indices = index.search(embeddings, cfg.neighbors + 1) # Sample itself is included
    distances, indices = distances[:,1:], indices[:,1:]
        
    # evaluate 
    neighbor_targets = np.take(np.array(labels), indices, axis=0) # Exclude sample itself for eval
    anchor_targets = np.repeat(np.array(labels).reshape(-1,1), cfg.neighbors, axis=1)
    accuracy = np.mean(neighbor_targets == anchor_targets)
    print(f'neighbors in the same class accuracy: {accuracy}')

    neighbors = []
    for i in range(len(indices)):
        neighbors.append([
            names[j] 
            for j in indices[i]
        ])


    # creating csv file with features
    with open(get_neighbors_path(cfg), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['name', 'label', 'neighbors'])
        for i in range(len(names)):
            csv_writer.writerow([
                names[i],
                labels[i],
                f"[{', '.join(neighbors[i])}]"
            ])

if __name__ == '__main__':
    main()
