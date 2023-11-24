
import os
import hydra
from src.utils.hydra import get_original_cwd_safe

DATA_FOLDER = 'src/models/data'

def get_cold_paws_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'cold_paws', str(cfg.training.train_samples))
    
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def get_precomputed_features_folder_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'precomputed_features')

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def get_features_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'features.csv')

    return path

def get_scan_features_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'scan_features.csv')

    return path

def get_neighbors_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'neighbors.csv')

    return path

def get_model_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'model.ckpt')

    return path

def get_scan_path(cfg, absolute=True):
    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, 'scan.ckpt')

    return path

def get_vis_folder_path(cfg, absolute=True):
    config_name = hydra.utils.HydraConfig.get().job.config_name

    folder_path = get_model_folder_path(cfg, absolute=absolute)
    path = os.path.join(folder_path, config_name)

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def get_model_folder_path(cfg, absolute=True):
    folder_path = get_data_folder_path(absolute=absolute)

    path = os.path.join(
        folder_path,
        cfg.dataset.name, 
        cfg.training.weights.type, 
        cfg.training.weights.version
    )

    if not os.path.exists(path):
        os.makedirs(path)
    
    return path

def get_data_folder_path(absolute=True):
    root = get_original_cwd_safe()

    path_relative = DATA_FOLDER
    path_absolute = os.path.join(
        root, path_relative
    )

    if absolute:
        return path_absolute
    
    return path_relative
