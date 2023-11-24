
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd

def is_hydra_initialized():
    return HydraConfig.initialized()

def get_original_cwd_safe():
    if is_hydra_initialized():
        return get_original_cwd()
    
    return './'

