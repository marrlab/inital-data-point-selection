
import torch

def flatten_tensor_dicts(ds):
    keys = list(ds[0].keys())

    return {
        key: torch.stack(d[key] for d in ds)
        for key in keys
    }
