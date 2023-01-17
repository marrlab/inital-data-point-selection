
import torch

def flatten_tensor_dicts(ds):
    print(ds)
    keys = list(ds[0].keys())

    return {
        key: torch.stack(tuple(d[key] for d in ds))
        for key in keys
    }
