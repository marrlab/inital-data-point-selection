
import torch

def flatten_tensor_dicts(ds):
    keys = list(ds[0].keys())

    flatten_d = {}
    for key in keys:
        if len(ds[0][key].shape) == 0:
            flatten_d[key] = torch.stack(tuple(d[key] for d in ds))
        elif len(ds[0][key].shape) == 1:
            flatten_d[key] = torch.cat(tuple(d[key] for d in ds))
        else:
            raise ValueError('unsupported tensor shape')

    return flatten_d
