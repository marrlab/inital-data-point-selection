
import torch
import pandas as pd
from typing import Iterable, Union
from utils.types import Result

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


def result_to_dataframe(result: Result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result

    return pd.concat(result, ignore_index=True)
