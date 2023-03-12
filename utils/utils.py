
import yaml
import torch
import wandb
import pandas as pd
from typing import Iterable
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


def load_dataframes(paths: Iterable[str], contains_index=True) -> Iterable[pd.DataFrame]:
    if contains_index:
        return [
            pd.read_csv(p, index_col=0)
            for p in paths
        ]
    else:
        return [
            pd.read_csv(p)
            for p in paths
        ]

def get_runs(project: str) -> pd.DataFrame:
    wandb.login(key='a29d7c338a594e427f18a0f1502e5a8f36e9adfb')
    api = wandb.Api()

    runs = api.runs(f'mireczech/{project}')

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)
        
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    return runs_df

def load_yaml_as_dict(yaml_path: str) -> dict:
    d = None
    with open(yaml_path, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    return d

def load_yaml_as_obj(yaml_path: str) -> object:
    d = load_yaml_as_dict(yaml_path)

    class Struct:
        def __init__(self, **entries): 
            self.__dict__.update(entries)

    obj = Struct(**d)

    return obj
