
import json
from omegaconf import DictConfig, OmegaConf
import wandb
import pandas as pd
from src.utils.utils import flatten_dict
from src.vis.constants import CRITERIA_ORDER, HALF_IN_HALF_TRAIN_SAMPLES_TO_CLUSTERS 

def login():
    wandb.login(key='a29d7c338a594e427f18a0f1502e5a8f36e9adfb')

def init_run(cfg: DictConfig):
    login()
    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project,
        config=wandb_config,
        settings=wandb.Settings(start_method='thread'))

def get_runs(project: str, filters: dict = {}) -> pd.DataFrame:
    print('fetching wandb runs for the following filters:')
    print(json.dumps(filters, indent=2))

    login()
    api = wandb.Api()

    filters['state'] = 'finished'
    filters['tags'] = {'$nin': ['v1']}
    runs = api.runs(f'mireczech/{project}', filters=filters)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    print(f'total of {len(runs_df)} have been retrieved')
    print()

    return runs_df

def dict_to_filters(d):
    d = flatten_dict(d, separator='.')
    for key in list(d.keys()):
        d['config.' + key] = d[key]
        del d[key]

    return d

def cast_dict_to_int(d):
    return {int(k): int(v) for k, v in d.items()}

def get_best_criterium_filters(criterium, train_samples=100, model_type='simclr', model_version='v1'):
    assert criterium in CRITERIA_ORDER
    assert train_samples in (100, 200, 500)

    filters = None
    if criterium in 'cold_paws':
        filters = dict_to_filters({
            'training': {
                'train_samples': train_samples,
                'oversample': True,
                'weights': {
                    'type': model_type,
                    'version': model_version,
                    'freeze': True,
                },
            },
            'use_scan_weights': False,
        })
    else:
        clusters = None
        if criterium in ['closest', 'furthest']:
            clusters = train_samples
        elif criterium == 'half_in_half':
            clusters = HALF_IN_HALF_TRAIN_SAMPLES_TO_CLUSTERS[train_samples]
        else:
            clusters = 1

        filters = dict_to_filters({
            'training': {
                'train_samples': train_samples,
                'oversample': True,
                'weights': {
                    'type': model_type,
                    'version': model_version,
                    'freeze': True,
                },
            },
            'features': {
                'scaling': 'standard'
            },
            'kmeans': {
                'mode': 'kmeans',
                'clusters': clusters,
                'criterium': criterium
            },
            'use_scan_weights': False,
        })

    return filters
