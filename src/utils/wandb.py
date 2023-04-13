
from omegaconf import DictConfig, OmegaConf
import wandb
import pandas as pd

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

def get_runs(project: str) -> pd.DataFrame:
    login()
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

    return runs_df

def cast_dict_to_int(d):
    return {int(k): int(v) for k, v in d.items()}
