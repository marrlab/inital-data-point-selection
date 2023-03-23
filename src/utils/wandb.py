
from omegaconf import DictConfig, OmegaConf
import wandb

def init_run(cfg: DictConfig):
    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project,
        config=wandb_config,
        settings=wandb.Settings(start_method='thread'))
