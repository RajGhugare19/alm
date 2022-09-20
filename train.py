import wandb
import hydra
import warnings
warnings.simplefilter("ignore", UserWarning)

from omegaconf import DictConfig

@hydra.main(config_path='cfgs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if cfg.benchmark == 'gym':
        from workspaces.mujoco_workspace import MujocoWorkspace as W
    else:
        raise NotImplementedError

    if cfg.wandb_log:
        project_name = 'alm_' + cfg.id
        with wandb.init(project=project_name, entity='raj19', config=dict(cfg), settings=wandb.Settings(start_method="thread")):
            wandb.run.name = cfg.wandb_run_name
            workspace = W(cfg)
            workspace.train()
    else:
        workspace = W(cfg)
        workspace.train()
    
if __name__ == '__main__':
    main()