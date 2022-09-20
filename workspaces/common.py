import numpy as np
import utils

def make_agent(env, device, cfg):
    
    if cfg.agent == 'alm':
        
        from agents.alm import AlmAgent

        num_states = np.prod(env.observation_space.shape)
        num_actions = np.prod(env.action_space.shape)
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        if cfg.id == 'Humanoid-v2':
            cfg.env_buffer_size = 1000000
        buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

        agent = AlmAgent(device, action_low, action_high, num_states, num_actions,
                            buffer_size, cfg.gamma, cfg.tau, cfg.target_update_interval,
                            cfg.lr, cfg.max_grad_norm, cfg.batch_size, cfg.seq_len, cfg.lambda_cost,
                            cfg.expl_start, cfg.expl_end, cfg.expl_duration, cfg.stddev_clip, 
                            cfg.latent_dims, cfg.hidden_dims, cfg.model_hidden_dims,
                            cfg.wandb_log, cfg.log_interval
                            )
                            
    else:
        raise NotImplementedError

    return agent

def make_env(cfg):
    if cfg.benchmark == 'gym':
        import gym
        if cfg.id == 'AntTruncatedObs-v2' or cfg.id == 'HumanoidTruncatedObs-v2':
            utils.register_mbpo_environments()

        def get_env(cfg):
            env = gym.make(cfg.id) 
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed=cfg.seed)
            env.observation_space.seed(cfg.seed)
            env.action_space.seed(cfg.seed)
            return env 

        return get_env(cfg), get_env(cfg)
    
    else:
        raise NotImplementedError