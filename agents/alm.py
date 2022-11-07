import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import numpy as np
import wandb
import utils
from models import Encoder, IdentityEncoder, ModelPrior, RewardPrior, Discriminator, Critic, Actor

class AlmAgent(object):
    def __init__(self, device, action_low, action_high, num_states, num_actions,
                env_buffer_size, gamma, tau, target_update_interval,
                lr, max_grad_norm, batch_size, seq_len, lambda_cost, 
                expl_start, expl_end, expl_duration, stddev_clip,
                latent_dims, hidden_dims, model_hidden_dims, 
                log_wandb, log_interval, identity_encoder
                ):

        self.device = device 
        self.action_low = action_low
        self.action_high = action_high
        
        #learning
        self.gamma = gamma 
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lambda_cost = lambda_cost

        #exploration
        self.expl_start = expl_start
        self.expl_end = expl_end
        self.expl_duration = expl_duration
        self.stddev_clip = stddev_clip

        #logging        
        self.log_wandb = log_wandb
        self.log_interval = log_interval

        self.env_buffer = utils.ReplayMemory(env_buffer_size, num_states, num_actions, np.float32)
        self._init_networks(num_states, num_actions, latent_dims, hidden_dims, model_hidden_dims, identity_encoder)
        self._init_optims(lr)

    def get_action(self, state, step, eval=False):
        std = utils.linear_schedule(self.expl_start, self.expl_end, self.expl_duration, step)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            z = self.encoder(state).sample()   
            action_dist = self.actor(z, std)    
            action = action_dist.sample(clip=None)
            
            if eval:
                action = action_dist.mean

        return action.cpu().numpy()[0]
    
    def get_representation(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            z = self.encoder(state).sample()   
        
        return z.cpu().numpy()

    def get_lower_bound(self, state_batch, action_batch):
        with torch.no_grad():
            z_batch = self.encoder_target(state_batch).sample()
            z_seq, action_seq = self._rollout_evaluation(z_batch, action_batch, std=0.1)

            reward = self.reward(z_seq[:-1], action_seq[:-1])
            kl_reward = self.classifier.get_reward(z_seq[:-1], action_seq[:-1], z_seq[1:])
            discount = self.gamma * torch.ones_like(reward)
            q_values_1, q_values_2 = self.critic(z_seq[-1], action_seq[-1])      
            q_values = torch.min(q_values_1, q_values_2) 

            returns = torch.cat([reward + self.lambda_cost * kl_reward, q_values.unsqueeze(0)])
            discount = torch.cat([torch.ones_like(discount[:1]), discount])
            discount = torch.cumprod(discount, 0)

            lower_bound = torch.sum(discount * returns, dim=0)
        return lower_bound.cpu().numpy()    

    def _rollout_evaluation(self, z_batch, action_batch, std):
        z_seq = [z_batch]
        action_seq = [action_batch]
        with torch.no_grad():
            for t in range(self.seq_len):
                z_batch = self.model(z_batch, action_batch).sample()

                action_dist = self.actor(z_batch.detach(), std)
                action_batch = action_dist.mean
                
                z_seq.append(z_batch)
                action_seq.append(action_batch)

        z_seq = torch.stack(z_seq, dim=0)
        action_seq = torch.stack(action_seq, dim=0)
        return z_seq, action_seq

    def update(self, step):
        metrics = dict()
        std = utils.linear_schedule(self.expl_start, self.expl_end, self.expl_duration, step)

        if step % self.log_interval == 0 and self.log_wandb:
            log = True 
        else:
            log = False 

        self.update_representation(std, step, log, metrics)
        self.update_rest(std, step, log, metrics)

        if step%self.target_update_interval==0:
            utils.soft_update(self.encoder_target, self.encoder, self.tau)
            utils.soft_update(self.critic_target, self.critic, self.tau)

        if log: 
            wandb.log(metrics, step=step)  

    def update_representation(self, std, step, log, metrics):    
        state_seq, action_seq, reward_seq, next_state_seq, done_seq = self.env_buffer.sample_seq(self.seq_len, self.batch_size)
        state_seq = torch.FloatTensor(state_seq).to(self.device)
        next_state_seq = torch.FloatTensor(next_state_seq).to(self.device)
        action_seq = torch.FloatTensor(action_seq).to(self.device)
        reward_seq = torch.FloatTensor(reward_seq).to(self.device)
        done_seq = torch.FloatTensor(done_seq).to(self.device)  
        
        alm_loss = self.alm_loss(state_seq, action_seq, next_state_seq, std, step, False, metrics)    
            
        self.model_opt.zero_grad()
        alm_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(utils.get_parameters(self.world_model_list), max_norm=self.max_grad_norm, error_if_nonfinite=True)
        self.model_opt.step()

        if log:
            metrics['alm_loss'] = alm_loss.item()
            metrics['model_grad_norm'] = model_grad_norm.item()

    def alm_loss(self, state_seq, action_seq, next_state_seq, std, step, log, metrics):
        z_dist = self.encoder(state_seq[0])
        z_batch = z_dist.rsample()
        alm_loss = 0
        log_q = log 
        for t in range(self.seq_len):
            if t > 0 and log:
                log = False 

            kl_loss, z_next_prior_batch = self._kl_loss(z_batch, action_seq[t], next_state_seq[t], log, metrics)
            reward_loss = self._alm_reward_loss(z_batch, action_seq[t], log, metrics)
            alm_loss += (kl_loss - reward_loss)

            z_batch = z_next_prior_batch

        Q = self._alm_value_loss(z_batch, std, log_q, metrics)
        alm_loss += (-Q)
        
        return alm_loss.mean()

    def _kl_loss(self, z_batch, action_batch, next_state_batch, log, metrics):
        z_next_prior_dist = self.model(z_batch, action_batch)
        with torch.no_grad():    
            z_next_dist = self.encoder_target(next_state_batch)
        kl = td.kl_divergence(z_next_prior_dist, z_next_dist).unsqueeze(-1)

        if log:
            metrics['kl'] = kl.mean().item()
            metrics['prior_entropy'] = z_next_prior_dist.entropy().mean()
            metrics['posterior_entropy'] = z_next_dist.entropy().mean()

        return kl, z_next_prior_dist.rsample()

    def _alm_reward_loss(self, z_batch, action_batch, log, metrics):
        with utils.FreezeParameters(self.reward_list):
            reward = self.reward(z_batch, action_batch)

        if log:
            metrics['alm_reward_batch'] = reward.mean().item()

        return reward

    def _alm_value_loss(self, z_batch, std, log, metrics):
        with torch.no_grad():  
            action_dist = self.actor(z_batch, std)
            action_batch = action_dist.sample(clip=self.stddev_clip)

        with utils.FreezeParameters(self.critic_list):
            Q1, Q2 = self.critic(z_batch, action_batch)
            Q = torch.min(Q1, Q2)

        if log:
            metrics['alm_q_batch'] = Q.mean().item()
        return Q

    def update_rest(self, std, step, log, metrics):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.env_buffer.sample(self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)  
        discount_batch = self.gamma*(1-done_batch)

        with torch.no_grad():
            z_dist = self.encoder_target(state_batch)
            z_next_prior_dist = self.model(z_dist.sample(), action_batch)
            z_next_dist = self.encoder_target(next_state_batch)

        #update reward and classifier
        self.update_reward(z_dist.sample(), action_batch, reward_batch, z_next_dist.sample(), z_next_prior_dist.sample(), log, metrics)
        
        #update critic
        self.update_critic(z_dist.sample(), action_batch, reward_batch, z_next_dist.sample(), discount_batch, std, log, metrics)
        
        #update actor
        self.update_actor(z_dist.sample(), std, log, metrics) 
        
        if log:
            metrics['seq_len'] = self.seq_len
            metrics['mse_loss'] = F.mse_loss(z_next_dist.sample(), z_next_prior_dist.sample()).item()
    
    def update_reward(self, z_batch, action_batch, reward_batch, z_next_batch, z_next_prior_batch, log, metrics):
        reward_loss = self._extrinsic_reward_loss(z_batch, action_batch, reward_batch.unsqueeze(-1), log, metrics)
        classifier_loss = self._intrinsic_reward_loss(z_batch, action_batch, z_next_batch, z_next_prior_batch, log, metrics)
        self.reward_opt.zero_grad()
        (reward_loss + classifier_loss).backward()
        reward_grad_norm = torch.nn.utils.clip_grad_norm_(utils.get_parameters(self.reward_list), max_norm=self.max_grad_norm, error_if_nonfinite=True)
        self.reward_opt.step()
        
        if log:
            metrics['reward_grad_norm'] = reward_grad_norm.mean()
            
    def _extrinsic_reward_loss(self, z_batch, action_batch, reward_batch, log, metrics):
        reward_pred = self.reward(z_batch, action_batch)
        reward_loss = F.mse_loss(reward_pred, reward_batch)

        if log:
            metrics['reward_loss'] = reward_loss.item()
            metrics['min_true_reward'] = torch.min(reward_batch).item()
            metrics['max_true_reward'] = torch.max(reward_batch).item()
            metrics['mean_true_reward'] = torch.mean(reward_batch).item()

        return reward_loss

    def _intrinsic_reward_loss(self, z, action_batch, z_next, z_next_prior, log, metrics):
        ip_batch_shape = z.shape[0]
        false_batch_idx = np.random.choice(ip_batch_shape, ip_batch_shape//2, replace=False)
        z_next_target = z_next 
        z_next_target[false_batch_idx] = z_next_prior[false_batch_idx]

        labels = torch.ones(ip_batch_shape, dtype=torch.long, device=self.device)
        labels[false_batch_idx] = 0.0

        logits = self.classifier(z, action_batch, z_next_target)
        classifier_loss = nn.CrossEntropyLoss()(logits, labels)

        if log:
            metrics['classifier_loss'] = classifier_loss.item()

        return classifier_loss

    def update_critic(self, z_batch, action_batch, reward_batch, z_next_batch, discount_batch, std, log, metrics):
        with torch.no_grad():    
            next_action_dist = self.actor(z_next_batch, std)
            next_action_batch = next_action_dist.sample(clip=self.stddev_clip)

            target_Q1, target_Q2 = self.critic_target(z_next_batch, next_action_batch)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch.unsqueeze(-1) + discount_batch.unsqueeze(-1)*(target_V)
            
        Q1, Q2 = self.critic(z_batch, action_batch)
        critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))/2   

        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(utils.get_parameters(self.critic_list), max_norm=self.max_grad_norm, error_if_nonfinite=True)
        self.critic_opt.step()

        if log:
            metrics['mean_q_target'] = torch.mean(target_Q).item()
            metrics['variance_q_target'] = torch.var(target_Q).item()
            metrics['min_q_target'] = torch.min(target_Q).item()
            metrics['max_q_target'] = torch.max(target_Q).item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['critic_grad_norm'] = critic_grad_norm.mean()
    
    def update_actor(self, z_batch, std, log, metrics):
        actor_loss = self._lambda_svg_loss(z_batch, std, log, metrics)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(utils.get_parameters(self.actor_list), max_norm=self.max_grad_norm, error_if_nonfinite=True)
        self.actor_opt.step()

        if log:
            metrics['actor_grad_norm'] = actor_grad_norm.mean()

    def _td_loss(self, z_batch, std, log, metrics):
        with utils.FreezeParameters([self.critic]):
            action_dist = self.actor(z_batch, std)
            action_batch = action_dist.sample(clip=self.stddev_clip)
            Q1, Q2 = self.critic(z_batch, action_batch)
            Q = torch.min(Q1, Q2)
        
        actor_loss = -Q.mean()
        return actor_loss

    def _lambda_svg_loss(self, z_batch, std, log, metrics):
        actor_loss = 0
        z_seq, action_seq = self._rollout_imagination(z_batch, std)

        with utils.FreezeParameters([self.model, self.reward, self.classifier, self.critic]):
            reward = self.reward(z_seq[:-1], action_seq[:-1])
            kl_reward = self.classifier.get_reward(z_seq[:-1], action_seq[:-1], z_seq[1:].detach())
            discount = self.gamma * torch.ones_like(reward)
            q_values_1, q_values_2 = self.critic(z_seq, action_seq.detach())      
            q_values = torch.min(q_values_1, q_values_2) 
            
            returns = lambda_returns(reward+self.lambda_cost*kl_reward, discount, q_values[:-1], q_values[-1], self.seq_len)
            discount = torch.cat([torch.ones_like(discount[:1]), discount])
            discount = torch.cumprod(discount[:-1], 0)
            actor_loss = -torch.mean(discount * returns)
            
        if log:
            metrics['min_imag_reward'] = torch.min(reward).item()
            metrics['max_imag_reward'] = torch.max(reward).item()
            metrics['mean_imag_reward'] = torch.mean(reward).item()
            metrics['min_imag_kl_reward'] = torch.min(kl_reward).item()
            metrics['max_imag_kl_reward'] = torch.max(kl_reward).item()
            metrics['mean_imag_kl_reward'] = torch.mean(kl_reward).item()
            metrics['actor_loss'] = actor_loss.item()
            metrics['lambda_cost'] = self.lambda_cost
            metrics['min_imag_value'] = torch.min(q_values).item()
            metrics['max_imag_value'] = torch.max(q_values).item()
            metrics['mean_imag_value'] = torch.mean(q_values).item()
            metrics['action_std'] = std 

        return actor_loss
        
    def _rollout_imagination(self, z_batch, std):
        z_seq = [z_batch]
        action_seq = []
        with utils.FreezeParameters([self.model]):
            for t in range(self.seq_len):
                action_dist = self.actor(z_batch.detach(), std)
                action_batch = action_dist.sample(self.stddev_clip)
                z_batch = self.model(z_batch, action_batch).rsample()
                action_seq.append(action_batch)
                z_seq.append(z_batch)

            action_dist = self.actor(z_batch.detach(), std)
            action_batch = action_dist.sample(self.stddev_clip)
            action_seq.append(action_batch)

        z_seq = torch.stack(z_seq, dim=0)
        action_seq = torch.stack(action_seq, dim=0)
        return z_seq, action_seq

    def _init_networks(self, num_states, num_actions, latent_dims, hidden_dims, model_hidden_dims, identity_encoder):
        if identity_encoder:
            self.encoder = IdentityEncoder(num_states).to(self.device)
            self.encoder_target = IdentityEncoder(num_states).to(self.device)
            utils.hard_update(self.encoder_target, self.encoder)
            latent_dims = num_states
        else:
            self.encoder = Encoder(num_states, hidden_dims, latent_dims).to(self.device)
            self.encoder_target = Encoder(num_states, hidden_dims, latent_dims).to(self.device)
            utils.hard_update(self.encoder_target, self.encoder)

        self.model = ModelPrior(latent_dims, num_actions, model_hidden_dims).to(self.device)
        self.reward = RewardPrior(latent_dims, hidden_dims, num_actions).to(self.device)
        self.classifier = Discriminator(latent_dims, hidden_dims, num_actions).to(self.device)

        self.critic = Critic(latent_dims, hidden_dims, num_actions).to(self.device)
        self.critic_target = Critic(latent_dims, hidden_dims, num_actions).to(self.device)
        utils.hard_update(self.critic_target, self.critic)

        self.actor = Actor(latent_dims, hidden_dims, num_actions, self.action_low, self.action_high).to(self.device)

        self.world_model_list = [self.model, self.encoder]
        self.reward_list = [self.reward, self.classifier]
        self.actor_list = [self.actor]
        self.critic_list = [self.critic]

    def _init_optims(self, lr):
        self.model_opt = torch.optim.Adam(utils.get_parameters(self.world_model_list), lr=lr['model'])
        self.reward_opt = torch.optim.Adam(utils.get_parameters(self.reward_list), lr=lr['reward'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr['actor'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr['critic'])

    def get_save_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            "model": self.model.state_dict(),
            "reward": self.reward.state_dict(),
            "classifier": self.classifier.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target":self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.encoder.load_state_dict(saved_dict["encoder"])
        self.encoder_target.load_state_dict(saved_dict["encoder_target"])
        self.model.load_state_dict(saved_dict["model"])
        self.reward.load_state_dict(saved_dict["reward"])
        self.classifier.load_state_dict(saved_dict["classifier"])
        self.critic.load_state_dict(saved_dict["critic"])
        self.critic_target.load_state_dict(saved_dict['critic_target'])
        self.actor.load_state_dict(saved_dict['actor'])
        
def lambda_returns(reward, discount, q_values, bootstrap, horizon, lambda_=0.95):
    next_values = torch.cat([q_values[1:], bootstrap[None]], 0)
    inputs = reward + discount * next_values * (1 - lambda_)
    last = bootstrap
    returns = []
    for t in reversed(range(horizon)):
        inp, disc = inputs[t], discount[t]
        last = inp + disc*lambda_*last
        returns.append(last)
    
    returns = torch.stack(list(reversed(returns)), dim=0)
    return returns