import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


class OneStateDiscriminator(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, skill_dim))
        self.apply(utils.weight_init)

    def forward(self, obs):
        return self.net(obs)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1024):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, obs_dim, skill_dim, embed_dim, d_ff, num_heads,
                 num_layers, traj_len):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.state_proj = nn.Linear(obs_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = SinusoidalPositionalEncoding(embed_dim, traj_len + 1)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                           nhead=num_heads,
                                           dim_feedforward=d_ff,
                                           dropout=0.0,
                                           activation='gelu',
                                           batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, skill_dim)
        self.apply(utils.weight_init)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, trajectory):
        batch_size = trajectory.shape[0]
        x = self.state_proj(trajectory)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.encoder(x)
        return self.head(x[:, 0])


class OursAgent(DDPGAgent):
    def __init__(self,
                 skill_dim,
                 traj_len,
                 intr_reward_scale,
                 state_log_coef,
                 state_mean_coef,
                 prior_coef,
                 disc_lr,
                 transformer_embed_dim,
                 transformer_d_ff,
                 transformer_heads,
                 transformer_layers,
                 update_encoder,
                 **kwargs):
        if kwargs['obs_type'] != 'states':
            raise NotImplementedError('OursAgent currently supports states only')

        self.skill_dim = skill_dim
        self.traj_len = traj_len
        self.intr_reward_scale = intr_reward_scale
        self.state_log_coef = state_log_coef
        self.state_mean_coef = state_mean_coef
        self.prior_coef = prior_coef
        self.update_encoder = update_encoder

        kwargs['meta_dim'] = skill_dim
        super().__init__(**kwargs)

        obs_dim = self.obs_dim - skill_dim
        self.state_disc = OneStateDiscriminator(obs_dim, skill_dim,
                                                kwargs['hidden_dim']).to(
                                                    kwargs['device'])
        self.traj_disc = TrajectoryDiscriminator(obs_dim, skill_dim,
                                                 transformer_embed_dim,
                                                 transformer_d_ff,
                                                 transformer_heads,
                                                 transformer_layers,
                                                 traj_len).to(
                                                     kwargs['device'])
        self.disc_opt = torch.optim.Adam(
            list(self.state_disc.parameters()) +
            list(self.traj_disc.parameters()),
            lr=disc_lr)
        self.state_disc.train()
        self.traj_disc.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'state_disc'):
            self.state_disc.train(training)
        if hasattr(self, 'traj_disc'):
            self.traj_disc.train(training)

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        return meta

    def update_discriminators(self, skill, next_obs, next_traj):
        metrics = dict()
        z = torch.argmax(skill, dim=1)

        state_logits = self.state_disc(next_obs)
        traj_logits = self.traj_disc(next_traj)
        state_loss = F.cross_entropy(state_logits, z)
        traj_loss = F.cross_entropy(traj_logits, z)
        disc_loss = state_loss + traj_loss

        self.disc_opt.zero_grad(set_to_none=True)
        disc_loss.backward()
        self.disc_opt.step()

        if self.use_tb or self.use_wandb:
            with torch.no_grad():
                state_acc = (state_logits.argmax(dim=1) == z).float().mean()
                traj_acc = (traj_logits.argmax(dim=1) == z).float().mean()
            metrics['ours_state_disc_loss'] = state_loss.item()
            metrics['ours_traj_disc_loss'] = traj_loss.item()
            metrics['ours_state_disc_acc'] = state_acc.item()
            metrics['ours_traj_disc_acc'] = traj_acc.item()

        return metrics

    def compute_intr_reward(self, skill, next_obs, next_traj):
        z = torch.argmax(skill, dim=1, keepdim=True)

        traj_logits = self.traj_disc(next_traj)
        log_traj = F.log_softmax(traj_logits, dim=-1)
        log_traj_z = log_traj.gather(1, z)

        state_logits = self.state_disc(next_obs)
        log_state = F.log_softmax(state_logits, dim=-1)
        log_state_z = log_state.gather(1, z)
        log_state_mean = log_state.mean(dim=-1, keepdim=True)

        prior = math.log(float(self.skill_dim))
        reward = (log_traj_z - self.state_log_coef * log_state_z +
                  self.state_mean_coef * log_state_mean -
                  self.prior_coef * prior)
        return reward * self.intr_reward_scale

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, extr_reward, discount, next_obs, next_traj, skill = (
            utils.to_torch(batch, self.device))

        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(
                self.update_discriminators(skill, next_obs.detach(),
                                           next_traj.detach()))
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs,
                                                       next_traj)
            reward = intr_reward
            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))
        metrics.update(self.update_actor(obs.detach(), step))

        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
