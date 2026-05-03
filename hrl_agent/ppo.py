from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import dm_env
import numpy as np
import torch
import torch.nn as nn
from dm_env import specs
from torch.distributions import Categorical


def _prod(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _flatten_observation(obs) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _infer_obs_dim(obs_spec) -> int:
    if isinstance(obs_spec, dict):
        return sum(_prod(spec.shape) for spec in obs_spec.values())
    return _prod(obs_spec.shape)


def _extract_discrete_num_actions(action_spec: specs.DiscreteArray) -> int:
    if hasattr(action_spec, "num_values"):
        return int(action_spec.num_values)
    if hasattr(action_spec, "maximum") and hasattr(action_spec, "minimum"):
        return int(action_spec.maximum - action_spec.minimum + 1)
    raise ValueError("Could not infer the number of discrete actions from action_spec")


def _extract_action_minimum(action_spec: specs.DiscreteArray) -> int:
    minimum = getattr(action_spec, "minimum", 0)
    if np.isscalar(minimum):
        return int(minimum)
    return int(np.asarray(minimum).item())


@dataclass
class PPOConfig:
    num_steps: int = 256
    num_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    hidden_dim: int = 256
    device: str = "cuda"


class SyncVectorDMEnv:
    """Synchronous vector wrapper for dm_env.Environment."""

    def __init__(self, env_fns: Sequence[Callable[[], dm_env.Environment]]):
        if not env_fns:
            raise ValueError("env_fns must contain at least one environment factory")
        self._envs = [fn() for fn in env_fns]
        self.num_envs = len(self._envs)
        self._action_spec = self._envs[0].action_spec()
        self._observation_spec = self._envs[0].observation_spec()
        self._action_minimum = _extract_action_minimum(self._action_spec)

        if not isinstance(self._action_spec, specs.DiscreteArray):
            raise TypeError(
                "PPO trainer currently supports only specs.DiscreteArray actions"
            )

        self._last_time_steps = [None for _ in range(self.num_envs)]
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def observation_spec(self):
        return self._observation_spec

    def reset(self) -> np.ndarray:
        observations = []
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        for idx, env in enumerate(self._envs):
            ts = env.reset()
            self._last_time_steps[idx] = ts
            observations.append(_flatten_observation(ts.observation))
        return np.stack(observations, axis=0)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got leading dim {actions.shape[0]}"
            )

        next_obs = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        discounts = np.ones(self.num_envs, dtype=np.float32)
        episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

        for idx, (env, action) in enumerate(zip(self._envs, actions)):
            env_action = np.asarray(
                int(action) + self._action_minimum, dtype=self._action_spec.dtype
            )
            ts = env.step(env_action)
            reward = 0.0 if ts.reward is None else float(ts.reward)
            discount = 1.0 if ts.discount is None else float(ts.discount)

            self._episode_returns[idx] += reward
            self._episode_lengths[idx] += 1

            rewards[idx] = reward
            discounts[idx] = discount
            dones[idx] = float(ts.last())

            if ts.last():
                episode_returns[idx] = self._episode_returns[idx]
                episode_lengths[idx] = self._episode_lengths[idx]
                ts = env.reset()
                self._episode_returns[idx] = 0.0
                self._episode_lengths[idx] = 0

            self._last_time_steps[idx] = ts
            next_obs.append(_flatten_observation(ts.observation))

        infos = {
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "discounts": discounts,
        }
        return np.stack(next_obs, axis=0), rewards, dones, infos


class DiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logits: torch.Tensor = self.policy_head(features)
        value: torch.Tensor = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor):
        _, value = self.forward(obs)
        return value


class PPOTrainer:
    def __init__(
        self,
        env_fns: Sequence[Callable[[], dm_env.Environment]],
        config: PPOConfig | None = None,
        model: nn.Module | None = None,
    ):
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.envs = SyncVectorDMEnv(env_fns)

        self.obs_dim = _infer_obs_dim(self.envs.observation_spec)
        self.num_actions = _extract_discrete_num_actions(self.envs.action_spec)
        self.num_envs = self.envs.num_envs
        self.batch_size = self.cfg.num_steps * self.num_envs

        if self.batch_size < self.cfg.minibatch_size:
            raise ValueError("minibatch_size must be <= num_steps * num_envs")

        if model is None:
            model = DiscreteActorCritic(
                obs_dim=self.obs_dim,
                num_actions=self.num_actions,
                hidden_dim=self.cfg.hidden_dim,
            )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

        self._next_obs = self.envs.reset()
        self._global_step = 0
        self._num_updates = 0

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def num_updates(self) -> int:
        return self._num_updates

    def _allocate_rollout_storage(self):
        obs = torch.zeros(
            (self.cfg.num_steps, self.num_envs, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.zeros(
            (self.cfg.num_steps, self.num_envs), dtype=torch.long, device=self.device
        )
        log_probs = torch.zeros_like(actions, dtype=torch.float32)
        rewards = torch.zeros_like(log_probs)
        dones = torch.zeros_like(log_probs)
        values = torch.zeros_like(log_probs)
        return obs, actions, log_probs, rewards, dones, values

    def collect_rollout(self):
        obs_buf, actions_buf, log_probs_buf, rewards_buf, dones_buf, values_buf = (
            self._allocate_rollout_storage()
        )
        episode_returns = []
        episode_lengths = []

        for step in range(self.cfg.num_steps):
            obs_tensor = torch.as_tensor(
                self._next_obs, dtype=torch.float32, device=self.device
            )
            obs_buf[step] = obs_tensor

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs_tensor)

            actions_buf[step] = action
            log_probs_buf[step] = log_prob
            values_buf[step] = value

            next_obs, rewards, dones, infos = self.envs.step(action.cpu().numpy())
            rewards_buf[step] = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            dones_buf[step] = torch.as_tensor(
                dones, dtype=torch.float32, device=self.device
            )

            finished_returns = infos["episode_returns"]
            finished_lengths = infos["episode_lengths"]
            for ret, length in zip(finished_returns, finished_lengths):
                if length > 0:
                    episode_returns.append(float(ret))
                    episode_lengths.append(int(length))

            self._next_obs = next_obs
            self._global_step += self.num_envs

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(
                self._next_obs, dtype=torch.float32, device=self.device
            )
            next_value = self.model.get_value(next_obs_tensor)

        advantages = torch.zeros_like(rewards_buf)
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for step in reversed(range(self.cfg.num_steps)):
            if step == self.cfg.num_steps - 1:
                next_non_terminal = 1.0 - dones_buf[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones_buf[step]
                next_values = values_buf[step + 1]

            delta = (
                rewards_buf[step]
                + self.cfg.gamma * next_values * next_non_terminal
                - values_buf[step]
            )
            last_gae = (
                delta
                + self.cfg.gamma
                * self.cfg.gae_lambda
                * next_non_terminal
                * last_gae
            )
            advantages[step] = last_gae

        returns = advantages + values_buf

        batch = {
            "obs": obs_buf.reshape(-1, self.obs_dim),
            "actions": actions_buf.reshape(-1),
            "log_probs": log_probs_buf.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values_buf.reshape(-1),
        }
        stats = {
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
        }
        return batch, stats

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["values"]

        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_minibatches = 0

        for _ in range(self.cfg.num_epochs):
            indices = torch.randperm(self.batch_size, device=self.device)
            for start in range(0, self.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = indices[start:end]

                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    obs[mb_inds], actions[mb_inds]
                )
                log_ratio = new_log_probs - old_log_probs[mb_inds]
                ratio = log_ratio.exp()

                mb_advantages = advantages[mb_inds]
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred = new_values
                value_pred_clipped = old_values[mb_inds] + (
                    value_pred - old_values[mb_inds]
                ).clamp(-self.cfg.clip_ratio, self.cfg.clip_ratio)
                value_loss_unclipped = (value_pred - returns[mb_inds]) ** 2
                value_loss_clipped = (value_pred_clipped - returns[mb_inds]) ** 2
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped, value_loss_clipped
                ).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().abs().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_kl += approx_kl
                num_minibatches += 1

        self._num_updates += 1
        return {
            "policy_loss": total_policy_loss / max(1, num_minibatches),
            "value_loss": total_value_loss / max(1, num_minibatches),
            "entropy": total_entropy / max(1, num_minibatches),
            "approx_kl": total_kl / max(1, num_minibatches),
        }

    def train(self, total_timesteps: int, log_fn: Callable[[dict], None] | None = None):
        while self.global_step < total_timesteps:
            batch, rollout_stats = self.collect_rollout()
            update_stats = self.update(batch)

            metrics = {
                "global_step": self.global_step,
                "num_updates": self.num_updates,
                **update_stats,
            }
            if rollout_stats["episode_returns"]:
                metrics["episode_return"] = float(
                    np.mean(rollout_stats["episode_returns"])
                )
                metrics["episode_length"] = float(
                    np.mean(rollout_stats["episode_lengths"])
                )

            if log_fn is not None:
                log_fn(metrics)

    def act(self, observation, deterministic: bool = True) -> int:
        obs = _flatten_observation(observation)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        return int(action.item()) + self.envs._action_minimum

    def save(self, path: str):
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "config": self.cfg,
        }
        torch.save(payload, path)

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._global_step = int(payload["global_step"])
        self._num_updates = int(payload["num_updates"])
