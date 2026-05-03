import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch

import dmc
import utils
from dmc_benchmark import PRIMAL_TASKS
from hrl_agent.ppo import PPOConfig, PPOTrainer
from logger import Logger
from sd_env import DiscreteSkillDiscoveryEnvWrapper
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        if cfg.agent.name not in {'diayn', 'ours'}:
            raise ValueError("train_hrl.py supports only agent=diayn or agent=ours")

        cfg.device = utils.resolve_device(cfg.device)
        cfg.agent.device = cfg.device
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        self.task = PRIMAL_TASKS[self.cfg.domain]

        # The low-level controller is instantiated from the original task env specs.
        base_env = dmc.make(self.task, cfg.obs_type, cfg.frame_stack,
                            cfg.action_repeat, cfg.seed)
        self.low_level_agent = make_agent(
            cfg.obs_type,
            base_env.observation_spec(),
            base_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )
        self.low_level_agent.train(False)

        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_low_level_snapshot()['agent']
            self.low_level_agent.init_from(pretrained_agent)

        def make_hrl_env(seed):
            env = dmc.make(self.task, cfg.obs_type, cfg.frame_stack,
                           cfg.action_repeat, seed)
            return DiscreteSkillDiscoveryEnvWrapper(env, self.low_level_agent)

        self.eval_env = make_hrl_env(cfg.seed)
        env_fns = [
            (lambda seed=cfg.seed + idx: make_hrl_env(seed))
            for idx in range(cfg.num_envs)
        ]

        ppo_cfg = PPOConfig(num_steps=cfg.ppo.num_steps,
                            num_epochs=cfg.ppo.num_epochs,
                            minibatch_size=cfg.ppo.minibatch_size,
                            gamma=cfg.discount,
                            gae_lambda=cfg.ppo.gae_lambda,
                            clip_ratio=cfg.ppo.clip_ratio,
                            value_coef=cfg.ppo.value_coef,
                            entropy_coef=cfg.ppo.entropy_coef,
                            learning_rate=cfg.ppo.learning_rate,
                            max_grad_norm=cfg.ppo.max_grad_norm,
                            normalize_advantages=cfg.ppo.normalize_advantages,
                            hidden_dim=cfg.ppo.hidden_dim,
                            device=cfg.device)
        self.agent = PPOTrainer(env_fns=env_fns, config=ppo_cfg)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_episode = 0
        self._snapshot_frames = sorted({int(frame) for frame in cfg.snapshots})
        self._next_snapshot_idx = 0

    @property
    def global_step(self):
        return self.agent.global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0.0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))

            while not time_step.last():
                action = self.agent.act(time_step.observation, deterministic=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        while train_until_step(self.global_step):
            self._maybe_save_snapshots()

            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            batch, rollout_stats = self.agent.collect_rollout()
            metrics = self.agent.update(batch)
            self.logger.log_metrics(metrics, self.global_frame, ty='train')

            num_finished = len(rollout_stats['episode_returns'])
            self._global_episode += num_finished

            elapsed_time, total_time = self.timer.reset()
            frames_in_rollout = self.cfg.ppo.num_steps * self.cfg.num_envs * self.cfg.action_repeat

            with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                log('fps', frames_in_rollout / max(elapsed_time, 1e-6))
                log('total_time', total_time)
                log('episode', self.global_episode)
                log('step', self.global_step)
                if rollout_stats['episode_returns']:
                    log('episode_reward',
                        float(np.mean(rollout_stats['episode_returns'])))
                    log('episode_length',
                        float(np.mean(rollout_stats['episode_lengths'])) *
                        self.cfg.action_repeat)

    def _maybe_save_snapshots(self):
        if self._next_snapshot_idx >= len(self._snapshot_frames):
            return

        next_frame = self._snapshot_frames[self._next_snapshot_idx]
        if self.global_frame >= next_frame:
            self.save_snapshot(next_frame)
            self._next_snapshot_idx += 1

    def save_snapshot(self, frame=None):
        if frame is None:
            frame = self.global_frame
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{frame}.pt'
        payload = {
            'hrl_model': self.agent.model.state_dict(),
            'hrl_optimizer': self.agent.optimizer.state_dict(),
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'config': self.cfg,
        }
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_low_level_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        snapshot_dir = (snapshot_base_dir / self.cfg.obs_type / self.cfg.domain /
                        self.cfg.agent.name)

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f, map_location=self.device)
            return payload

        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload

        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload

    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=self.device)
        self.agent.model.load_state_dict(payload['hrl_model'])
        self.agent.optimizer.load_state_dict(payload['hrl_optimizer'])
        self.agent._global_step = int(payload['global_step'])
        self._global_episode = int(payload['global_episode'])


@hydra.main(config_path='.', config_name='train_hrl')
def main(cfg):
    from train_hrl import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)
    workspace.train()


if __name__ == '__main__':
    main()
