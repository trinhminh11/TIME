import dm_env
import numpy as np
from dm_env import specs

from agent.diayn import DIAYNAgent
from agent.ours import OursAgent
from dmc import ExtendedTimeStepWrapper, ExtendedTimeStep
import utils
import torch


# wrapper for HRL env that actions = choose skill, and skill is one-hot encoded vector that gets fed into the agent's policy as meta input. The wrapper will also need to keep track of the current timestep in the episode, so that it can update the skill every N steps (where N is a hyperparameter).
class DiscreteSkillDiscoveryEnvWrapper(dm_env.Environment):
    def __init__(self, env: ExtendedTimeStepWrapper, agent: DIAYNAgent | OursAgent, t: int = 1):
        self._env = env
        self._agent = agent
        self.skill_dim = agent.skill_dim
        self._action_spec = specs.DiscreteArray(
            self.skill_dim, dtype=np.int32, name="skill"
        )
        self._observation_spec = env.observation_spec()

        self._last_timestep = None
        self.t = t

    def reset(self):
        timestep = self._env.reset()
        self._last_timestep = timestep
        return ExtendedTimeStep(
            step_type=self._last_timestep.step_type,
            reward=self._last_timestep.reward,
            discount=self._last_timestep.discount,
            observation=self.observation_transform_for_env(self._last_timestep.observation),
            action=self._last_timestep.action,
            info=self._last_timestep.info,
        )

    def observation_transform_for_agent(self, observation):
        return observation

    def observation_transform_for_env(self, observation):
        return observation


    def step(self, action):
        meta = {"skill": np.zeros(self._agent.skill_dim, dtype=np.float32)}
        meta["skill"][int(action)] = 1.0

        reward = 0

        with utils.eval_mode(self._agent):
            with torch.no_grad():
                for _ in range(self.t):
                    action = self._agent.act(
                        self.observation_transform_for_agent(self._last_timestep.observation), meta, 1000000, eval_mode=True
                    )

                    timestep = self._env.step(action)
                    self._last_timestep = ExtendedTimeStep(
                        step_type=timestep.step_type,
                        reward=timestep.reward,
                        discount=timestep.discount,
                        observation=self.observation_transform_for_env(timestep.observation),
                        action=action,
                        info=timestep.info,
                    )
                    reward += timestep.reward
                    if self._last_timestep.last():
                        break

        return ExtendedTimeStep(
            step_type=self._last_timestep.step_type,
            reward=reward,
            discount=self._last_timestep.discount,
            observation=self.observation_transform_for_env(self._last_timestep.observation),
            action=self._last_timestep.action,
            info=self._last_timestep.info,
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class AntMazeFromAntPretrainedEnvWrapper(DiscreteSkillDiscoveryEnvWrapper):
    def __init__(self, env: ExtendedTimeStepWrapper, agent: DIAYNAgent | OursAgent, t: int = 1):
        super().__init__(env, agent, t)

        self._observation_spec = specs.Array((env.observation_spec().shape[0]-2, ), dtype=np.float32, name="observation")

    def observation_transform_for_agent(self, observation):
        return observation[:-2] # exclude the x, y of goals

    def observation_transform_for_env(self, observation):
        return observation[:-2] # exclude the x, y of current position of the agent, since the low-level agent was pretrained without that information

class Goal1DEnvWrapper(DiscreteSkillDiscoveryEnvWrapper):
    def __init__(self, env: ExtendedTimeStepWrapper, x_lim: tuple[float, float], acceptance_radius: float, agent: DIAYNAgent | OursAgent, t: int = 1):
        super().__init__(env, agent, t)

        self._x_lim = x_lim
        self._acceptance_radius = acceptance_radius
        self._observation_spec = specs.Array((env.observation_spec().shape[0]+1, ), dtype=np.float32, name="observation")
        self._action_spec = env.action_spec()

        self._goal = None

    def get_goal(self):
        if self._goal is None:
            self._goal = np.array([np.random.uniform(*self._x_lim)], dtype=np.float32)

        return self._goal

    def observation_transform_for_agent(self, observation):
        return np.concatenate([observation, self.get_goal()])

    def reset(self):
        self._goal = None
        return super().reset()

    def step(self, action):
        meta = {"skill": np.zeros(self._agent.skill_dim, dtype=np.float32)}
        meta["skill"][int(action)] = 1.0

        reward = 0

        with utils.eval_mode(self._agent):
            with torch.no_grad():
                for _ in range(self.t):
                    action = self._agent.act(
                        self.observation_transform_for_agent(self._last_timestep.observation), meta, 1000000, eval_mode=True
                    )

                    timestep = self._env.step(action)

                    x = self._env.physics.data.xpos["torso"][0]

                    distance = np.linalg.norm(np.array([x]) - self.get_goal())
                    cur_reward = np.exp(-distance)

                    if distance <= self._acceptance_radius:
                        step_type = dm_env.StepType.LAST
                    else:
                        step_type = timestep.step_type

                    self._last_timestep = ExtendedTimeStep(
                        step_type=step_type,
                        reward=cur_reward,
                        discount=timestep.discount,
                        observation=self.observation_transform_for_env(timestep.observation),
                        action=action,
                        info=timestep.info,
                    )


                    # reward += timestep.reward
                    if self._last_timestep.last():
                        break

        return ExtendedTimeStep(
            step_type=self._last_timestep.step_type,
            reward=reward,
            discount=self._last_timestep.discount,
            observation=self.observation_transform_for_env(self._last_timestep.observation),
            action=self._last_timestep.action,
            info=self._last_timestep.info,
        )


class Goal2DEnvWrapper(dm_env.Environment):
    def __init__(self, env: ExtendedTimeStepWrapper, x_lim: tuple[float, float], y_lim: tuple[float, float], acceptance_radius: float):
        self._env = env
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._acceptance_radius = acceptance_radius
        self._observation_spec = specs.Array((env.observation_spec().shape[0]+2, ), dtype=np.float32, name="observation")
        self._action_spec = env.action_spec()

        self._goal = None

    def get_goal(self):
        if self._goal is None:
            self._goal = np.array([np.random.uniform(*self._x_lim), np.random.uniform(*self._y_lim)], dtype=np.float32)

        return self._goal

    def reset(self):
        self._goal = None
        return super().reset()

    def step(self, action):
        meta = {"skill": np.zeros(self._agent.skill_dim, dtype=np.float32)}
        meta["skill"][int(action)] = 1.0

        reward = 0

        with utils.eval_mode(self._agent):
            with torch.no_grad():
                for _ in range(self.t):
                    action = self._agent.act(
                        self.observation_transform_for_agent(self._last_timestep.observation), meta, 1000000, eval_mode=True
                    )

                    timestep = self._env.step(action)

                    x, y = self._env.physics.data.xpos["torso"][:2]

                    distance = np.linalg.norm(np.array([x, y]) - self.get_goal())

                    cur_reward = np.exp(-distance)

                    if distance <= self._acceptance_radius:
                        step_type = dm_env.StepType.LAST
                    else:
                        step_type = timestep.step_type

                    self._last_timestep = ExtendedTimeStep(
                        step_type=step_type,
                        reward=cur_reward,
                        discount=timestep.discount,
                        observation=self.observation_transform_for_env(timestep.observation),
                        action=action,
                        info=timestep.info,
                    )


                    # reward += timestep.reward
                    if self._last_timestep.last():
                        break

        return ExtendedTimeStep(
            step_type=self._last_timestep.step_type,
            reward=reward,
            discount=self._last_timestep.discount,
            observation=self.observation_transform_for_env(self._last_timestep.observation),
            action=self._last_timestep.action,
            info=self._last_timestep.info,
        )

