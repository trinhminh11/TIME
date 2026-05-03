import dm_env
import numpy as np
from dm_env import specs

from agent.diayn import DIAYNAgent
from agent.ours import OursAgent
from dmc import ExtendedTimeStepWrapper
import utils


# wrapper for HRL env that actions = choose skill, and skill is one-hot encoded vector that gets fed into the agent's policy as meta input. The wrapper will also need to keep track of the current timestep in the episode, so that it can update the skill every N steps (where N is a hyperparameter).
class DiscreteSkillDiscoveryEnvWrapper(dm_env.Environment):
    def __init__(self, env: ExtendedTimeStepWrapper, agent: DIAYNAgent | OursAgent):
        self._env = env
        self._agent = agent
        self.skill_dim = agent.skill_dim
        self._action_spec = specs.DiscreteArray(
            self.skill_dim, dtype=np.int32, name="skill"
        )
        self._observation_spec = env.observation_spec()

        self._last_timestep = None

    def reset(self):
        timestep = self._env.reset()
        self._last_timestep = timestep
        return timestep

    def step(self, action):
        meta = {"skill": np.zeros(self._agent.skill_dim, dtype=np.float32)}
        meta["skill"][int(action)] = 1.0

        with utils.eval_mode(self._agent):
            action = self._agent.act(
                self._last_timestep.observation, meta, 1000000, eval_mode=True
            )

        timestep = self._env.step(action)
        self._last_timestep = timestep
        return timestep

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)
