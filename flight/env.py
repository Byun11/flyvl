"""flybody vision-guided flight as a gymnasium env.

Observation: {"eyes": (2, 32, 32) uint8 luminance of left/right eye cameras,
              "proprio": every other walker observable concatenated (float32)}.
Action: [-1, 1]^12, rescaled to the task's action spec.
Seeds: training envs use seeds < 1000; validation uses 1000..1019 and test 2000..2019 (program.md).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class FlightEnv(gym.Env):
    def __init__(self, task: str = "bumps", seed: int = 0):
        from flybody.fly_envs import vision_guided_flight  # imported per process
        self._env = vision_guided_flight(bumps_or_trench=task, random_state=np.random.RandomState(seed))
        spec = self._env.action_spec()
        self._lo, self._hi = spec.minimum, spec.maximum
        self.action_space = gym.spaces.Box(-1, 1, spec.shape, np.float32)
        obs = self._convert(self._env.reset().observation)
        self.observation_space = gym.spaces.Dict({
            "eyes": gym.spaces.Box(0, 255, obs["eyes"].shape, np.uint8),
            "proprio": gym.spaces.Box(-np.inf, np.inf, obs["proprio"].shape, np.float32)})

    @staticmethod
    def _convert(o):
        eyes = np.stack([o["walker/left_eye"], o["walker/right_eye"]]).astype(np.float32) @ LUMA
        prop = [np.ravel(v) for k, v in o.items() if not k.endswith("_eye")]
        return {"eyes": eyes.astype(np.uint8), "proprio": np.concatenate(prop).astype(np.float32)}

    def reset(self, seed=None, options=None):
        return self._convert(self._env.reset().observation), {}

    def step(self, action):
        a = self._lo + (np.clip(action, -1, 1) + 1) * 0.5 * (self._hi - self._lo)
        ts = self._env.step(a)
        terminated = ts.last() and ts.discount == 0      # crash
        truncated = ts.last() and not terminated         # time limit
        return self._convert(ts.observation), float(ts.reward or 0.0), terminated, truncated, {}
