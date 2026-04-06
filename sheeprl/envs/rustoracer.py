from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
from gymnasium.core import RenderFrame
import numpy as np
from gymnasium import spaces



class RustoracerWrapper(gym.Env):
    def __init__(self, id: str, yaml_path: str, seed: Optional[int] = None) -> None:
        from rustoracerpy import RustoracerEnv
        env = RustoracerEnv(yaml = yaml_path, render_mode = "human")
        self._env = env
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low = self._env.single_observation_space.low,
                    high = self._env.single_observation_space.high,
                    shape = self._env.single_observation_space.shape,
                    dtype = np.float32,
                )
            }
        )
    
        self.action_space = self._env.single_action_space
        self.reward_range = (-np.inf, np.inf)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self._render_mode = "human"
        self._metadata = {"render_fps": 60}

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"state":obs[0].astype(np.float32)}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        actions = np.ascontiguousarray(np.expand_dims(action, axis = 0), dtype = np.float64)
        obs, reward, terminated, truncated, info = self._env.step(actions)
        return self._convert_obs(obs), float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self._env.reset(seed = seed)
        return self._convert_obs(obs), info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self._env.render()

    def close(self) -> None:
        try:
            self._env.close()
        except AttributeError:
            pass
