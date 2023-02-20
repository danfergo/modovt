import gym
import numpy as np
from gym import spaces


class SBGymAdapter(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, name):
        super().__init__()

        self.env = gym.make(name, render_mode='human')

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = self.env.action_space  # spaces.Discrete(N_DISCRETE_ACTIONS)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = self.env.observation_space
        #  spaces.Box(low=0, high=255,  shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        obs, rew, done, _, info = self.env.step(action)
        print(info)
        return obs, rew, done, info

    def reset(self):
        obs, _ = self.env.reset()
        return obs  # reward, done, info can't be included

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()
