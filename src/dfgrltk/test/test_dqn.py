import random

import gym
import numpy as np
import torch

from src.dfgrltk.algorithms.dqn import DQN

from src.dfgrltk.nn.mlp import mlp
from src.dfgrltk.policies.av import AVPolicy
from src.dfgrltk.policies.egreedy import EGreedyPolicy

from src.dfgrltk.replay_buffer import ReplayBuffer

env = gym.make('CartPole-v1', render_mode='human')

av = mlp((env.observation_space.shape[0], 64, 64, env.action_space.n))

policy = AVPolicy(value_fn=mlp((env.observation_space.shape[0], 64, 64, env.action_space.n)))
target_policy = AVPolicy(value_fn=mlp((env.observation_space.shape[0], 64, 64, env.action_space.n)))
behaviour_policy = EGreedyPolicy(greedy=policy)

rb = ReplayBuffer(max_episodes=99999)
rl = DQN(rb, policy, target_policy)
