import gym

from src.dfgrltk.algorithms.reinforce import Reinforce
from src.dfgrltk.nn.mlp import mlp
from src.dfgrltk.policies.pg import PGPolicy

from src.dfgrltk.replay_buffer import ReplayBuffer

env = gym.make('CartPole-v1', render_mode='human')
# env = gym.make('LunarLander-v2', render_mode='human')

policy = PGPolicy(
    actor=mlp((env.observation_space.shape[0], 64, 64, env.action_space.n))
)

rb = ReplayBuffer(max_episodes=1)
rl = Reinforce(rb, policy)
rl.online_learn(
    on_reset=lambda: env.reset()[0],
    on_step=lambda a: env.step(a)[0:3],
)
