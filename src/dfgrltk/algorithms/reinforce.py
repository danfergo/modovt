import torch

from src.dfgrltk.algorithms.rl import RL
from src.dfgrltk.policies.pg import PGPolicy

from src.dfgrltk.replay_buffer import ReplayBuffer

import numpy as np

class Reinforce(RL):
    """
        Policy gradient / actor-based.
        Trained: on policy.
        Action-space: continuous
    """

    def __init__(self, replay_buffer: ReplayBuffer, policy: PGPolicy, **kwargs):
        super().__init__(replay_buffer, **kwargs)
        self.policy = policy
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def update(self):
        for t, (state, action, r) in enumerate(self.rb.all('state', 'action', 'reward')):
            sum_discounted_rewards = self.sdr(t)
            log_prob = self.policy.log_prob(state, action)

            # the negative sign is due to the fact that we want to
            # maximize the sdr, and the optimizer minimizes
            loss = - log_prob * sum_discounted_rewards

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def online_learn(self, on_reset=None, on_step=None, n_steps=50000):
        step = 0

        while step <= n_steps:
            self.rb.start_episode()

            obs = on_reset()
            done = False

            while not done:
                action = self.policy(obs)
                next_obs, rew, done = on_step(action.item())

                self.rb.append({
                    'state': torch.tensor(obs),
                    'action': action,
                    'reward': rew
                })

                obs = next_obs
                step += 1

            # learn policy.
            self.update()
            print('reward:', np.sum(self.rb.all('reward')), len(self.rb))

            # env.render()

