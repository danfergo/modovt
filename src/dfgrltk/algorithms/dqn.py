import torch

from src.dfgrltk.algorithms.rl import RL
from src.dfgrltk.replay_buffer import ReplayBuffer

import numpy as np


class DQN(RL):

    def __init__(self, rb: ReplayBuffer,
                 policy,
                 target_policy,
                 batch_size=32,
                 n_steps=100,
                 n_step_update=10):
        super().__init__(rb)

        self.updates_counter = 0
        self.n_step_update = n_step_update
        self.batch_size = batch_size

        self.policy = policy
        self.target_policy = target_policy

        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def update(self):
        self.updates_counter += 1

        batch = self.rb.sample(self.batch_size, 't', 'state', 'action', 'reward')
        rr = torch.tensor(np.array([r for t, s, a, r in batch]))
        ss = torch.tensor(np.array([s.numpy() for t, s, a, r in batch]))
        aa = torch.tensor(np.array([a.numpy() for t, s, a, r in batch]))
        print(ss, ss.grad)

        targets = rr + self.gamma * self.target_policy.max_value(ss)
        print(targets.size())

        q_values = self.policy.value(ss, aa)
        print(q_values.size())

        loss = (targets - q_values) ** 2

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.updates_counter >= self.n_step_update:
            # copy weights from behaviour to target.
            self.updates_counter = 0

    def online_learn(self, on_reset=None, on_step=None, n_steps=50000):

        step = 0
        max_steps = 50000

        while step <= max_steps:
            self.rb.start_episode()

            obs, = on_reset()
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
                if step > 100:
                    self.rl.update()

            print('reward:', np.sum(self.rb.all('reward')), len(self.rb))

            self.env.render()
