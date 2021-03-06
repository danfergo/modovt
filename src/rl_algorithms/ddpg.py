import copy
import os
import random
import sys
from math import exp

import numpy as np
import torch
from torch.autograd import Variable

from experimenter import experiment, e


class ReplayBuffer:
    def __init__(self, buffer=None):
        self.buffer = buffer or []

    def sample_batch(self, batch_size):
        samples_indexes = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in samples_indexes]

    def store(self, timestep):
        self.buffer.append(timestep)

    def __len__(self):
        return len(self.buffer)

    def save(self, p):
        buff = [(o.detach().numpy(), a.detach().numpy(), r, o1.detach().numpy(), d) for (o, a, r, o1, d) in self.buffer]
        np.save(p, np.array(buff, dtype=object))
        return self

    @staticmethod
    def load(p, ensure_exists=False):
        if os.path.exists(p):
            buff = np.load(p, allow_pickle=True)
            return ReplayBuffer([
                (
                    torch.tensor(o, dtype=torch.float),
                    torch.tensor(a, dtype=torch.float),
                    r,
                    torch.tensor(o1, dtype=torch.float),
                    d
                )
                for (o, a, r, o1, d) in buff
            ])
        elif ensure_exists:
            raise FileNotFoundError(p)
        else:
            return ReplayBuffer()

    @staticmethod
    def load_all(ps, ensure_exists=True):
        return [ReplayBuffer.load(p, ensure_exists) for p in ps]


@experiment("", {
    'n_exploration_episodes': 3,
    'n_episode_steps': 256,
    'n_start_episodes': 256,
    'n_update_batches': 64,
    'update_batch_size': 1024,
    'opt_lr': 0.001,
    'q_fn_lr': 0.001,
    'gamma': 0.99,
    'polyak_factor': 0.995,
    'dc_idx': sys.argv[1]
})
class DDPG:

    def __init__(self, q_function, policy):
        self.n_exploration_episodes = e.n_exploration_episodes
        self.n_episode_steps = e.n_episode_steps
        self.n_start_episodes = e.n_start_episodes
        self.n_update_batches = e.n_update_batches
        self.update_batch_size = e.update_batch_size
        self.gamma = e.gamma
        self.opt_lr = e.opt_lr
        self.q_fn_lr = e.q_fn_lr
        self.polyak_factor = e.polyak_factor

        self.q_fn = q_function
        self.policy = policy

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.opt_lr)
        self.q_fn_opt = torch.optim.Adam(self.q_fn.parameters(), lr=self.q_fn_lr)
        self.dc_idx = e.dc_idx

    # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/15
    def polyak_update(self, polyak_factor, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(polyak_factor * target_param.data + (1.0 - polyak_factor) * param.data)

    def update(self, policy_targ, q_fn_targ, replay_buffer):
        self.policy.train()

        for i in range(self.n_update_batches):
            batch = replay_buffer.sample_batch(self.update_batch_size)

            # compute targets
            rs = torch.tensor(np.array([-exp(1 / r) for o, a, r, o1, d in batch]))
            # print(rs)
            aa = torch.cat([a for o, a, r, o1, d in batch], dim=0)
            os = torch.cat([o for o, a, r, o1, d in batch], dim=0)
            o1s = torch.cat([o1 for o, a, r, o1, d in batch], dim=0)

            targets = rs + q_fn_targ(o1s, policy_targ(o1s)) * self.gamma
            # targets = [target_fn(timestep) for timestep in batch]

            # update q function
            loss = torch.mean((self.q_fn(os, aa) - targets) ** 2)
            loss = Variable(loss, requires_grad=True)
            self.q_fn_opt.zero_grad()
            loss.backward()
            self.q_fn_opt.step()

            # update policy, minus signal because we want to maximize
            loss = - torch.mean(self.q_fn(os, self.policy(os)))
            loss = Variable(loss, requires_grad=True)
            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()

            # update target networks with polyak update
            self.polyak_update(self.polyak_factor, q_fn_targ, self.q_fn)
            self.polyak_update(self.polyak_factor, policy_targ, self.policy)

    def gather_data(self, get_observation, get_reward, take_action, reset, replay_buffer):
        rr = 0
        self.policy.eval()
        with torch.no_grad():
            for k in range(self.n_exploration_episodes):
                reset()

                for i in range(self.n_episode_steps):
                    # get observation from the environment
                    o = get_observation()

                    # select action, following the main policy
                    o = torch.tensor(o, dtype=torch.float)
                    o = torch.unsqueeze(o, 0)  # make a batch of one.

                    # if epoch >= self.n_start_episodes:
                    epsilon = np.random.normal(0, 0.01, 6)
                    epsilon = torch.Tensor(np.array([epsilon]))
                    a = self.policy(o) + epsilon  # todo what to clip
                    # else:
                    # a = self.policy(o) * 0 + torch.Tensor(np.array([np.random.normal(0, 1, 6)]))

                    # a = clip(a.detach().numpy()[0])
                    take_action(a.detach().numpy()[0])

                    # get reward
                    r = get_reward()

                    # get next observation from the environment
                    o1 = get_observation()
                    o1 = torch.tensor(o1, dtype=torch.float)
                    o1 = torch.unsqueeze(o1, 0)  # make a batch of one.

                    d = i == self.n_episode_steps - 1
                    replay_buffer.store((o, a, r, o1, d))

                    rr = rr + r
            rr /= (self.n_episode_steps * self.n_exploration_episodes)
        return rr

    def run(self, get_observation, get_reward, take_action, reset, warm_startup=10000):

        # input: initial policy parameters theta, q-function parameters fi
        policy = self.policy
        q_function = self.q_fn
        replay_buffer = ReplayBuffer.load(e.out(str(self.dc_idx) + '_replay_buffer.npy'))

        # set target parameters equal to the main parameters (policy and q-function)
        policy_targ = copy.deepcopy(policy)
        q_function_targ = copy.deepcopy(q_function)

        epoch = 0

        while True:

            if (epoch > 9999 and epoch % 10000 == 0):
                print('explore..')
                rr = self.gather_data(get_observation, get_reward, take_action, reset, replay_buffer)

                e.emit('plot', {'name': 'reward', 'value': rr})
                e.emit('il_patch', {'name': 'meta', 'data': {
                    'epoch': epoch,
                    'replay_buffer_size': len(replay_buffer)
                }})
            if epoch % 100 == 0:
                reset()
            print('epoch:', epoch, ' ', (epoch > 99 and epoch % 100 == 0))
            self.update(policy_targ, q_function_targ, replay_buffer)

            epoch += 1
            # replay_buffer.save(e.out(str(self.dc_idx) + '_replay_buffer'))
