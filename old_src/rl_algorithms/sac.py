import copy
import os
import random
import sys

import numpy as np
import torch
from torch.autograd import Variable

from experimenter import experiment, e


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

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

    def load(self, p):
        if os.path.exists(p):
            buff = np.load(p, allow_pickle=True)
            self.buffer = [(torch.tensor(o, dtype=torch.float),
                            torch.tensor(a, dtype=torch.float), r,
                            torch.tensor(o1, dtype=torch.float), d) for (o, a, r, o1, d) in buff]


@experiment("", {
    'n_exploration_steps': 256,
    'n_start_episodes': 256,
    'n_update_batches': 64,
    'update_batch_size': 8,
    'opt_lr': 0.1,
    'q_fn_lr': 0.1,
    'v_fn_lr': 0.1,
    'gamma': 0.99,
    'polyak_factor': 0.995,
    'dc_idx': sys.argv[1]
})
class DDPG:

    def __init__(self, value_function, q_function, policy):
        self.n_exploration_steps = e.n_exploration_steps
        self.n_start_episodes = e.n_start_episodes
        self.n_update_batches = e.n_update_batches
        self.update_batch_size = e.update_batch_size
        self.gamma = e.gamma
        self.opt_lr = e.opt_lr
        self.q_fn_lr = e.q_fn_lr
        self.v_fn_lr = e.v_fn_lr
        self.polyak_factor = e.polyak_factor

        self.q_fn = q_function
        self.policy = policy

        self.v_fn_opt = torch.optim.Adam(self.v_fn.parameters(), lr=self.v_fn_lr)
        self.q_fn_opt = torch.optim.Adam(self.q_fn.parameters(), lr=self.q_fn_lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.opt_lr)
        self.dc_idx = e.dc_idx

    # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/15
    def polyak_update(self, polyak_factor, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(polyak_factor * target_param.data + (1.0 - polyak_factor) * param)

    def update(self, q_fns, replay_buffer):
        self.policy.train()

        for i in range(self.n_update_batches):
            batch = replay_buffer.sample_batch(self.update_batch_size)

            # compute targets
            rt = torch.tensor(np.array([r for o, a, r, o1, d in batch]))
            at = torch.cat([a for o, a, r, o1, d in batch], dim=0)
            ot = torch.cat([o1 for o, a, r, o1, d in batch], dim=0)
            ot_1 = torch.cat([o1 for o, a, r, o1, d in batch], dim=0)
            q_sa_t = torch.minimum(q_fns[0](ot, at), q_fns[1](ot, at))

            # pi_st = torch.distributions.Categorical(probs=self.nn(st))

            v_target = self.v_fn(ot) * (self.v_fn(ot) - q_sa_t + self.policy.log_prob(ot, at))
            q_targets = [self.q_fn.prob(at, st)]

            # update q function
            # loss = torch.tensor(
            #     [torch.square(self.q_fn(o, a) - targets[i]) for i, (o, a, r, o1, d) in enumerate(batch)])
            loss = torch.mean(self.q_fn(ot, at) - targets)
            loss = Variable(loss, requires_grad=True)
            self.q_fn_opt.zero_grad()
            loss.backward()
            self.q_fn_opt.step()

            # update policy, minus signal because we want to maximize
            # loss = torch.tensor([self.q_fn(o, self.policy(o)) for i, (o, a, r, o1, d) in enumerate(batch)])
            loss = - torch.mean(self.q_fn(ot, self.policy(ot)))
            loss = Variable(loss, requires_grad=True)
            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()

            # update target networks with polyak update
            self.polyak_update(self.polyak_factor, q_fn_targ, self.q_fn)
            self.polyak_update(self.polyak_factor, policy_targ, self.policy)

    def gather_data(self, policy, replay_buffer, get_observation, take_action, get_reward):
        self.policy.eval()
        with torch.no_grad():
            rr = 0
            for i in range(self.n_exploration_steps):
                # get observation from the environment
                o = get_observation()

                # select action, following the main policy
                o = torch.tensor(o, dtype=torch.float)
                o = torch.unsqueeze(o, 0)  # make a batch of one.

                # if epoch >= self.n_start_episodes:
                #     epsilon = np.random.normal(0, 0.01, 6)
                #     epsilon = torch.Tensor(np.array([epsilon]))
                #     a = self.policy(o) + epsilon  # todo what to clip
                # else:
                a = self.policy(o) * 0 + torch.Tensor(np.array([np.random.normal(0, 1, 6)]))

                # a = clip(a.detach().numpy()[0])
                take_action(a.detach().numpy()[0])

                # get reward
                r = get_reward()

                # get next observation from the environment
                o1 = get_observation()
                o1 = torch.tensor(o1, dtype=torch.float)
                o1 = torch.unsqueeze(o1, 0)  # make a batch of one.

                d = i == self.n_exploration_steps - 1
                replay_buffer.store((o, a, r, o1, d))

                rr = rr + r
            rr /= self.n_exploration_steps
        return rr

        def run(self, get_observation, get_reward, take_action, reset):

            # input: value and target value functions (psi), 2 twin q-functions (fi) and policy (theta)
            # and empty Replay Buffer
            q = [copy.deepcopy(self.q_fn), copy.deepcopy(self.q_fn)]
            v = self.v_fn
            v_targ = copy.deepcopy(v)
            pi = self.policy

            replay_buffer = ReplayBuffer()
            # replay_buffer.load(e.out(str(self.dc_idx) + '_replay_buffer.npy'))

            epoch = 0
            while True:
                rr = self.gather_data(pi, replay_buffer, get_observation, take_action, get_reward)

                e.emit('plot', {'name': 'reward', 'value': rr})
                e.emit('il_patch', {'name': 'meta', 'data': {
                    'epoch': epoch,
                    'replay_buffer_size': len(replay_buffer)
                }})
                epoch += 1
                replay_buffer.save(e.out(str(self.dc_idx) + '_replay_buffer'))

                self.update(q, v, v_targ, pi, replay_buffer)

                reset()
