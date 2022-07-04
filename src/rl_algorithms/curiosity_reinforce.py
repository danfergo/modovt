import numpy as np
import torch

from experimenter import EConfigBlock, e
from lib.train import SupervisedTrainer
from base64 import decode

n_actions = 6
lr = 0.01
discount_factor = 0.95
T = 10


class CuriosityReinforce:

    def __init__(self, dynamics_model, policy):

        # dynamics model
        self.dynamics_model = dynamics_model
        self.dynamics_model_opt = torch.optim.Adadelta(self.dynamics_model.parameters())
        self.dynamics_model_loss = torch.nn.L1Loss()

        self.s_trainer = SupervisedTrainer(
            self.dynamics_model,
            self.dynamics_model_opt,
            self.dynamics_model_loss
        )

        # policy model
        self.policy = policy
        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def eval_return(self, t, rewards):
        return sum([(discount_factor ** k) * r for k, r in enumerate(rewards[t:])])

    def run(self, get_observation, get_reward, take_action):

        epoch = 1

        while True:
            # episode buffers
            episode_actions = []
            episode_observations = []
            episode_rewards = []

            # generate an episode, explore.
            for t in range(T):
                # get observation from the environment
                o = get_observation()
                o = torch.tensor(o, dtype=torch.float)
                o = torch.unsqueeze(o, 0)  # make a batch of one.

                # calculate the reward
                r = get_reward()

                # sample action, given the observation, following the policy
                pi_st = torch.distributions.Categorical(probs=self.policy(o))
                a = pi_st.sample().item()

                # st, at, rt, into the buffers
                episode_actions.append(a)
                episode_observations.append(o)
                episode_rewards.append(r)

                # take action and advance the environment
                take_action(a)

            # print('learning')
            returns = [self.eval_return(t, episode_rewards) for t in range(len(episode_rewards))]

            # learn dynamics
            x = [torch.cat((episode_observations[t - 1], torch.tensor([[at]], dtype=torch.float32)), dim=1) for
                 t, (st, at) in enumerate(zip(episode_observations, episode_actions)) if t > 0]
            y = [st for t, st in enumerate(episode_observations) if t > 0]

            loss, y_pred = self.s_trainer.train_on_batch((
                torch.stack(x),
                torch.stack(y))
            )
            e.emit('plot', {'name': 'dynamics_loss', 'value': loss.item()})
            # print('dynamics loss: ', loss)

            # learn policy

            for st, at, gt in zip(episode_observations, episode_actions, returns):
                at = torch.tensor(at, dtype=torch.int)
                pi_st = torch.distributions.Categorical(probs=self.policy(st))

                log_prob = pi_st.log_prob(at)  # the log probability of this action being taken, i.e. log_πθ(at|st)

                loss = - log_prob * gt  # log_πθ(at|st) * g
                # the negative is used because we want to maximize the expected returns,
                # while the optimizer minimizes

                self.policy_opt.zero_grad()  # zero the radient graph for every parameter x
                loss.backward()  # compute the gradients dloss/dx
                self.policy_opt.step()  # performs the gradient update for every x i.e., x += -lr * x.grad

            e.emit('plot', {'name': 'reward', 'value': np.mean(np.array(returns))})
            e.emit('il_patch', {'name': 'meta', 'data': {'epoch': epoch}})
            e.emit('epoch_end', {'epoch': epoch})
            epoch += 1
