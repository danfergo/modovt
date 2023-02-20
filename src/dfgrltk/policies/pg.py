import torch
from torch.distributions import Categorical

from src.dfgrltk.policies.policy import Policy


class PGPolicy(Policy):

    def __init__(self, actor, action_space=0):
        super().__init__(action_space)
        self.actor = actor

    def parameters(self):
        return self.actor.parameters()

    def log_prob(self, observation, action):
        if self.action_space == Policy.DISCRETE_AS:
            return Categorical(self.actor(torch.tensor(observation)))\
                .log_prob(torch.tensor(action, dtype=torch.int))
        # todo
        raise 'PG not implemented for handling continuous actions'

    def __call__(self, observation):
        if self.action_space == Policy.DISCRETE_AS:
            # gets the probabilities vector/distribution from the nn
            # and samples an action from this distribution.
            return Categorical(self.actor(torch.tensor(observation))).sample()
        # todo
        raise 'PG not implemented for handling continuous actions'
