import random

from src.dfgrltk.policies.policy import Policy
from src.dfgrltk.policies.random import RandomPolicy


class EGreedyPolicy(Policy):

    def __init__(self, greedy, e=0.1, rand=None):
        self.rand = RandomPolicy() if rand is None else rand
        self.greedy = greedy
        self.e = e
        super().__init__()

    def __call__(self, observation):
        t, o = observation
        # with a small probability
        if random.random() < self.e:
            return self.rand(o)
        else:
            return self.greedy(o)
