from src.dfgrltk.policies.policy import Policy

import random


class RandomPolicy(Policy):

    def __init__(self):
        pass

    def __call__(self, observation):
        return {
            k: random.random() * 2 - 1
            for k in observation
        }
