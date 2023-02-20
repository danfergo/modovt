import torch

from src.dfgrltk.policies.policy import Policy


class AVPolicy(Policy):

    def __init__(self, value_fn, action_space=0):
        super().__init__(action_space)
        self.value_fn = value_fn

    def parameters(self):
        return self.value_fn.parameters()

    def value(self, observation, action):
        actions = self.value_fn(observation)
        print('---------------------')
        print(actions)
        print('-----')
        print(action)
        print('-----')
        print(actions.select(1, 1))
        print(actions.size(), action.size())

        return actions[action.item()]

    def max_value(self, observation):
        return torch.max(self.value_fn(observation))

    def __call__(self, observation):
        observation = torch.tensor(observation)

        if self.action_space == Policy.DISCRETE_AS:
            return torch.argmax(self.value_fn(observation))
        raise 'AV not implemented for handling continuous actions'
