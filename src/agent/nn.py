import torch
from torch import nn


def minimal_discrete_actor(input_size, output_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_size),
        torch.nn.Softmax(dim=1)
    )


def minimal_dynamics_model(input_size, output_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_size),
    )


def minimal_continuous_actor(input_size, output_size, hidden_layer_size=1024):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_layer_size),
        torch.nn.BatchNorm1d(hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_size, hidden_layer_size),
        torch.nn.BatchNorm1d(hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_size, output_size),
        torch.nn.Tanh(),
    )


class minimal_q_fn(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_size=1024):
        super(minimal_q_fn, self).__init__()

        self.linear = nn.Linear(state_size + action_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, 1)
        self.bn1 = nn.BatchNorm1d(hidden_layer_size)
        self.relu = nn.ReLU()

    def forward(self, st, at):
        combined = torch.cat((st, at), dim=1)
        combined = self.relu(self.bn1(self.linear(combined)))
        combined = self.relu(self.bn1(self.linear2(combined)))
        return self.linear3(combined)
