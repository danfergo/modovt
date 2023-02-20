import torch.nn


def mlp(nn_shape, softmax=True):
    layers = []
    n_layers = len(nn_shape)
    for i in range(n_layers - 1):
        layers.append(torch.nn.Linear(nn_shape[i], nn_shape[i + 1]))
        if i < n_layers - 2:
            layers.append(torch.nn.ReLU())

    if softmax:
        layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*layers)
