import torch.nn as nn


def build_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int):
    layers = []
    layers.append(nn.Flatten())

    if n_layers == 0:
        layers.append(nn.Linear(in_dim, out_dim))

    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)
