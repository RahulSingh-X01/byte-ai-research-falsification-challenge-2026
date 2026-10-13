"""
model.py
Small MLP for binary classification.
"""

import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-layer perceptron for binary classification.
    Input:  (batch, input_dim)
    Output: (batch,) raw logits
    """

    def __init__(self, input_dim, hidden_dim=64, n_hidden=2, activation="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)