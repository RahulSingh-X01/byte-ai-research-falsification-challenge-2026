"""
train.py
Generic training loop for the MLP.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import MLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    X_train, y_train, X_test, y_test,
    input_dim=None,
    hidden_dim=64, n_hidden=2, activation="relu",
    optimizer_name="adam", lr=1e-3,
    batch_size=128, max_epochs=200,
    seed=0, device="cpu",
    epoch_callback=None,  # new
    callback_every=20,   # new
):
    """
    Train an MLP. Returns (model, history).
    history keys: train_acc, test_acc (lists), epochs_to_95 (int or None)

    If epoch_callback is provided, it is called as epoch_callback(model, epoch)
    every `callback_every` epochs (after that epoch's update).
    """
    set_seed(seed)
    if input_dim is None:
        input_dim = X_train.shape[1]

    # Convert labels {-1, +1} -> {0, 1} for BCEWithLogitsLoss
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor((y_train + 1) / 2, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor((y_test + 1) / 2, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    model = MLP(input_dim, hidden_dim, n_hidden, activation).to(device)
    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history = {"train_acc": [], "test_acc": [], "epochs_to_95": None}

    X_tr_dev, y_tr_dev = X_tr.to(device), y_tr.to(device)
    X_te_dev, y_te_dev = X_te.to(device), y_te.to(device)

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc = ((model(X_tr_dev) > 0).float() == y_tr_dev).float().mean().item()
            te_acc = ((model(X_te_dev) > 0).float() == y_te_dev).float().mean().item()

        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if history["epochs_to_95"] is None and tr_acc >= 0.95:
            history["epochs_to_95"] = epoch + 1

        # NEW: callback
        if epoch_callback is not None and (epoch + 1) % callback_every == 0:
            epoch_callback(model, epoch + 1)

    return model, history