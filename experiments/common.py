"""Shared data and hyperparameters for the NumPy experiments."""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

ASSETS = os.path.join(ROOT, "docs", "assets")
SEEDS = range(5)
NUM_WORKERS = 4
BATCH_SIZE = 16
INNER_HPARAMS = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4)
OUTER_LR = 0.7
MOMENTUM_COEF = 0.9


def make_data(seed, num_samples=400, input_dim=8, num_classes=3, n_train=320):
    """Synthetic linearly-generated classification data, split into train and eval."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, input_dim))
    true_w = rng.standard_normal((input_dim, num_classes))
    y = np.argmax(x @ true_w + 0.1 * rng.standard_normal((num_samples, num_classes)), axis=1)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def save(fig, name):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    fig.savefig(path)
    print(f"saved {os.path.relpath(path, ROOT)}")
