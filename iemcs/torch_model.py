"""PyTorch MLP match-outcome model, wrapped as a scikit-learn classifier.

Subclasses ``BaseEstimator``/``ClassifierMixin`` so it satisfies the same
``fit`` / ``predict_proba`` contract as the other base learners and survives the
``sklearn.base.clone`` calls used by the stacking loop in :mod:`iemcs.model`.

Deliberately CPU-only: the model is tiny and the dataset small (~27k x 11), so GPU
transfer would only add latency and non-determinism. ``_HAS_TORCH`` lets callers
auto-detect availability and skip it gracefully when torch is not installed.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # torch not installed -> caller skips this learner
    _HAS_TORCH = False


if _HAS_TORCH:
    class _MLP(nn.Module):
        def __init__(self, n_in: int, hidden=(48, 24), p_drop: float = 0.25):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
                d = h
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Small regularised MLP predicting P(team_a wins)."""

    def __init__(self, hidden=(48, 24), p_drop=0.25, lr=1e-3, weight_decay=1e-4,
                 epochs=50, batch_size=512, seed=0):
        self.hidden = hidden
        self.p_drop = p_drop
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is not installed")
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.scaler_ = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler_.transform(X), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)

        self.net_ = _MLP(X.shape[1], tuple(self.hidden), self.p_drop)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True,
        )
        self.net_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.net_(xb), yb).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(self.scaler_.transform(X),
                                            dtype=torch.float32))
            p = torch.sigmoid(logits).numpy().astype(np.float64)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
