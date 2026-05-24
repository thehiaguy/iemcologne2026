"""
[4/6] neural_net.py
-------------------
PyTorch residual MLP for binary match-outcome prediction.

Architecture
  BatchNorm → Linear(hidden) → SiLU → Dropout
  → N × ResBlock(LayerNorm + SiLU + Dropout)
  → Linear(hidden//4) → SiLU → Dropout → Linear(1)

Trained with:
  Mixed precision (torch.autocast + GradScaler)
  AdamW + CosineAnnealingWarmRestarts
  BCEWithLogitsLoss
  Early stopping on validation AUC

Exports
-------
  ResBlock
  MatchPredictor
  train_pytorch()   → (model, best_val_auc)
  pytorch_predict() → probability array
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from o1_config import (
    PT_HIDDEN, PT_N_BLOCKS, PT_DROPOUT,
    PT_LR, PT_WD, PT_BATCH, PT_EPOCHS, PT_PATIENCE,
    PT_T0, PT_T_MULT,
)

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MatchPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden:   int   = PT_HIDDEN,
        n_blocks: int   = PT_N_BLOCKS,
        dropout:  float = PT_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(hidden, dropout * 0.5) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input_proj(x))).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pytorch(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    input_dim: int,
    history_path=None,
) -> Tuple[MatchPredictor, float]:
    """Train MatchPredictor and return (cpu model, best_val_auc)."""
    log.info("Training PyTorch ResNet MLP on %s  (input_dim=%d)", DEVICE, input_dim)
    if DEVICE.type == "cuda":
        log.info(
            "  GPU: %s  VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    model     = MatchPredictor(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PT_LR, weight_decay=PT_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=PT_T0, T_mult=PT_T_MULT
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_on    = DEVICE.type == "cuda"
    scaler_g  = torch.GradScaler(enabled=amp_on)

    train_ds = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(y_tr),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=PT_BATCH,
        shuffle=True,
        pin_memory=amp_on,
        num_workers=0,
    )

    best_auc, best_state, patience = 0.0, None, 0
    _history = []

    for epoch in range(PT_EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=amp_on):
                loss = criterion(model(xb), yb)
            epoch_loss += loss.item()
            n_batches  += 1
            scaler_g.scale(loss).backward()
            scaler_g.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_g.step(optimizer)
            scaler_g.update()
        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            X_va_t     = torch.FloatTensor(X_va).to(DEVICE)
            y_va_t     = torch.FloatTensor(y_va).to(DEVICE)
            val_logits = model(X_va_t)
            val_loss   = criterion(val_logits, y_va_t).item()
            val_probs  = torch.sigmoid(val_logits).cpu().numpy()

        val_auc = roc_auc_score(y_va, val_probs) if len(set(y_va)) > 1 else 0.0
        _history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_auc": val_auc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if epoch % 25 == 0:
            log.info("  epoch %3d  val_AUC=%.4f  best=%.4f", epoch, val_auc, best_auc)

        if patience >= PT_PATIENCE:
            log.info("  Early stop at epoch %d", epoch)
            break

    if best_state:
        model.load_state_dict(best_state)
    log.info("PyTorch best val AUC: %.4f", best_auc)

    if history_path is not None:
        import pandas as pd
        pd.DataFrame(_history).to_csv(history_path, index=False)
        log.info("Training history saved to %s", history_path)

    return model.cpu(), best_auc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def pytorch_predict(model: MatchPredictor, X: np.ndarray) -> np.ndarray:
    """Return win probabilities as a numpy array."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        return torch.sigmoid(logits).numpy()
