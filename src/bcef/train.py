"""Train only the base_convex_bank probability fusion head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import TextIO

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import BCEFModel
from .scores import INPUT_NAMES, ScoreTable, probability_logits


class ConvexBlend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(len(INPUT_NAMES)))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.matmul(torch.softmax(self.weight_logits, dim=0)) + self.bias


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 3407
    epochs: int = 200
    patience: int = 25
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-3
    weight_l2: float = 0.01
    device: str = "cpu"

    def validate(self):
        for name in ("seed", "epochs", "patience", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid {name}")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must be at most 2**32 - 1")
        for name in ("lr", "weight_decay", "weight_l2"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                raise ValueError(f"Invalid {name}")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or a CUDA device")


def training_cutoff(labels, probabilities) -> float:
    """Maximize training Youden J among finite probability thresholds."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    usable = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    indexes = np.flatnonzero(usable)
    return float(thresholds[indexes[np.argmax((tpr - fpr)[usable])]])


def _validate_table(table: ScoreTable):
    logits = probability_logits(table.probabilities)
    if len(table.case_ids) != len(logits) or len(set(table.case_ids)) != len(logits):
        raise ValueError("case_ids must be unique and match the score rows")
    if any(not isinstance(case_id, str) or not case_id.strip() or case_id != case_id.strip()
           for case_id in table.case_ids):
        raise ValueError("case_ids must be nonempty strings without surrounding whitespace")
    labels = np.asarray(table.labels)
    if labels.shape != (len(logits),) or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Each training/validation table must have both binary label classes")
    return logits, labels.astype(np.float32)


def fit(train: ScoreTable, validation: ScoreTable, config: TrainConfig = TrainConfig(),
        *, log: TextIO | None = None) -> tuple[BCEFModel, dict]:
    config.validate()
    x_train, y_train = _validate_table(train)
    x_val, y_val = _validate_table(validation)
    if set(train.case_ids) & set(validation.case_ids):
        raise ValueError("Training and validation case_ids must be disjoint")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = ConvexBlend().to(device)
    pos = max(1.0, float(y_train.sum()))
    neg = max(1.0, float((y_train == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=config.batch_size, shuffle=True)
    val_tensor = torch.from_numpy(x_val).to(device)
    best, stale, history = None, 0, []
    for epoch in tqdm(range(1, config.epochs + 1), desc="BCEF training", file=log):
        model.train()
        for logits, labels in loader:
            logits, labels = logits.to(device), labels.to(device)
            w = torch.softmax(model.weight_logits, dim=0)
            loss = loss_fn(model(logits), labels) + config.weight_l2 * torch.sum((w - 1.0 / len(INPUT_NAMES)) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite; check inputs and optimizer settings")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_tensor)).cpu().numpy()
        auc = float(roc_auc_score(y_val, val_prob))
        history.append({"epoch": epoch, "validation_auc": auc})
        if best is None or auc > best["auc"] + 1e-5:
            best = {"auc": auc, "epoch": epoch,
                    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if log:
            log.write(f"epoch={epoch} validation_auc={auc:.8f} best_epoch={best['epoch']}\n")
            log.flush()
        if stale >= config.patience:
            break
    model.load_state_dict(best["state"])
    portable = BCEFModel(model.weight_logits.detach().cpu().tolist(), float(model.bias.detach().cpu()), 0.5)
    # Derive the threshold using the same portable predictor used at inference.
    portable.cutoff = training_cutoff(y_train, portable.predict_proba(train.probabilities))
    return portable, {"config": asdict(config), "best_epoch": best["epoch"],
                      "best_validation_auc": best["auc"], "train_n": len(y_train),
                      "validation_n": len(y_val), "cutoff": portable.cutoff,
                      "cutoff_source": "training_youden", "history": history}
