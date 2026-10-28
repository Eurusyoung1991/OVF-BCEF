"""Strict wide-table input contract and clipped probability logits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPSILON = 1e-5
INPUT_NAMES = (
    "current_base",
    "mri_zd_nyul_siglip_mean_q90",
    "mri_zd_nyul_siglip_q95",
    "mri_zd_nyul_radiodino_mean_q90",
    "mri_bs_pctl_trainref_biomed_gated_mean_max",
    "mri_bs_zscore_biomed_gated_mean",
    "mri_bal_zscore_biomed_gated_mean_q90",
    "mri_bal_histmatch_radiodino_att_mean",
    "mri_bal_pctl_siglip_gated_mean",
)


def validate_probabilities(values) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be numeric") from exc
    if array.ndim != 2 or array.shape[1] != len(INPUT_NAMES) or array.shape[0] == 0:
        raise ValueError("Expected a nonempty [N, 9] probability matrix")
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return array


def probability_logits(values) -> np.ndarray:
    """Historical convention: float64 clipping/logit, then float32."""
    p = np.clip(validate_probabilities(values), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p)).astype(np.float32)


@dataclass(frozen=True)
class ScoreTable:
    case_ids: tuple[str, ...]
    probabilities: np.ndarray
    labels: np.ndarray | None = None


def load_scores(path: str | Path, *, require_labels: bool = False) -> ScoreTable:
    """Load by column name; labels and other metadata are ignored at inference."""
    ids, values, labels = [], [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) != len(set(header)):
            raise ValueError("CSV must have a nonempty header with unique column names")
        required = ["case_id", *INPUT_NAMES] + (["label"] if require_labels else [])
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        positions = [header.index(name) for name in INPUT_NAMES]
        id_position = header.index("case_id")
        label_position = header.index("label") if require_labels else None
        seen = set()
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {line} has the wrong number of fields")
            case_id = row[id_position]
            if not case_id.strip() or case_id != case_id.strip() or case_id in seen:
                raise ValueError(f"CSV row {line} has a blank, padded or duplicate case_id")
            seen.add(case_id)
            ids.append(case_id)
            try:
                values.append([float(row[index]) for index in positions])
                if require_labels:
                    labels.append(float(row[label_position]))
            except ValueError as exc:
                raise ValueError(f"CSV row {line} contains a nonnumeric score or label") from exc
    probabilities = validate_probabilities(values)
    y = None
    if require_labels:
        y = np.asarray(labels, dtype=np.float32)
        if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Training and validation labels must be binary 0/1")
        if len(np.unique(y)) != 2:
            raise ValueError("Training and validation tables must each contain both label classes")
    return ScoreTable(tuple(ids), probabilities, y)
