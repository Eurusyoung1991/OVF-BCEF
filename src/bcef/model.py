"""Portable float32 fusion inference with a complete JSON parameter format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scores import EPSILON, INPUT_NAMES, probability_logits

FORMAT_VERSION = "bcef-score-v1"


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class BCEFModel:
    """Global softmax weights and a scalar bias over nine clipped logits."""

    def __init__(self, weight_logits, bias: float, cutoff: float):
        raw = list(weight_logits)
        if len(raw) != len(INPUT_NAMES):
            raise ValueError("Exactly nine weight logits are required")
        values = [_number(value, "weight logit") for value in raw]
        with np.errstate(over="ignore"):
            self.weight_logits = np.asarray(values, dtype=np.float32)
            self.bias = np.float32(_number(bias, "bias"))
        if not np.isfinite(self.weight_logits).all() or not np.isfinite(self.bias):
            raise ValueError("Parameters must be representable as finite float32")
        self.cutoff = _number(cutoff, "cutoff")
        if not 0 <= self.cutoff <= 1:
            raise ValueError("cutoff must be in [0, 1]")

    @property
    def weights(self) -> np.ndarray:
        with np.errstate(over="ignore", under="ignore"):
            exponents = np.exp(self.weight_logits - np.max(self.weight_logits))
        return exponents / np.sum(exponents, dtype=np.float32)

    def predict_proba(self, probabilities) -> np.ndarray:
        logits = probability_logits(probabilities)
        z = logits @ self.weights + self.bias
        p = np.empty_like(z)
        positive = z >= 0
        p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        p[~positive] = exp_z / (1.0 + exp_z)
        return p

    def predict(self, probabilities) -> np.ndarray:
        return (self.predict_proba(probabilities) >= self.cutoff).astype(np.int64)

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "method": "base_convex_bank",
            "input_names": list(INPUT_NAMES),
            "epsilon": EPSILON,
            "compute_dtype": "float32",
            "weight_logits": [float(value) for value in self.weight_logits],
            "bias": float(self.bias),
            "cutoff": self.cutoff,
            "cutoff_source": "training_youden",
        }

    def save(self, path: str | Path) -> None:
        """Create a new artifact; never overwrite an existing parameter file."""
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "BCEFModel":
        with Path(path).open(encoding="utf-8") as handle:
            obj = json.load(handle, object_pairs_hook=_unique_object)
        required = {"format_version", "method", "input_names", "epsilon", "compute_dtype",
                    "weight_logits", "bias", "cutoff", "cutoff_source"}
        if not isinstance(obj, dict) or set(obj) != required:
            raise ValueError("Incomplete or unknown model artifact fields")
        if (obj["format_version"] != FORMAT_VERSION or obj["method"] != "base_convex_bank"
                or obj["input_names"] != list(INPUT_NAMES)
                or _number(obj["epsilon"], "epsilon") != EPSILON
                or obj["compute_dtype"] != "float32"
                or obj["cutoff_source"] != "training_youden"):
            raise ValueError("Unsupported model format, input order or numerical convention")
        if not isinstance(obj["weight_logits"], list):
            raise ValueError("weight_logits must be a JSON array")
        return cls(obj["weight_logits"], obj["bias"], obj["cutoff"])
