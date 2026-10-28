"""Generate artificial example tables without reading any external inputs."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .scores import INPUT_NAMES


def generate(output_dir: Path, seed: int = 2026):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, size, labeled in (("train", 128, True), ("validation", 64, True), ("inference", 32, False)):
        labels = np.arange(size) % 2
        rng.shuffle(labels)
        latent = (2 * labels - 1)[:, None] * 0.65 + rng.normal(0, 1.0, (size, 1))
        logits = latent + rng.normal(0, 0.7, (size, len(INPUT_NAMES)))
        probabilities = 1 / (1 + np.exp(-logits))
        with (output_dir / f"{name}.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case_id", *INPUT_NAMES, *(["label"] if labeled else [])])
            for index, (row, label) in enumerate(zip(probabilities, labels)):
                writer.writerow([f"synthetic_{name}_{index:04d}", *row.tolist(), *([int(label)] if labeled else [])])


def main():
    parser = argparse.ArgumentParser(description="Generate purely synthetic BCEF example inputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        generate(args.output_dir, args.seed)
        print(f"Synthetic examples created: {args.output_dir}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
