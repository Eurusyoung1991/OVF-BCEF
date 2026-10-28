"""Command line interfaces; inference never imports the training module."""

import argparse
import csv
import json
from pathlib import Path

from .model import BCEFModel
from .scores import load_scores


def train_main():
    parser = argparse.ArgumentParser(description="Train BCEF on nine probability columns")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for all training outputs")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--weight-l2", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        from .train import TrainConfig, fit
        config = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__})
        config.validate()
        train = load_scores(args.train, require_labels=True)
        validation = load_scores(args.validation, require_labels=True)
        if set(train.case_ids) & set(validation.case_ids):
            raise ValueError("Training and validation case_ids must be disjoint")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "progress.log").open("x", encoding="utf-8") as log:
            model, summary = fit(train, validation, config, log=log)
        model.save(args.output_dir / "model.json")
        with (args.output_dir / "training_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Training complete: {args.output_dir} (progress.log, model.json, training_summary.json)")
    except ImportError as exc:
        parser.error(f"Training dependencies are required: pip install '.[train]' ({exc.name})")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def predict_main():
    parser = argparse.ArgumentParser(description="Predict with a complete BCEF JSON model; labels are not required")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New CSV file")
    args = parser.parse_args()
    try:
        model = BCEFModel.load(args.model)
        table = load_scores(args.input)
        probabilities = model.predict_proba(table.probabilities)
        predictions = probabilities >= model.cutoff
        with args.output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case_id", "prob_pos", "pred"])
            for case_id, probability, prediction in zip(table.case_ids, probabilities, predictions):
                writer.writerow([case_id, float(probability), int(prediction)])
        print(f"Predicted {len(table.case_ids)} rows: {args.output}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
