import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bcef import BCEFModel, EPSILON, INPUT_NAMES, load_scores
from bcef.scores import ScoreTable, probability_logits
from bcef.synthetic import generate


class FileCase(unittest.TestCase):
    def setUp(self):
        # Retained temporarily for inspection; no project files are deleted.
        self.root = Path(tempfile.mkdtemp(prefix="bcef-test-"))

    def write_csv(self, header=None, rows=None, name="input.csv"):
        path = self.root / name
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header or ["case_id", *INPUT_NAMES])
            writer.writerows(rows if rows is not None else [["synthetic_case", *([0.5] * 9)]])
        return path


class ModelTests(FileCase):
    def test_forward_matches_independent_float64_formula(self):
        rng = np.random.default_rng(9)
        p = rng.uniform(0, 1, (50, 9))
        p[0, :2] = [0, 1]
        logits = np.linspace(-2, 2, 9)
        model = BCEFModel(logits.tolist(), -0.13, 0.47)
        w = np.exp(logits - logits.max())
        w /= w.sum()
        clipped = np.clip(p, EPSILON, 1 - EPSILON)
        z = np.log(clipped / (1 - clipped)) @ w - 0.13
        np.testing.assert_allclose(model.predict_proba(p), 1 / (1 + np.exp(-z)), atol=2e-6, rtol=0)
        self.assertTrue((model.weights >= 0).all())
        self.assertAlmostEqual(float(model.weights.sum()), 1.0, places=6)

    def test_zero_initialization_means_uniform_logit_average(self):
        p = np.full((3, 9), 0.7)
        model = BCEFModel([0.0] * 9, 0.0, 0.7)
        np.testing.assert_allclose(model.weights, np.full(9, 1 / 9), atol=1e-7)
        np.testing.assert_allclose(model.predict_proba(p), 0.7, atol=1e-7)

    def test_save_load_exactly_preserves_outputs_and_labels(self):
        model = BCEFModel([float(x) for x in range(9)], 0.03, 0.6)
        p = np.random.default_rng(1).uniform(size=(40, 9))
        path = self.root / "model.json"
        model.save(path)
        loaded = BCEFModel.load(path)
        np.testing.assert_array_equal(model.predict_proba(p), loaded.predict_proba(p))
        np.testing.assert_array_equal(model.predict(p), loaded.predict(p))
        with self.assertRaises(FileExistsError):
            model.save(path)

    def test_parameter_artifact_validation(self):
        baseline = BCEFModel([0.0] * 9, 0.0, 0.5).to_dict()
        mutations = [dict(input_names=list(reversed(INPUT_NAMES))), dict(epsilon=0.01),
                     dict(weight_logits=[0.0] * 8), dict(bias=float("nan")),
                     dict(weight_logits=[1e100] * 9), dict(cutoff=2), dict(cutoff=True),
                     dict(method="gate"), dict(compute_dtype="float64"), dict(extra=1)]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                path = self.root / f"invalid_{index}.json"
                path.write_text(json.dumps({**baseline, **mutation}))
                with self.assertRaises(ValueError):
                    BCEFModel.load(path)
        path = self.root / "duplicate.json"
        path.write_text('{"bias": 0, "bias": 1}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            BCEFModel.load(path)
        path = self.root / "incomplete.json"
        path.write_text('{}')
        with self.assertRaises(ValueError):
            BCEFModel.load(path)

    def test_invalid_probability_matrices(self):
        model = BCEFModel([0.0] * 9, 0.0, 0.5)
        invalid = [[], np.ones((2, 8)), np.ones(9), np.full((1, 9), np.nan),
                   np.full((1, 9), np.inf), np.full((1, 9), -0.01), np.full((1, 9), 1.01)]
        for p in invalid:
            with self.subTest(shape=np.shape(p)), self.assertRaises(ValueError):
                model.predict_proba(p)

    def test_extreme_parameters_produce_finite_probabilities(self):
        model = BCEFModel([3e38, -3e38, *([0.0] * 7)], -3e38, 0.5)
        self.assertTrue(np.isfinite(model.predict_proba(np.ones((2, 9)))).all())


class SchemaTests(FileCase):
    def test_column_reordering_and_row_order(self):
        header = [*reversed(INPUT_NAMES), "case_id", "extra"]
        scores = np.linspace(0, 1, 9).tolist()
        table = load_scores(self.write_csv(header, [[*reversed(scores), "synthetic_b", "metadata"],
                                                   [*reversed(scores), "synthetic_a", "metadata"]]))
        np.testing.assert_array_equal(table.probabilities[0], scores)
        self.assertEqual(table.case_ids, ("synthetic_b", "synthetic_a"))

    def test_inference_does_not_parse_labels(self):
        path = self.write_csv(["case_id", *INPUT_NAMES, "label"], [["synthetic", *([0.5] * 9), "ignored"]])
        self.assertIsNone(load_scores(path).labels)
        with self.assertRaises(ValueError):
            load_scores(path, require_labels=True)

    def test_malformed_schema_rejected(self):
        cases = [(["case_id", *INPUT_NAMES[:-1]], [["a", *([0.5] * 8)]]),
                 (["case_id", *INPUT_NAMES, INPUT_NAMES[0]], [["a", *([0.5] * 10)]]),
                 (["case_id", *INPUT_NAMES], []),
                 (["case_id", *INPUT_NAMES], [["a", *([0.5] * 8)]]),
                 (["case_id", *INPUT_NAMES], [["a", *([0.5] * 10)]])]
        for index, (header, rows) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_scores(self.write_csv(header, rows, f"invalid_{index}.csv"))

    def test_invalid_case_ids(self):
        for index, ids in enumerate(([""], [" a"], ["a "], ["a", "a"])):
            path = self.write_csv(rows=[[case_id, *([0.5] * 9)] for case_id in ids], name=f"ids_{index}.csv")
            with self.assertRaises(ValueError):
                load_scores(path)

    def test_invalid_labels(self):
        for index, labels in enumerate(([0, 0], [0, 2], [0, float("nan")], [0, 0.5])):
            path = self.write_csv(["case_id", *INPUT_NAMES, "label"],
                                  [[f"synthetic_{i}", *([0.5] * 9), y] for i, y in enumerate(labels)],
                                  f"labels_{index}.csv")
            with self.assertRaises(ValueError):
                load_scores(path, require_labels=True)

    def test_invalid_csv_probability_values(self):
        for index, value in enumerate(("NaN", "inf", "-0.1", "1.1", "text", "")):
            path = self.write_csv(rows=[["synthetic", value, *([0.5] * 8)]], name=f"score_{index}.csv")
            with self.assertRaises(ValueError):
                load_scores(path)


class WorkflowTests(FileCase):
    def test_training_checkpoint_cutoff_and_roundtrip(self):
        from bcef.train import TrainConfig, fit, training_cutoff
        generate(self.root / "synthetic")
        train = load_scores(self.root / "synthetic/train.csv", require_labels=True)
        val = load_scores(self.root / "synthetic/validation.csv", require_labels=True)
        log = io.StringIO()
        config = TrainConfig(epochs=5, patience=2, batch_size=64)
        model, summary = fit(train, val, config, log=log)
        selected = summary["history"][summary["best_epoch"] - 1]["validation_auc"]
        self.assertEqual(selected, summary["best_validation_auc"])
        self.assertEqual(model.cutoff, training_cutoff(train.labels, model.predict_proba(train.probabilities)))
        self.assertIn("epoch=1", log.getvalue())
        self.assertIn("BCEF training", log.getvalue())
        self.assertGreaterEqual(len(summary["history"]), 1)
        self.assertLessEqual(len(summary["history"]), 5)
        path = self.root / "trained.json"
        model.save(path)
        np.testing.assert_array_equal(model.predict(train.probabilities), BCEFModel.load(path).predict(train.probabilities))
        second, second_summary = fit(train, val, config, log=io.StringIO())
        self.assertEqual(model.to_dict(), second.to_dict())
        self.assertEqual(summary, second_summary)

    def test_training_rejects_overlap_and_invalid_config(self):
        from bcef.train import TrainConfig, fit
        table = ScoreTable(("synthetic_a", "synthetic_b"), np.full((2, 9), 0.5), np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit(table, table)
        for config in (TrainConfig(epochs=0), TrainConfig(lr=float("nan")), TrainConfig(weight_l2=-1),
                       TrainConfig(seed=-1), TrainConfig(seed=2**32)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_constant_predictions_have_finite_training_cutoff(self):
        from bcef.train import training_cutoff
        self.assertEqual(training_cutoff([0, 1, 0, 1], [0.5] * 4), 0.5)

    def test_torch_forward_and_zero_parameters(self):
        import torch
        from bcef.train import ConvexBlend
        blend = ConvexBlend()
        np.testing.assert_array_equal(blend.weight_logits.detach().numpy(), np.zeros(9))
        self.assertEqual(blend.bias.item(), 0)
        rng = np.random.default_rng(3)
        values = rng.uniform(size=(200, 9))
        with torch.no_grad():
            blend.weight_logits.copy_(torch.linspace(-1, 1, 9))
            blend.bias.fill_(0.12)
            expected = torch.sigmoid(blend(torch.from_numpy(probability_logits(values)))).numpy()
        portable = BCEFModel(blend.weight_logits.detach().tolist(), blend.bias.item(), 0.5)
        np.testing.assert_allclose(portable.predict_proba(values), expected, atol=2e-6, rtol=0)

    def test_inference_cli_without_torch_sklearn_or_labels(self):
        generate(self.root / "synthetic")
        BCEFModel([0.0] * 9, 0.0, 0.5).save(self.root / "model.json")
        script = '''
import importlib.abc
import sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn', 'pandas', 'tqdm'}:
            raise AssertionError('Inference imported a training dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
from bcef.cli import predict_main
predict_main()
'''
        output = self.root / "predictions.csv"
        command = [sys.executable, "-c", script, "--model", str(self.root / "model.json"),
                   "--input", str(self.root / "synthetic/inference.csv"), "--output", str(output)]
        result = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)
        with output.open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["case_id", "prob_pos", "pred"])
            self.assertEqual(len(list(reader)), 32)
        before = output.read_bytes()
        repeated = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
