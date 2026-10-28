# OVF-BCEF

**Base-guided Constrained Expert-bank Fusion (BCEF)** combines a base-model probability and eight MRI expert probabilities using learned global convex weights in logit space.

This repository provides the **score-level `base_convex_bank` head**: training, portable JSON model export, inference, and a synthetic example generator. Bring your own nine input scores and train a fusion head. Research-trained parameters and patient data are not distributed. Image preprocessing, feature extraction, single-modality models, construction of `current_base`, and clinical fusion are outside this package.

For probabilities `p_i`, BCEF computes:

```text
z_i = log(clip(p_i, 1e-5, 1 - 1e-5) / (1 - clip(p_i, 1e-5, 1 - 1e-5)))
w_i = softmax(a)_i
p_BCEF = sigmoid(sum_i(w_i * z_i) + bias)
```

The nine weights are shared across all cases, nonnegative, and sum to one. Weight logits and bias start at zero, giving uniform initial weights. The base weight is learned along with the expert weights.

## Install

Python 3.10 or later is required. From the repository directory:

```bash
python -m pip install '.[train]'
```

For inference with an existing compatible JSON model, use `python -m pip install .`; only NumPy is required. Installing or importing the inference package does not start training or download models.

## Run a synthetic example

All example inputs are generated mathematically from a random seed. Commands create new output locations and refuse to overwrite existing files or directories.

```bash
bcef-synthetic --output-dir runs/synthetic
bcef-train --train runs/synthetic/train.csv --validation runs/synthetic/validation.csv --output-dir runs/demo --epochs 10 --device cpu
bcef-predict --model runs/demo/model.json --input runs/synthetic/inference.csv --output runs/demo/predictions.csv
```

Training writes `model.json`, `training_summary.json`, and a persistent `progress.log` containing the overall epoch progress bar and validation history. Inference produces `case_id,prob_pos,pred`, preserving input row order. Labels are not required or used at inference. The saved training-derived cutoff is applied with `prob_pos >= cutoff`.

## Input scores

CSV input has one row per case, a unique `case_id`, and all nine columns listed in [the data format](docs/data_format.md). Training and validation additionally require a binary `label` and both classes in each table. Identifiers must be disjoint across training and validation. Headers may be reordered: scores are read by name. Missing scores, duplicate columns or identifiers, nonnumeric/nonfinite values, and probabilities outside `[0,1]` are rejected.

Use base and expert scores from your own upstream models. In particular, `current_base` is a precomputed probability, rather than an average calculated by this package. Training cases used to develop upstream models and held-out evaluation data must follow your own study design.

## Python inference

```python
from bcef import BCEFModel, load_scores

model = BCEFModel.load("runs/demo/model.json")
table = load_scores("runs/synthetic/inference.csv")
probabilities = model.predict_proba(table.probabilities)
predictions = model.predict(table.probabilities)
```

See [method and training settings](docs/method.md), [parameter format](docs/model_format.md), and [dependencies and authorship](docs/third_party_notices.md).

## Tests

After installing the training extra:

```bash
python -m unittest discover -s tests -v
```

The tests cover the formula, input validation, save/load consistency, training/validation separation, and label-free inference without training dependencies. Synthetic validation demonstrates software behavior; it does not reproduce study performance.

The CPU verification environment uses Python 3.10.13, NumPy 1.26.0, PyTorch 2.5.1, scikit-learn 1.3.1, and tqdm 4.66.1. Dependency ranges in the package metadata allow installation on other compatible environments; those combinations have not all been tested.

## License

MIT. See [LICENSE](LICENSE). External dependencies retain their own licenses.
