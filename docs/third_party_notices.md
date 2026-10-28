# Dependencies and authorship

The BCEF score-head implementation is provided by the project author. The public package contains the extracted fusion formula, a reorganized trainer, strict CSV validation, JSON serialization, command line interfaces, synthetic data generation, and tests.

External packages are installed as dependencies rather than copied into this repository:

| Dependency | Use |
| --- | --- |
| NumPy | Input validation and portable numerical inference |
| PyTorch (training extra) | Fusion parameters, automatic differentiation and optimization |
| scikit-learn (training extra) | Validation ROC AUC and training ROC cutoff |
| tqdm (training extra) | Persistent overall training progress |
| setuptools | Package build backend |

Each dependency retains its own license and notices; see its installed distribution metadata. This package does not redistribute image backbones, upstream pretrained weights, segmentation models or third-party research code snapshots.

The author's code is distributed under the [MIT License](../LICENSE). This license does not grant rights to upstream model weights or study data.
