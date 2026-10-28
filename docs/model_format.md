# Portable model format

Training produces a complete UTF-8 JSON artifact with these required fields:

| Field | Meaning |
| --- | --- |
| `format_version` | `bcef-score-v1` |
| `method` | `base_convex_bank` |
| `input_names` | The nine ordered input names |
| `epsilon` | `0.00001` |
| `compute_dtype` | `float32` |
| `weight_logits` | Nine finite pre-softmax weights |
| `bias` | Finite scalar bias |
| `cutoff` | Finite training-derived probability cutoff in `[0,1]` |
| `cutoff_source` | `training_youden` |

Weight logits and bias must be representable as float32. Loading rejects duplicate JSON keys, incomplete or unknown fields, a different input order, and unsupported numerical conventions. Saving refuses to overwrite an existing file. No pickle, executable payloads, patient tables, or runtime parameter recovery are involved.

`training_summary.json` is a separate training record containing settings, sample counts, checkpoint selection, cutoff and per-epoch validation AUC. It contains no case identifiers, score rows, input paths, or labels. It is not required for inference. `progress.log` retains the training progress bar and epoch summaries.

The repository distributes code for creating and consuming this format. Users generate their own parameter files by training; no research-trained parameter artifact is bundled.
