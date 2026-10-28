# Method and training

BCEF takes the clipped logits of nine probabilities in the fixed order documented in [data_format.md](data_format.md). Trainable weight logits `a` are transformed by softmax, and a scalar bias is added to the weighted sum. Sigmoid produces the positive-class probability. Weights are global across patients. The model has nine weight logits and one scalar bias; both are initialized at zero.

For a training mini-batch, the objective is:

```text
BCEWithLogitsLoss(pos_weight = training_negatives / training_positives)
  + weight_l2 * sum_i((softmax(a)_i - 1/9)^2)
```

The class weight is calculated from the training set. The penalty encourages weights toward the uniform distribution. AdamW weight decay is a separate optimizer setting.

| Setting | Default |
| --- | --- |
| Random seed | 3407 |
| Maximum epochs | 200 |
| Early-stopping patience | 25 |
| Batch size | 128 |
| Learning rate | 0.003 |
| AdamW weight decay | 0.001 |
| Uniform-weight penalty coefficient | 0.01 |
| Gradient norm clipping | 5.0 |
| Device | CPU |

Mini-batches are shuffled. At the end of each epoch, validation ROC AUC is evaluated. A checkpoint replaces the best checkpoint only if its AUC improves by more than `1e-5`; the first epoch always establishes a checkpoint. Early stopping counts epochs without such an improvement. The selected checkpoint is restored before parameter export and cutoff calculation.

The classification cutoff maximizes Youden's `TPR - FPR` on the training predictions from the exported predictor. Ties use the first finite threshold in descending ROC threshold order. Thresholds outside `[0,1]`, including the ROC infinity sentinel, are excluded. This gives a finite cutoff for constant or poorly ranked scores as well. Inference always uses this saved cutoff, without refitting or selecting a new threshold.

Probabilities are clipped and converted to logits in float64, then cast to float32. Training uses float32. Portable inference computes float32 softmax weights, the weighted logits, and a stable sigmoid. CPU library implementations can introduce small rounding differences; synthetic numerical comparisons use an absolute tolerance of `2e-6`. Exact equality across PyTorch/NumPy releases or hardware is not guaranteed.

This package extracts the score-head training mechanism. Removing unused reliability features and unrelated X-ray branches makes the nine-score head independent. Strict table checks, explicit JSON serialization, finite cutoff selection, and NumPy inference make its inputs and outputs portable. The package does not include historical model parameters or real-data reference predictions. Retraining on new scores yields a new model; the synthetic example is an execution example rather than a study-performance reproduction.
