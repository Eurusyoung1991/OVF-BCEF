# Data format

Use a UTF-8 CSV with one header row and one row per case. A UTF-8 byte-order mark is accepted. Exactly nine probability inputs are required. Additional metadata columns are ignored. Inference also ignores any `label` column, including its contents.

| Order | Required probability column | Upstream MRI source / pooling |
| --- | --- | --- |
| 1 | `current_base` | Precomputed X-ray + primary MRI base probability |
| 2 | `mri_zd_nyul_siglip_mean_q90` | `mri_v2_nyul_fg_siglip2_mean` / mean_q90 |
| 3 | `mri_zd_nyul_siglip_q95` | `mri_v2_nyul_fg_siglip2_mean` / q95 |
| 4 | `mri_zd_nyul_radiodino_mean_q90` | `mri_v2_nyul_fg_radiodino_mean` / mean_q90 |
| 5 | `mri_bs_pctl_trainref_biomed_gated_mean_max` | `mri_v2_pctl_trainref_fg_biomedclip_gated_attention` / mean_max |
| 6 | `mri_bs_zscore_biomed_gated_mean` | `mri_v2_zscore_fg_biomedclip_gated_attention` / mean |
| 7 | `mri_bal_zscore_biomed_gated_mean_q90` | `mri_v2_zscore_fg_biomedclip_gated_attention` / mean_q90 |
| 8 | `mri_bal_histmatch_radiodino_att_mean` | `mri_v2_histmatch_fg_radiodino_attention` / mean |
| 9 | `mri_bal_pctl_siglip_gated_mean` | `mri_v2_pctl_fg_siglip2_gated_attention` / mean |

These stable input names retain their original mappings. Eight expert scores arise from six source configurations, with two pooling methods for some configurations. Source names containing `gated` or `attention` are identifiers; the indicated raw pooling method determines the score source. The primary MRI probability contributing to `current_base` uses the same Nyul/SigLIP2/mean_q90 source as the first MRI expert.

`case_id` must be a nonempty unique string with no surrounding whitespace. If different cohorts reuse local identifiers, create globally unique identifiers before generating the table. Training and validation case identifiers must be disjoint. Training and validation each require a numeric `label` column with only 0 and 1, containing both classes. Row order is preserved, and no implicit join or row dropping occurs.

Every probability must be numeric, finite, and within `[0,1]`. Boundary values of 0 and 1 are valid and are clipped before the logit transform. Empty tables, duplicate headers, missing columns, and rows with the wrong number of fields are errors. CSV column order may vary; API matrices must follow the order above.

The package does not require center labels or a fixed number of cohorts. Its public examples use only generated synthetic identifiers and scores. The example generator writes labeled `train.csv` and `validation.csv`, and label-free `inference.csv`.
