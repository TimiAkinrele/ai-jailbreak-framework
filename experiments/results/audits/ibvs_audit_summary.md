# IBVS Audit Summary

- Input: `/Users/timiakinrele/VSCode/dissertation/ai-jailbreak-classifier/experiments/results/audits/ibvs_component_audit_sample_splitA.csv`
- Alert column: `model_alert`
- Rows: 200

## Component Precision/Recall/F1

|   tp |   fp |   fn |   tn |   precision |   recall |       f1 |   support_positive |   pred_positive | component                          | manual_label_column                   |   n_evaluated |
|-----:|-----:|-----:|-----:|------------:|---------:|---------:|-------------------:|----------------:|:-----------------------------------|:--------------------------------------|--------------:|
|    7 |    2 |   13 |  178 |    0.777778 | 0.35     | 0.482759 |                 20 |               9 | hierarchy_override                 | gt_hierarchy_override                 |           200 |
|    0 |    0 |    1 |  199 |    0        | 0        | 0        |                  1 |               0 | system_developer_spoof             | gt_system_developer_spoof             |           200 |
|    2 |    1 |    5 |  192 |    0.666667 | 0.285714 | 0.4      |                  7 |               3 | evasion                            | gt_evasion                            |           200 |
|    0 |    0 |    0 |  200 |    0        | 0        | 0        |                  0 |               0 | tool_misuse_directive              | gt_tool_misuse_directive              |           200 |
|    0 |    1 |    1 |  198 |    0        | 0        | 0        |                  1 |               1 | benign_meta_educational_discussion | gt_benign_meta_educational_discussion |           200 |

## Triage Utility (Overall)

| ood_set   |   n_total |   n_alerts |   explanation_coverage |   suppression_only_rate |   actionable_explanation_precision |   avg_rules_per_alert |   avg_positive_components_per_alert |
|:----------|----------:|-----------:|-----------------------:|------------------------:|-----------------------------------:|----------------------:|------------------------------------:|
| all       |       200 |        103 |               0.427184 |                0.126214 |                           0.840909 |              0.757282 |                            0.281553 |

## Triage Utility (By OOD Set)

| ood_set            |   n_total |   n_alerts |   explanation_coverage |   suppression_only_rate |   actionable_explanation_precision |   avg_rules_per_alert |   avg_positive_components_per_alert |
|:-------------------|----------:|-----------:|-----------------------:|------------------------:|-----------------------------------:|----------------------:|------------------------------------:|
| injection          |        66 |         40 |               0.5      |               0.225     |                               0.9  |              0.85     |                            0.3      |
| injection_standard |        68 |         49 |               0.408163 |               0.0816327 |                               0.75 |              0.816327 |                            0.265306 |
| primary            |        66 |         14 |               0.285714 |               0         |                               1    |              0.285714 |                            0.285714 |

- Examples exported: `/Users/timiakinrele/VSCode/dissertation/ai-jailbreak-classifier/experiments/results/audits/ibvs_triage_examples.csv` (20 rows)
