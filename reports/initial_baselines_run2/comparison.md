# Baseline Comparison

| Experiment | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-baseline | 6 | 0.9805 | 0.9844 | 0.9844 | 0 | - |
| minimal-experiment | 6 | 0.5098 | 0.4766 | 0.9531 | 3 | 20 |

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6]
    y-axis "Accuracy" 0 --> 1.03
    line "fixed-mlp-baseline" [0.9062, 0.9141, 0.9609, 0.9688, 0.9688, 0.9844]
    line "minimal-experiment" [0.9062, 0.9141, 0.8828, 0.9531, 0.4766, 0.4766]
```
