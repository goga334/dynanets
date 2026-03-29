# Baseline Comparison

| Experiment | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-baseline | 6 | 0.5098 | 0.4766 | 0.4766 | 0 | - |
| minimal-experiment | 6 | 0.4941 | 0.5625 | 0.8594 | 3 | 20 |

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6]
    y-axis "Accuracy" 0 --> 1.0
    line "fixed-mlp-baseline" [0.4766, 0.4766, 0.4766, 0.4766, 0.4766, 0.4766]
    line "minimal-experiment" [0.6875, 0.8594, 0.3672, 0.5000, 0.5391, 0.5625]
```
