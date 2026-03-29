# Benchmark Summary

Seeds: 7, 11

## Aggregate Plots

![Mean final validation accuracy](mean_final_val_accuracy.png)

![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)

| Experiment | Type | Runs | Mean final val acc | Std final val acc | Mean best val acc | Mean adaptations | Mean final hidden dim | Best seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gradmax-spirals10 | dynamic | 2 | 0.6153 | 0.0447 | 0.6533 | 2.00 | 14.0 | 7 |
| fixed-mlp-spirals10 | baseline | 2 | 0.6120 | 0.0147 | 0.6540 | 0.00 | - | 7 |

## Per-Seed Results

### gradmax-spirals10
- seed 7: final=0.6600, best=0.6947, adaptations=2
- seed 11: final=0.5707, best=0.6120, adaptations=2

### fixed-mlp-spirals10
- seed 7: final=0.5973, best=0.6680, adaptations=0
- seed 11: final=0.6267, best=0.6400, adaptations=0

## Representative Architectures

### gradmax-spirals10 (best seed 7)
```mermaid
flowchart LR
    title["gradmax-spirals10"]
    input["Input (10)"]
    hidden_1["Hidden 1 (14)"]
    output["Output (2)"]
    input --> hidden_1
    hidden_1 --> output
```

### fixed-mlp-spirals10 (best seed 7)
```mermaid
flowchart LR
    title["fixed-mlp-spirals10"]
    input["Input (10)"]
    hidden_1["Hidden 1 (18)"]
    output["Output (2)"]
    input --> hidden_1
    hidden_1 --> output
```
