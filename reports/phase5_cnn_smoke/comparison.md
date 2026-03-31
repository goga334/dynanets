# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-cnn-patterns | baseline | 25 | 0.2113 | 0.2125 | 0.2125 | 0 | - |
| wide-cnn-patterns | baseline | 25 | 0.6006 | 0.5100 | 0.6375 | 0 | - |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `fixed-cnn-patterns`: device=cpu; requested_device=auto; torch=2.10.0+cpu; cuda_available=False
- `wide-cnn-patterns`: device=cpu; requested_device=auto; torch=2.10.0+cpu; cuda_available=False

## Workflow Stages

### fixed-cnn-patterns
- train: epochs=25, range=1..25, adaptation_enabled=False, final_val=0.21250000596046448
- workflow_metadata={'configured_total_epochs': 25, 'executed_total_epochs': 25, 'stage_count': 1}

### wide-cnn-patterns
- train: epochs=25, range=1..25, adaptation_enabled=False, final_val=0.5099999904632568
- workflow_metadata={'configured_total_epochs': 25, 'executed_total_epochs': 25, 'stage_count': 1}


## Adaptation Timeline

## Architecture Graphs

### fixed-cnn-patterns
```mermaid
flowchart LR
    title["fixed-cnn-patterns"]
    input["Input (1x28x28)"]
    conv_1["Conv 1 (16, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (32, k=3)"]
    pool_2["MaxPool 2"]
    global_pool["GlobalPool"]
    fc_hidden_1["FC 1 (64)"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> global_pool
    global_pool --> fc_hidden_1
    fc_hidden_1 --> output
```

### wide-cnn-patterns
```mermaid
flowchart LR
    title["wide-cnn-patterns"]
    input["Input (1x28x28)"]
    conv_1["Conv 1 (24, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (48, k=3)"]
    pool_2["MaxPool 2"]
    global_pool["GlobalPool"]
    fc_hidden_1["FC 1 (96)"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> global_pool
    global_pool --> fc_hidden_1
    fc_hidden_1 --> output
```

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    y-axis "Accuracy" 0 --> 1.0
    line "fixed-cnn-patterns" [0.2075, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.2125, 0.2125, 0.2125, 0.2125, 0.2125]
    line "wide-cnn-patterns" [0.0800, 0.1800, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.2050, 0.2050, 0.2050, 0.2050, 0.2050, 0.2050, 0.3175, 0.3175, 0.4100, 0.5075, 0.5075, 0.5900, 0.6375, 0.5925, 0.5100]
```
