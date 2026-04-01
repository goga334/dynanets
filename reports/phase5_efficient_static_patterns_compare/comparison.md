# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-cnn-patterns | baseline | 25 | 0.1125 | 0.1000 | 0.2075 | 0 | 0 |
| wide-cnn-patterns-bn | baseline | 30 | 0.7025 | 0.3725 | 0.3925 | 0 | 0 |
| squeezenet-style-patterns | baseline | 8 | 0.9067 | 0.8867 | 0.8867 | 0 | - |
| condensenet-style-patterns | baseline | 8 | 0.9092 | 0.8867 | 0.8867 | 0 | - |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `fixed-cnn-patterns`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `wide-cnn-patterns-bn`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `squeezenet-style-patterns`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `condensenet-style-patterns`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

## Constraint Summary

| Experiment | Params | Nonzero params | Weight sparsity | FLOP proxy | Activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-cnn-patterns | 7562 | 7562 | 0.0000 | 2061098 | 4810 |
| wide-cnn-patterns-bn | 16474 | 16474 | 0.0000 | 4505914 | 7210 |
| squeezenet-style-patterns | 22354 | 22354 | 0.0000 | 1292032 | 36466 |
| condensenet-style-patterns | 22034 | 22034 | 0.0000 | 5457920 | 37642 |

## Workflow Stages

### fixed-cnn-patterns
- train: epochs=25, range=1..25, adaptation_enabled=False, final_val=0.09999999403953552
- workflow_metadata={'configured_total_epochs': 25, 'executed_total_epochs': 25, 'stage_count': 1}

### wide-cnn-patterns-bn
- train: epochs=30, range=1..30, adaptation_enabled=False, final_val=0.3725000023841858
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 1}

### squeezenet-style-patterns
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.88671875
- workflow_metadata={'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1}

### condensenet-style-patterns
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.88671875
- workflow_metadata={'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1}


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

### wide-cnn-patterns-bn
```mermaid
flowchart LR
    title["wide-cnn-patterns-bn"]
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

### squeezenet-style-patterns
```mermaid
flowchart LR
    title["squeezenet-style-patterns"]
    input["Input (1x28x28)"]
    block_1["Conv 1 (out=24, k=3)"]
    pool_1["MaxPool 1"]
    block_2["Fire 2 (sq=16, out=32)"]
    pool_2["MaxPool 2"]
    block_3["Fire 3 (sq=24, out=48)"]
    block_4["Fire 4 (sq=32, out=64)"]
    classifier["Classifier (10)"]
    input --> block_1
    block_1 --> pool_1
    pool_1 --> block_2
    block_2 --> pool_2
    pool_2 --> block_3
    block_3 --> block_4
    block_4 --> classifier
```

### condensenet-style-patterns
```mermaid
flowchart LR
    title["condensenet-style-patterns"]
    input["Input (1x28x28)"]
    block_1["Conv 1 (out=24, k=3)"]
    pool_1["MaxPool 1"]
    block_2["Grouped 2 (out=32, g=2)"]
    block_3["Grouped 3 (out=48, g=4)"]
    pool_3["MaxPool 3"]
    block_4["Grouped 4 (out=64, g=4)"]
    classifier["Classifier (10)"]
    input --> block_1
    block_1 --> pool_1
    pool_1 --> block_2
    block_2 --> block_3
    block_3 --> pool_3
    pool_3 --> block_4
    block_4 --> classifier
```

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    y-axis "Accuracy" 0 --> 1.0
    line "fixed-cnn-patterns" [0.1075, 0.2075, 0.2075, 0.2075, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000]
    line "wide-cnn-patterns-bn" [0.0800, 0.0000, 0.0875, 0.0950, 0.0950, 0.0950, 0.0950, 0.0950, 0.0950, 0.2000, 0.2000, 0.2000, 0.2000, 0.1875, 0.1875, 0.2925, 0.2925, 0.3750, 0.3925, 0.3925, 0.3125, 0.2875, 0.2875, 0.2875, 0.2875, 0.2875, 0.3575, 0.3725, 0.3725, 0.3725]
    line "squeezenet-style-patterns" [0.0723, 0.0723, 0.2734, 0.4980, 0.8359, 0.8594, 0.8867, 0.8867]
    line "condensenet-style-patterns" [0.1230, 0.1230, 0.8867, 0.8867, 0.8867, 0.8867, 0.8867, 0.8867]
```
