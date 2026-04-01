# Benchmark Summary

Seeds: 7, 42

## Aggregate Plots

![Mean final validation accuracy](mean_final_val_accuracy.png)

![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)

| Experiment | Type | Runs | Mean final val acc | Std final val acc | Mean best val acc | Mean adaptations | Mean final hidden dim | Best seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| condensenet-style-cifar10 | baseline | 2 | 0.4702 | 0.0520 | 0.5235 | 0.00 | - | 7 |
| fixed-cnn-cifar10 | baseline | 2 | 0.3982 | 0.0342 | 0.5465 | 0.00 | 0.0 | 42 |
| squeezenet-style-cifar10 | baseline | 2 | 0.3282 | 0.0248 | 0.3324 | 0.00 | - | 7 |

## Constraint Summary

| Experiment | Mean params | Mean nonzero params | Mean weight sparsity | Mean FLOP proxy | Mean activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| condensenet-style-cifar10 | 29418 | 29418 | 0.0000 | 11159424 | 67594 |
| fixed-cnn-cifar10 | 53186 | 53186 | 0.0000 | 10772746 | 10578 |
| squeezenet-style-cifar10 | 46594 | 46594 | 0.0000 | 4317184 | 66570 |

## Experiment Notes

- `condensenet-style-cifar10`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `fixed-cnn-cifar10`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `squeezenet-style-cifar10`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

## Per-Seed Results

### condensenet-style-cifar10
- seed 7: final=0.4182, best=0.5248, adaptations=0, params=29418, nonzero=29418, sparsity=0.0000
- seed 42: final=0.5222, best=0.5222, adaptations=0, params=29418, nonzero=29418, sparsity=0.0000

### fixed-cnn-cifar10
- seed 7: final=0.3640, best=0.5368, adaptations=0, params=53186, nonzero=53186, sparsity=0.0000
- seed 42: final=0.4324, best=0.5562, adaptations=0, params=53186, nonzero=53186, sparsity=0.0000

### squeezenet-style-cifar10
- seed 7: final=0.3530, best=0.3530, adaptations=0, params=46594, nonzero=46594, sparsity=0.0000
- seed 42: final=0.3034, best=0.3118, adaptations=0, params=46594, nonzero=46594, sparsity=0.0000

## Representative Stage Histories

### condensenet-style-cifar10 (best seed 7)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.41819998621940613

### fixed-cnn-cifar10 (best seed 42)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.4323999881744385

### squeezenet-style-cifar10 (best seed 7)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.3529999852180481

## Representative Architectures

### condensenet-style-cifar10 (best seed 7)
```mermaid
flowchart LR
    title["condensenet-style-cifar10"]
    input["Input (3x32x32)"]
    block_1["Conv 1 (out=32, k=3)"]
    pool_1["MaxPool 1"]
    block_2["Grouped 2 (out=48, g=4)"]
    block_3["Grouped 3 (out=64, g=4)"]
    pool_3["MaxPool 3"]
    block_4["Grouped 4 (out=96, g=8)"]
    classifier["Classifier (10)"]
    input --> block_1
    block_1 --> pool_1
    pool_1 --> block_2
    block_2 --> block_3
    block_3 --> pool_3
    pool_3 --> block_4
    block_4 --> classifier
```

### fixed-cnn-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["fixed-cnn-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (24, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (48, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (72, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    fc_hidden_1["FC 1 (128)"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> fc_hidden_1
    fc_hidden_1 --> output
```

### squeezenet-style-cifar10 (best seed 7)
```mermaid
flowchart LR
    title["squeezenet-style-cifar10"]
    input["Input (3x32x32)"]
    block_1["Conv 1 (out=32, k=3)"]
    pool_1["MaxPool 1"]
    block_2["Fire 2 (sq=24, out=48)"]
    pool_2["MaxPool 2"]
    block_3["Fire 3 (sq=32, out=64)"]
    block_4["Fire 4 (sq=48, out=96)"]
    classifier["Classifier (10)"]
    input --> block_1
    block_1 --> pool_1
    pool_1 --> block_2
    block_2 --> pool_2
    pool_2 --> block_3
    block_3 --> block_4
    block_4 --> classifier
```
