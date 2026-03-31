# Benchmark Summary

Seeds: 7, 11, 23, 42, 99

## Aggregate Plots

![Mean final validation accuracy](mean_final_val_accuracy.png)

![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)

| Experiment | Type | Runs | Mean final val acc | Std final val acc | Mean best val acc | Mean adaptations | Mean final hidden dim | Best seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| network-slimming-mnist | workflow | 5 | 0.8029 | 0.0815 | 0.8364 | 1.00 | 0.0 | 7 |
| wide-cnn-mnist-bn | baseline | 5 | 0.7938 | 0.0496 | 0.8365 | 0.00 | 0.0 | 99 |
| conditional-computation-mnist | workflow | 5 | 0.5926 | 0.0285 | 0.5940 | 0.00 | - | 42 |
| dynamic-slimmable-mnist | workflow | 5 | 0.5826 | 0.0290 | 0.5842 | 0.00 | - | 7 |
| fixed-cnn-mnist | baseline | 5 | 0.4753 | 0.0092 | 0.4753 | 0.00 | 0.0 | 99 |

## Constraint Summary

| Experiment | Mean params | Mean nonzero params | Mean weight sparsity | Mean FLOP proxy | Mean activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| network-slimming-mnist | 12187 | 12187 | 0.0000 | 3119397 | 5976 |
| wide-cnn-mnist-bn | 16474 | 16474 | 0.0000 | 4505914 | 7210 |
| conditional-computation-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |
| dynamic-slimmable-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |
| fixed-cnn-mnist | 7562 | 7562 | 0.0000 | 2061098 | 4810 |

## Experiment Notes

- `network-slimming-mnist`: workflow=network_slimming; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `wide-cnn-mnist-bn`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `conditional-computation-mnist`: workflow=conditional_computation; route_summary={'policy': 'early_exit', 'mode': 'eval', 'confidence_threshold': 0.9, 'early_exit_fraction': 0.0, 'full_path_fraction': 1.0, 'mean_width': 1.0, 'mean_cost_ratio': 1.0}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `dynamic-slimmable-mnist`: workflow=dynamic_slimmable; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'confidence_threshold': 0.85, 'route_counts': {'0.5': 9, '0.75': 8, '1.0': 119}, 'mean_width': 0.9522, 'mean_cost_ratio': 0.9252}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `fixed-cnn-mnist`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

## Per-Seed Results

### network-slimming-mnist
- seed 7: final=0.8568, best=0.8568, adaptations=1, params=12187, nonzero=12187, sparsity=0.0000
- seed 11: final=0.8444, best=0.8444, adaptations=1, params=12187, nonzero=12187, sparsity=0.0000
- seed 23: final=0.8490, best=0.8490, adaptations=1, params=12187, nonzero=12187, sparsity=0.0000
- seed 42: final=0.8228, best=0.8228, adaptations=1, params=12187, nonzero=12187, sparsity=0.0000
- seed 99: final=0.6414, best=0.8090, adaptations=1, params=12187, nonzero=12187, sparsity=0.0000

### wide-cnn-mnist-bn
- seed 7: final=0.8178, best=0.8444, adaptations=0, params=16474, nonzero=16474, sparsity=0.0000
- seed 11: final=0.7536, best=0.8002, adaptations=0, params=16474, nonzero=16474, sparsity=0.0000
- seed 23: final=0.8292, best=0.8292, adaptations=0, params=16474, nonzero=16474, sparsity=0.0000
- seed 42: final=0.8502, best=0.8502, adaptations=0, params=16474, nonzero=16474, sparsity=0.0000
- seed 99: final=0.7184, best=0.8586, adaptations=0, params=16474, nonzero=16474, sparsity=0.0000

### conditional-computation-mnist
- seed 7: final=0.6108, best=0.6108, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 11: final=0.6110, best=0.6110, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 23: final=0.5602, best=0.5602, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 42: final=0.6246, best=0.6246, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 99: final=0.5562, best=0.5632, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000

### dynamic-slimmable-mnist
- seed 7: final=0.6148, best=0.6148, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 11: final=0.6094, best=0.6094, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 23: final=0.5516, best=0.5516, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 42: final=0.5920, best=0.5920, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000
- seed 99: final=0.5452, best=0.5534, adaptations=0, params=11146, nonzero=11146, sparsity=0.0000

### fixed-cnn-mnist
- seed 7: final=0.4776, best=0.4776, adaptations=0, params=7562, nonzero=7562, sparsity=0.0000
- seed 11: final=0.4742, best=0.4742, adaptations=0, params=7562, nonzero=7562, sparsity=0.0000
- seed 23: final=0.4638, best=0.4638, adaptations=0, params=7562, nonzero=7562, sparsity=0.0000
- seed 42: final=0.4696, best=0.4696, adaptations=0, params=7562, nonzero=7562, sparsity=0.0000
- seed 99: final=0.4912, best=0.4912, adaptations=0, params=7562, nonzero=7562, sparsity=0.0000

## Representative Stage Histories

### network-slimming-mnist (best seed 7)
- network_slimming_sparse_train: epochs=5, range=1..5, adaptation_enabled=False, final_val=0.8144000172615051
- network_slimming_finetune: epochs=3, range=6..8, adaptation_enabled=False, final_val=0.8568000197410583

### wide-cnn-mnist-bn (best seed 99)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.7184000015258789

### conditional-computation-mnist (best seed 42)
- conditional_computation_train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.6245999932289124

### dynamic-slimmable-mnist (best seed 7)
- dynamic_slimmable_train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.614799976348877

### fixed-cnn-mnist (best seed 99)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.4912000000476837

## Representative Architectures

### network-slimming-mnist (best seed 7)
```mermaid
flowchart LR
    title["network-slimming-mnist"]
    input["Input (1x28x28)"]
    conv_1["Conv 1 (20, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (39, k=3)"]
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

### wide-cnn-mnist-bn (best seed 99)
```mermaid
flowchart LR
    title["wide-cnn-mnist-bn"]
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

### conditional-computation-mnist (best seed 42)
```mermaid
flowchart LR
    title["conditional-computation-mnist"]
    input["Input (1x28x28)"]
    conv_1["Conv 1 (24, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (48, k=3)"]
    pool_2["MaxPool 2"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> global_pool
    global_pool --> output
```

### dynamic-slimmable-mnist (best seed 7)
```mermaid
flowchart LR
    title["dynamic-slimmable-mnist"]
    input["Input (1x28x28)"]
    conv_1["Conv 1 (24, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (48, k=3)"]
    pool_2["MaxPool 2"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> global_pool
    global_pool --> output
```

### fixed-cnn-mnist (best seed 99)
```mermaid
flowchart LR
    title["fixed-cnn-mnist"]
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
