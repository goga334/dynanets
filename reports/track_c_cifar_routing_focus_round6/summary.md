# Benchmark Summary

Seeds: 7, 23, 42

## Aggregate Plots

![Mean validation accuracy by epoch](mean_validation_accuracy_by_epoch.png)

![Mean final validation accuracy](mean_final_val_accuracy.png)

![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)

![Mean parameter count](mean_parameter_count.png)

![Mean FLOP proxy](mean_forward_flop_proxy.png)

![Accuracy vs FLOP proxy](accuracy_vs_flop_proxy.png)

| Experiment | Type | Runs | Mean final val acc | Std final val acc | Mean best val acc | Mean adaptations | Mean final hidden dim | Best seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep-cnn-cifar10 | baseline | 3 | 0.5511 | 0.0194 | 0.5511 | 0.00 | 0.0 | 23 |
| wide-cnn-cifar10-bn | baseline | 3 | 0.5402 | 0.0430 | 0.5405 | 0.00 | 0.0 | 42 |
| network-slimming-cifar10 | workflow | 3 | 0.4627 | 0.0081 | 0.5289 | 1.00 | 0.0 | 23 |
| conditional-computation-cifar10 | workflow | 3 | 0.4273 | 0.0135 | 0.4320 | 0.00 | - | 7 |
| skipnet-cifar10 | workflow | 3 | 0.3677 | 0.0021 | 0.3703 | 0.00 | - | 42 |
| iamnn-cifar10 | workflow | 3 | 0.3617 | 0.0248 | 0.3871 | 0.00 | - | 23 |
| channel-gating-cifar10 | workflow | 3 | 0.3508 | 0.0238 | 0.3508 | 0.00 | - | 42 |
| instance-wise-sparsity-cifar10 | workflow | 3 | 0.3384 | 0.0154 | 0.3433 | 0.00 | - | 42 |
| dynamic-slimmable-cifar10 | workflow | 3 | 0.3272 | 0.0156 | 0.3332 | 0.00 | - | 23 |

## Constraint Summary

| Experiment | Mean params | Mean nonzero params | Mean weight sparsity | Mean FLOP proxy | Mean activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep-cnn-cifar10 | 76066 | 76066 | 0.0000 | 10818346 | 10738 |
| wide-cnn-cifar10-bn | 92298 | 92298 | 0.0000 | 18498346 | 14090 |
| network-slimming-cifar10 | 37111 | 37111 | 0.0000 | 7434602 | 8740 |
| conditional-computation-cifar10 | 117730 | 117730 | 0.0000 | 28105450 | 17410 |
| skipnet-cifar10 | 75754 | 75754 | 0.0000 | 18355594 | 13930 |
| iamnn-cifar10 | 71866 | 71866 | 0.0000 | 19982890 | 15058 |
| channel-gating-cifar10 | 75754 | 75754 | 0.0000 | 18355594 | 13930 |
| instance-wise-sparsity-cifar10 | 46538 | 46538 | 0.0000 | 11113290 | 10586 |
| dynamic-slimmable-cifar10 | 59258 | 59258 | 0.0000 | 32107722 | 21626 |

## Experiment Notes

- `deep-cnn-cifar10`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `wide-cnn-cifar10-bn`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `network-slimming-cifar10`: workflow=network_slimming; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `conditional-computation-cifar10`: workflow=conditional_computation; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.92, 'target_accept_rate': 0.08, 'early_exit_fraction': 0.0809, 'eligible_fraction': 0.1103, 'mean_gate_score': 0.0082, 'max_gate_score': 0.0325, 'mean_exit_confidence': 0.314, 'full_path_fraction': 0.9191, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.9199}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `skipnet-cifar10`: workflow=skipnet; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.21, 'target_cost_ratio': 0.9, 'target_accept_rate': 0.1, 'early_exit_fraction': 0.2794, 'eligible_fraction': 0.5515, 'mean_gate_score': 0.1841, 'max_gate_score': 0.4114, 'mean_exit_confidence': 0.3499, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'early'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'early'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7929}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `iamnn-cifar10`: workflow=iamnn; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.2, 'target_cost_ratio': 0.72, 'target_accept_rate': 0.12, 'early_exit_fraction': 0.1176, 'eligible_fraction': 0.6618, 'mean_gate_score': 0.0541, 'max_gate_score': 0.1033, 'mean_exit_confidence': 0.4435, 'full_path_fraction': 0.8824, 'trace_samples': [{'sample': 0, 'path': 'early'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.9226}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `channel-gating-cifar10`: workflow=channel_gating; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.2, 'target_cost_ratio': 0.88, 'target_accept_rate': 0.44, 'stage_target_accept_rates': {'0.75': 0.1515, '0.875': 0.28, '1.0': None}, 'route_counts': {'0.75': 21, '0.875': 32, '1.0': 83}, 'trace_samples': [{'sample': 0, 'width': 0.875}, {'sample': 1, 'width': 0.875}, {'sample': 2, 'width': 0.75}, {'sample': 3, 'width': 0.875}, {'sample': 4, 'width': 0.75}, {'sample': 5, 'width': 1.0}, {'sample': 6, 'width': 1.0}, {'sample': 7, 'width': 1.0}], 'mean_width': 0.932, 'mean_cost_ratio': 0.8779}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `instance-wise-sparsity-cifar10`: workflow=instance_wise_sparsity; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.18, 'target_cost_ratio': 0.68, 'target_accept_rate': 0.4, 'stage_target_accept_rates': {'0.75': 0.1633, '0.875': 0.3}, 'route_counts': {'0.75': 135, '0.875': 1, '1.0': 0}, 'trace_samples': [{'sample': 0, 'width': 0.75}, {'sample': 1, 'width': 0.75}, {'sample': 2, 'width': 0.75}, {'sample': 3, 'width': 0.75}, {'sample': 4, 'width': 0.75}, {'sample': 5, 'width': 0.75}, {'sample': 6, 'width': 0.75}, {'sample': 7, 'width': 0.75}], 'mean_width': 0.7509, 'mean_cost_ratio': 0.5667}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `dynamic-slimmable-cifar10`: workflow=dynamic_slimmable; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.9, 'target_accept_rate': 0.4, 'stage_target_accept_rates': {'0.75': 0.1282, '0.875': 0.24, '1.0': None}, 'route_counts': {'0.75': 17, '0.875': 29, '1.0': 90}, 'trace_samples': [{'sample': 0, 'width': 1.0}, {'sample': 1, 'width': 0.875}, {'sample': 2, 'width': 0.75}, {'sample': 3, 'width': 1.0}, {'sample': 4, 'width': 0.875}, {'sample': 5, 'width': 0.875}, {'sample': 6, 'width': 1.0}, {'sample': 7, 'width': 1.0}], 'mean_width': 0.9421, 'mean_cost_ratio': 0.8966}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

## Per-Seed Results

### deep-cnn-cifar10
- seed 7: final=0.5560, best=0.5560, adaptations=0, params=76066, nonzero=76066, sparsity=0.0000
- seed 23: final=0.5720, best=0.5720, adaptations=0, params=76066, nonzero=76066, sparsity=0.0000
- seed 42: final=0.5252, best=0.5252, adaptations=0, params=76066, nonzero=76066, sparsity=0.0000

### wide-cnn-cifar10-bn
- seed 7: final=0.5692, best=0.5692, adaptations=0, params=92298, nonzero=92298, sparsity=0.0000
- seed 23: final=0.4794, best=0.4802, adaptations=0, params=92298, nonzero=92298, sparsity=0.0000
- seed 42: final=0.5720, best=0.5720, adaptations=0, params=92298, nonzero=92298, sparsity=0.0000

### network-slimming-cifar10
- seed 7: final=0.4522, best=0.5322, adaptations=1, params=37111, nonzero=37111, sparsity=0.0000
- seed 23: final=0.4640, best=0.5516, adaptations=1, params=37111, nonzero=37111, sparsity=0.0000
- seed 42: final=0.4718, best=0.5030, adaptations=1, params=37111, nonzero=37111, sparsity=0.0000

### conditional-computation-cifar10
- seed 7: final=0.4464, best=0.4464, adaptations=0, params=117730, nonzero=117730, sparsity=0.0000
- seed 23: final=0.4176, best=0.4296, adaptations=0, params=117730, nonzero=117730, sparsity=0.0000
- seed 42: final=0.4178, best=0.4200, adaptations=0, params=117730, nonzero=117730, sparsity=0.0000

### skipnet-cifar10
- seed 7: final=0.3648, best=0.3654, adaptations=0, params=75754, nonzero=75754, sparsity=0.0000
- seed 23: final=0.3690, best=0.3690, adaptations=0, params=75754, nonzero=75754, sparsity=0.0000
- seed 42: final=0.3694, best=0.3764, adaptations=0, params=75754, nonzero=75754, sparsity=0.0000

### iamnn-cifar10
- seed 7: final=0.3852, best=0.3852, adaptations=0, params=71866, nonzero=71866, sparsity=0.0000
- seed 23: final=0.3726, best=0.3908, adaptations=0, params=71866, nonzero=71866, sparsity=0.0000
- seed 42: final=0.3274, best=0.3852, adaptations=0, params=71866, nonzero=71866, sparsity=0.0000

### channel-gating-cifar10
- seed 7: final=0.3500, best=0.3500, adaptations=0, params=75754, nonzero=75754, sparsity=0.0000
- seed 23: final=0.3220, best=0.3220, adaptations=0, params=75754, nonzero=75754, sparsity=0.0000
- seed 42: final=0.3804, best=0.3804, adaptations=0, params=75754, nonzero=75754, sparsity=0.0000

### instance-wise-sparsity-cifar10
- seed 7: final=0.3190, best=0.3336, adaptations=0, params=46538, nonzero=46538, sparsity=0.0000
- seed 23: final=0.3396, best=0.3396, adaptations=0, params=46538, nonzero=46538, sparsity=0.0000
- seed 42: final=0.3566, best=0.3566, adaptations=0, params=46538, nonzero=46538, sparsity=0.0000

### dynamic-slimmable-cifar10
- seed 7: final=0.3074, best=0.3254, adaptations=0, params=59258, nonzero=59258, sparsity=0.0000
- seed 23: final=0.3454, best=0.3454, adaptations=0, params=59258, nonzero=59258, sparsity=0.0000
- seed 42: final=0.3288, best=0.3288, adaptations=0, params=59258, nonzero=59258, sparsity=0.0000

## Representative Stage Histories

### deep-cnn-cifar10 (best seed 23)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5720000267028809

### wide-cnn-cifar10-bn (best seed 42)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5720000267028809

### network-slimming-cifar10 (best seed 23)
- network_slimming_sparse_train: epochs=5, range=1..5, adaptation_enabled=False, final_val=0.44339999556541443
- network_slimming_finetune: epochs=3, range=6..8, adaptation_enabled=False, final_val=0.46399998664855957

### conditional-computation-cifar10 (best seed 7)
- conditional_computation_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.382999986410141
- conditional_computation_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.446399986743927

### skipnet-cifar10 (best seed 42)
- skipnet_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.3449999988079071
- skipnet_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.3693999946117401

### iamnn-cifar10 (best seed 23)
- iamnn_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.335999995470047
- iamnn_routing: epochs=2, range=5..6, adaptation_enabled=False, final_val=0.35740000009536743
- iamnn_consolidation: epochs=2, range=7..8, adaptation_enabled=False, final_val=0.3725999891757965

### channel-gating-cifar10 (best seed 42)
- channel_gating_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.29019999504089355
- channel_gating_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.38040000200271606

### instance-wise-sparsity-cifar10 (best seed 42)
- instance_wise_sparsity_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.28619998693466187
- instance_wise_sparsity_routing: epochs=2, range=5..6, adaptation_enabled=False, final_val=0.32359999418258667
- instance_wise_sparsity_consolidation: epochs=2, range=7..8, adaptation_enabled=False, final_val=0.35659998655319214

### dynamic-slimmable-cifar10 (best seed 23)
- dynamic_slimmable_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.29260000586509705
- dynamic_slimmable_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.34540000557899475

## Representative Architectures

### deep-cnn-cifar10 (best seed 23)
```mermaid
flowchart LR
    title["deep-cnn-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (24, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (48, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (72, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    fc_hidden_1["FC 1 (192)"]
    fc_hidden_2["FC 2 (96)"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> fc_hidden_1
    fc_hidden_1 --> fc_hidden_2
    fc_hidden_2 --> output
```

### wide-cnn-cifar10-bn (best seed 42)
```mermaid
flowchart LR
    title["wide-cnn-cifar10-bn"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (32, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (64, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (96, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    fc_hidden_1["FC 1 (160)"]
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

### network-slimming-cifar10 (best seed 23)
```mermaid
flowchart LR
    title["network-slimming-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (20, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (39, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (58, k=3)"]
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

### conditional-computation-cifar10 (best seed 7)
```mermaid
flowchart LR
    title["conditional-computation-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (40, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (80, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (120, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> output
```

### skipnet-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["skipnet-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (32, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (64, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (96, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> output
```

### iamnn-cifar10 (best seed 23)
```mermaid
flowchart LR
    title["iamnn-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (36, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (72, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (72, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> output
```

### channel-gating-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["channel-gating-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (32, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (64, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (96, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> output
```

### instance-wise-sparsity-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["instance-wise-sparsity-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (24, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (48, k=3)"]
    pool_2["MaxPool 2"]
    conv_3["Conv 3 (80, k=3)"]
    pool_3["MaxPool 3"]
    global_pool["GlobalPool"]
    output["Output (10)"]
    input --> conv_1
    conv_1 --> pool_1
    pool_1 --> conv_2
    conv_2 --> pool_2
    pool_2 --> conv_3
    conv_3 --> pool_3
    pool_3 --> global_pool
    global_pool --> output
```

### dynamic-slimmable-cifar10 (best seed 23)
```mermaid
flowchart LR
    title["dynamic-slimmable-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (56, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (112, k=3)"]
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
