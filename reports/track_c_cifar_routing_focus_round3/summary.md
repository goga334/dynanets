# Benchmark Summary

Seeds: 7, 23, 42

## Aggregate Plots

![Mean validation accuracy by epoch](mean_validation_accuracy_by_epoch.png)

![Mean final validation accuracy](mean_final_val_accuracy.png)

![Per-seed final validation accuracy](per_seed_final_val_accuracy.png)

| Experiment | Type | Runs | Mean final val acc | Std final val acc | Mean best val acc | Mean adaptations | Mean final hidden dim | Best seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep-cnn-cifar10 | baseline | 3 | 0.5511 | 0.0194 | 0.5511 | 0.00 | 0.0 | 23 |
| wide-cnn-cifar10-bn | baseline | 3 | 0.5402 | 0.0430 | 0.5405 | 0.00 | 0.0 | 42 |
| network-slimming-cifar10 | workflow | 3 | 0.4627 | 0.0081 | 0.5289 | 1.00 | 0.0 | 23 |
| skipnet-cifar10 | workflow | 3 | 0.3277 | 0.0096 | 0.3277 | 0.00 | - | 42 |
| conditional-computation-cifar10 | workflow | 3 | 0.3253 | 0.0127 | 0.3253 | 0.00 | - | 42 |
| iamnn-cifar10 | workflow | 3 | 0.3226 | 0.0132 | 0.3226 | 0.00 | - | 42 |
| dynamic-slimmable-cifar10 | workflow | 3 | 0.3189 | 0.0074 | 0.3204 | 0.00 | - | 42 |
| channel-gating-cifar10 | workflow | 3 | 0.3173 | 0.0073 | 0.3179 | 0.00 | - | 42 |
| instance-wise-sparsity-cifar10 | workflow | 3 | 0.3160 | 0.0129 | 0.3179 | 0.00 | - | 42 |

## Constraint Summary

| Experiment | Mean params | Mean nonzero params | Mean weight sparsity | Mean FLOP proxy | Mean activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep-cnn-cifar10 | 76066 | 76066 | 0.0000 | 10818346 | 10738 |
| wide-cnn-cifar10-bn | 92298 | 92298 | 0.0000 | 18498346 | 14090 |
| network-slimming-cifar10 | 37111 | 37111 | 0.0000 | 7434602 | 8740 |
| skipnet-cifar10 | 43882 | 43882 | 0.0000 | 23981962 | 18538 |
| conditional-computation-cifar10 | 43882 | 43882 | 0.0000 | 23981962 | 18538 |
| iamnn-cifar10 | 43882 | 43882 | 0.0000 | 23981962 | 18538 |
| dynamic-slimmable-cifar10 | 43882 | 43882 | 0.0000 | 23981962 | 18538 |
| channel-gating-cifar10 | 43882 | 43882 | 0.0000 | 23981962 | 18538 |
| instance-wise-sparsity-cifar10 | 43882 | 43882 | 0.0000 | 23981962 | 18538 |

## Experiment Notes

- `deep-cnn-cifar10`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `wide-cnn-cifar10-bn`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `network-slimming-cifar10`: workflow=network_slimming; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `skipnet-cifar10`: workflow=skipnet; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.21, 'target_cost_ratio': 0.9, 'target_accept_rate': 0.1, 'early_exit_fraction': 0.1029, 'eligible_fraction': 0.1838, 'mean_gate_score': 0.011, 'max_gate_score': 0.0871, 'mean_exit_confidence': 0.3298, 'full_path_fraction': 0.8971, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.9002}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `conditional-computation-cifar10`: workflow=conditional_computation; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.92, 'target_accept_rate': 0.08, 'early_exit_fraction': 0.0809, 'eligible_fraction': 0.1765, 'mean_gate_score': 0.0101, 'max_gate_score': 0.0767, 'mean_exit_confidence': 0.3422, 'full_path_fraction': 0.9191, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.9216}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `iamnn-cifar10`: workflow=iamnn; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.2, 'target_cost_ratio': 0.72, 'target_accept_rate': 0.12, 'early_exit_fraction': 0.1176, 'eligible_fraction': 0.2059, 'mean_gate_score': 0.0119, 'max_gate_score': 0.0958, 'mean_exit_confidence': 0.3317, 'full_path_fraction': 0.8824, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.8859}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `dynamic-slimmable-cifar10`: workflow=dynamic_slimmable; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.9, 'target_accept_rate': 0.4, 'stage_target_accept_rates': {'0.75': 0.1282, '0.875': 0.24, '1.0': None}, 'route_counts': {'0.75': 17, '0.875': 29, '1.0': 90}, 'trace_samples': [{'sample': 0, 'width': 0.75}, {'sample': 1, 'width': 1.0}, {'sample': 2, 'width': 0.75}, {'sample': 3, 'width': 1.0}, {'sample': 4, 'width': 0.875}, {'sample': 5, 'width': 0.875}, {'sample': 6, 'width': 0.875}, {'sample': 7, 'width': 1.0}], 'mean_width': 0.9421, 'mean_cost_ratio': 0.8968}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `channel-gating-cifar10`: workflow=channel_gating; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.2, 'target_cost_ratio': 0.88, 'target_accept_rate': 0.44, 'stage_target_accept_rates': {'0.75': 0.1515, '0.875': 0.28, '1.0': None}, 'route_counts': {'0.75': 21, '0.875': 32, '1.0': 83}, 'trace_samples': [{'sample': 0, 'width': 0.75}, {'sample': 1, 'width': 1.0}, {'sample': 2, 'width': 0.75}, {'sample': 3, 'width': 1.0}, {'sample': 4, 'width': 0.875}, {'sample': 5, 'width': 0.875}, {'sample': 6, 'width': 0.75}, {'sample': 7, 'width': 1.0}], 'mean_width': 0.932, 'mean_cost_ratio': 0.879}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `instance-wise-sparsity-cifar10`: workflow=instance_wise_sparsity; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.18, 'target_cost_ratio': 0.68, 'target_accept_rate': 0.4, 'stage_target_accept_rates': {'0.75': 0.1633, '0.875': 0.3, '1.0': None}, 'route_counts': {'0.75': 22, '0.875': 34, '1.0': 80}, 'trace_samples': [{'sample': 0, 'width': 0.875}, {'sample': 1, 'width': 1.0}, {'sample': 2, 'width': 0.75}, {'sample': 3, 'width': 1.0}, {'sample': 4, 'width': 0.75}, {'sample': 5, 'width': 0.875}, {'sample': 6, 'width': 0.875}, {'sample': 7, 'width': 1.0}], 'mean_width': 0.9283, 'mean_cost_ratio': 0.8724}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

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

### skipnet-cifar10
- seed 7: final=0.3184, best=0.3184, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 23: final=0.3238, best=0.3238, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 42: final=0.3410, best=0.3410, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000

### conditional-computation-cifar10
- seed 7: final=0.3152, best=0.3152, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 23: final=0.3174, best=0.3174, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 42: final=0.3432, best=0.3432, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000

### iamnn-cifar10
- seed 7: final=0.3146, best=0.3146, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 23: final=0.3120, best=0.3120, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 42: final=0.3412, best=0.3412, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000

### dynamic-slimmable-cifar10
- seed 7: final=0.3138, best=0.3138, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 23: final=0.3134, best=0.3160, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 42: final=0.3294, best=0.3314, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000

### channel-gating-cifar10
- seed 7: final=0.3114, best=0.3114, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 23: final=0.3130, best=0.3134, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 42: final=0.3276, best=0.3288, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000

### instance-wise-sparsity-cifar10
- seed 7: final=0.3010, best=0.3034, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 23: final=0.3146, best=0.3180, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000
- seed 42: final=0.3324, best=0.3324, adaptations=0, params=43882, nonzero=43882, sparsity=0.0000

## Representative Stage Histories

### deep-cnn-cifar10 (best seed 23)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5720000267028809

### wide-cnn-cifar10-bn (best seed 42)
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5720000267028809

### network-slimming-cifar10 (best seed 23)
- network_slimming_sparse_train: epochs=5, range=1..5, adaptation_enabled=False, final_val=0.44339999556541443
- network_slimming_finetune: epochs=3, range=6..8, adaptation_enabled=False, final_val=0.46399998664855957

### skipnet-cifar10 (best seed 42)
- skipnet_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.2930000126361847
- skipnet_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.3409999907016754

### conditional-computation-cifar10 (best seed 42)
- conditional_computation_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.29120001196861267
- conditional_computation_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.3431999981403351

### iamnn-cifar10 (best seed 42)
- iamnn_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.2948000133037567
- iamnn_routing: epochs=2, range=5..6, adaptation_enabled=False, final_val=0.32580000162124634
- iamnn_consolidation: epochs=2, range=7..8, adaptation_enabled=False, final_val=0.34119999408721924

### dynamic-slimmable-cifar10 (best seed 42)
- dynamic_slimmable_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.2750000059604645
- dynamic_slimmable_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.3294000029563904

### channel-gating-cifar10 (best seed 42)
- channel_gating_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.2757999897003174
- channel_gating_routing: epochs=4, range=5..8, adaptation_enabled=False, final_val=0.32760000228881836

### instance-wise-sparsity-cifar10 (best seed 42)
- instance_wise_sparsity_warmup: epochs=4, range=1..4, adaptation_enabled=False, final_val=0.27160000801086426
- instance_wise_sparsity_routing: epochs=2, range=5..6, adaptation_enabled=False, final_val=0.3149999976158142
- instance_wise_sparsity_consolidation: epochs=2, range=7..8, adaptation_enabled=False, final_val=0.33239999413490295

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

### skipnet-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["skipnet-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (48, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (96, k=3)"]
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

### conditional-computation-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["conditional-computation-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (48, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (96, k=3)"]
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

### iamnn-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["iamnn-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (48, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (96, k=3)"]
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

### dynamic-slimmable-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["dynamic-slimmable-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (48, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (96, k=3)"]
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

### channel-gating-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["channel-gating-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (48, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (96, k=3)"]
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

### instance-wise-sparsity-cifar10 (best seed 42)
```mermaid
flowchart LR
    title["instance-wise-sparsity-cifar10"]
    input["Input (3x32x32)"]
    conv_1["Conv 1 (48, k=3)"]
    pool_1["MaxPool 1"]
    conv_2["Conv 2 (96, k=3)"]
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
