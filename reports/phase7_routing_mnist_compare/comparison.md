# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-cnn-mnist | baseline | 8 | 0.4566 | 0.4696 | 0.4696 | 0 | 0 |
| wide-cnn-mnist-bn | baseline | 8 | 0.8703 | 0.8502 | 0.8502 | 0 | 0 |
| dynamic-slimmable-mnist | workflow | 8 | 0.5783 | 0.5920 | 0.5920 | 0 | - |
| conditional-computation-mnist | workflow | 8 | 0.6029 | 0.6246 | 0.6246 | 0 | - |
| network-slimming-mnist | workflow | 8 | 0.8281 | 0.8228 | 0.8228 | 1 | 0 |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `fixed-cnn-mnist`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `wide-cnn-mnist-bn`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `dynamic-slimmable-mnist`: workflow=dynamic_slimmable; route_summary={'policy': 'dynamic_width', 'mode': 'eval', 'confidence_threshold': 0.85, 'route_counts': {'0.5': 3, '0.75': 6, '1.0': 127}, 'mean_width': 0.9779, 'mean_cost_ratio': 0.9644}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `conditional-computation-mnist`: workflow=conditional_computation; route_summary={'policy': 'early_exit', 'mode': 'eval', 'confidence_threshold': 0.9, 'early_exit_fraction': 0.0, 'full_path_fraction': 1.0, 'mean_width': 1.0, 'mean_cost_ratio': 1.0}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `network-slimming-mnist`: workflow=network_slimming; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

## Constraint Summary

| Experiment | Params | Nonzero params | Weight sparsity | FLOP proxy | Activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-cnn-mnist | 7562 | 7562 | 0.0000 | 2061098 | 4810 |
| wide-cnn-mnist-bn | 16474 | 16474 | 0.0000 | 4505914 | 7210 |
| dynamic-slimmable-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |
| conditional-computation-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |
| network-slimming-mnist | 12187 | 12187 | 0.0000 | 3119397 | 5976 |

## Workflow Stages

### fixed-cnn-mnist
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.46959999203681946
- workflow_metadata={'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1}

### wide-cnn-mnist-bn
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.8501999974250793
- workflow_metadata={'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1}

### dynamic-slimmable-mnist
- dynamic_slimmable_train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5920000076293945
- workflow_metadata={'workflow_name': 'dynamic_slimmable', 'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1, 'routing_policy': 'dynamic_width', 'width_multipliers': [1.0, 0.75, 0.5], 'eval_width_multipliers': [0.5, 0.75, 1.0], 'route_summary': {'policy': 'dynamic_width', 'mode': 'eval', 'confidence_threshold': 0.85, 'route_counts': {'0.5': 3, '0.75': 6, '1.0': 127}, 'mean_width': 0.9779, 'mean_cost_ratio': 0.9644}}

### conditional-computation-mnist
- conditional_computation_train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.6245999932289124
- workflow_metadata={'workflow_name': 'conditional_computation', 'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1, 'routing_policy': 'early_exit', 'confidence_threshold': 0.9, 'route_summary': {'policy': 'early_exit', 'mode': 'eval', 'confidence_threshold': 0.9, 'early_exit_fraction': 0.0, 'full_path_fraction': 1.0, 'mean_width': 1.0, 'mean_cost_ratio': 1.0}}

### network-slimming-mnist
- network_slimming_sparse_train: epochs=5, range=1..5, adaptation_enabled=False, final_val=0.6582000255584717
- network_slimming_finetune: epochs=3, range=6..8, adaptation_enabled=False, final_val=0.8227999806404114
- workflow_metadata={'workflow_name': 'network_slimming', 'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 2, 'prune_fraction': 0.2, 'min_channels_per_block': 12, 'before_conv_channels': [24, 48], 'after_conv_channels': [20, 39]}


## Adaptation Timeline

### network-slimming-mnist
- epoch 5: `prune_channels` params={'prune_fraction': 0.2, 'min_channels_per_block': 12} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'parameter_count_delta': -4287, 'nonzero_parameter_count_delta': -4287, 'weight_sparsity_delta': 0.0, 'forward_flop_proxy_delta': -1386517, 'activation_elements_delta': -1234, 'num_conv_blocks_delta': 0, 'conv_channels_before': [24, 48], 'conv_channels_after': [20, 39], 'channels_changed': True} before={'conv_channels': [24, 48], 'num_conv_blocks': 2, 'classifier_hidden_dims': [96], 'nonzero_parameter_count': 16474, 'masked_weight_count': 0, 'weight_sparsity': 0.0, 'mask_state_names': ['conv_0.weight', 'conv_1.weight', 'linear_0.weight', 'linear_1.weight'], 'device': 'cuda', 'use_batch_norm': True, 'batch_norm_sparsity_strength': 0.0, 'supported_events': ['apply_weight_mask', 'prune_channels'], 'architecture_family': 'cnn', 'parameter_count': 16474, 'forward_flop_proxy': 4505914, 'activation_elements': 7210} after={'conv_channels': [20, 39], 'num_conv_blocks': 2, 'classifier_hidden_dims': [96], 'nonzero_parameter_count': 12187, 'masked_weight_count': 0, 'weight_sparsity': 0.0, 'mask_state_names': ['conv_0.weight', 'conv_1.weight', 'linear_0.weight', 'linear_1.weight'], 'device': 'cuda', 'use_batch_norm': True, 'batch_norm_sparsity_strength': 0.0, 'supported_events': ['apply_weight_mask', 'prune_channels'], 'architecture_family': 'cnn', 'parameter_count': 12187, 'forward_flop_proxy': 3119397, 'activation_elements': 5976, 'conv_channels_before_prune': [24, 48]} capabilities=['apply_weight_mask', 'prune_channels']

## Architecture Graphs

### fixed-cnn-mnist
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

### wide-cnn-mnist-bn
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

### dynamic-slimmable-mnist
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

### conditional-computation-mnist
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

### network-slimming-mnist
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

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8]
    y-axis "Accuracy" 0 --> 1.0
    line "fixed-cnn-mnist" [0.2138, 0.2806, 0.3088, 0.3708, 0.3958, 0.4142, 0.4428, 0.4696]
    line "wide-cnn-mnist-bn" [0.3566, 0.4694, 0.5874, 0.6740, 0.6346, 0.7842, 0.8458, 0.8502]
    line "dynamic-slimmable-mnist" [0.2238, 0.3092, 0.3030, 0.3904, 0.4886, 0.5432, 0.5548, 0.5920]
    line "conditional-computation-mnist" [0.2174, 0.3160, 0.3132, 0.4038, 0.5086, 0.5628, 0.5836, 0.6246]
    line "network-slimming-mnist" [0.3704, 0.4820, 0.5610, 0.7132, 0.6582, 0.7226, 0.7528, 0.8228]
```
