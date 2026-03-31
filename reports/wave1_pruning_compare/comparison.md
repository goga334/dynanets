# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-spirals10 | baseline | 30 | 0.6194 | 0.6000 | 0.6480 | 0 | - |
| gradmax-spirals10 | dynamic | 30 | 0.6020 | 0.5800 | 0.6013 | 2 | 14 |
| weights-connections-spirals10 | dynamic | 30 | 0.5906 | 0.6053 | 0.6107 | 6 | 18 |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `fixed-mlp-spirals10`: device=cpu; requested_device=auto; torch=2.11.0+cpu; cuda_available=False
- `gradmax-spirals10`: adaptation=gradmax; device=cpu; requested_device=auto; torch=2.11.0+cpu; cuda_available=False
- `weights-connections-spirals10`: adaptation=weights_connections; workflow=scheduled; device=cpu; requested_device=auto; torch=2.11.0+cpu; cuda_available=False

## Constraint Summary

| Experiment | Params | Nonzero params | Weight sparsity | FLOP proxy | Activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-spirals10 | 236 | 236 | 0.0000 | 452 | 20 |
| gradmax-spirals10 | 184 | 184 | 0.0000 | 352 | 16 |
| weights-connections-spirals10 | 236 | 134 | 0.4722 | 452 | 20 |

## Workflow Stages

### fixed-mlp-spirals10
- train: epochs=30, range=1..30, adaptation_enabled=False, final_val=0.6000000238418579
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 1}

### gradmax-spirals10
- train: epochs=30, range=1..30, adaptation_enabled=True, final_val=0.5799999833106995
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 1}

### weights-connections-spirals10
- dense_warmup: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5920000076293945
- prune: epochs=12, range=9..20, adaptation_enabled=True, final_val=0.5613333582878113
- finetune: epochs=10, range=21..30, adaptation_enabled=False, final_val=0.6053333282470703
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 3}


## Adaptation Timeline

### gradmax-spirals10
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 26, 'nonzero_parameter_count_delta': 26, 'weight_sparsity_delta': 0.0, 'forward_flop_proxy_delta': 50, 'activation_elements_delta': 2, 'hidden_dims_changed': True} before={'hidden_dim': 10, 'hidden_dims': [10], 'num_hidden_layers': 1, 'masked_weight_count': 0, 'nonzero_parameter_count': 132, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 132, 'forward_flop_proxy': 252, 'activation_elements': 12} after={'hidden_dim': 12, 'hidden_dims': [12], 'num_hidden_layers': 1, 'masked_weight_count': 0, 'nonzero_parameter_count': 158, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 158, 'forward_flop_proxy': 302, 'activation_elements': 14} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']
- epoch 16: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 26, 'nonzero_parameter_count_delta': 26, 'weight_sparsity_delta': 0.0, 'forward_flop_proxy_delta': 50, 'activation_elements_delta': 2, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'num_hidden_layers': 1, 'masked_weight_count': 0, 'nonzero_parameter_count': 158, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 158, 'forward_flop_proxy': 302, 'activation_elements': 14} after={'hidden_dim': 14, 'hidden_dims': [14], 'num_hidden_layers': 1, 'masked_weight_count': 0, 'nonzero_parameter_count': 184, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 184, 'forward_flop_proxy': 352, 'activation_elements': 16} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']

### weights-connections-spirals10
- epoch 10: `apply_weight_mask` params={'threshold': 0.027810515835881233, 'target_sparsity': 0.08} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.0787037037037037, 'forward_flop_proxy_delta': 0, 'activation_elements_delta': 0, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 0, 'nonzero_parameter_count': 236, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 17, 'nonzero_parameter_count': 219, 'weight_sparsity': 0.0787037037037037, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.027810515835881233, 'target_weight_sparsity': 0.08} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']
- epoch 12: `apply_weight_mask` params={'threshold': 0.049374934285879135, 'target_sparsity': 0.1587037037037037} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.0787037037037037, 'forward_flop_proxy_delta': 0, 'activation_elements_delta': 0, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 17, 'nonzero_parameter_count': 219, 'weight_sparsity': 0.0787037037037037, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.027810515835881233, 'target_weight_sparsity': 0.08} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 34, 'nonzero_parameter_count': 202, 'weight_sparsity': 0.1574074074074074, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.049374934285879135, 'target_weight_sparsity': 0.1587037037037037} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']
- epoch 14: `apply_weight_mask` params={'threshold': 0.06899179518222809, 'target_sparsity': 0.2374074074074074} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370369, 'forward_flop_proxy_delta': 0, 'activation_elements_delta': 0, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 34, 'nonzero_parameter_count': 202, 'weight_sparsity': 0.1574074074074074, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.049374934285879135, 'target_weight_sparsity': 0.1587037037037037} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 51, 'nonzero_parameter_count': 185, 'weight_sparsity': 0.2361111111111111, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.06899179518222809, 'target_weight_sparsity': 0.2374074074074074} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']
- epoch 16: `apply_weight_mask` params={'threshold': 0.09855222702026367, 'target_sparsity': 0.3161111111111111} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370372, 'forward_flop_proxy_delta': 0, 'activation_elements_delta': 0, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 51, 'nonzero_parameter_count': 185, 'weight_sparsity': 0.2361111111111111, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.06899179518222809, 'target_weight_sparsity': 0.2374074074074074} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 68, 'nonzero_parameter_count': 168, 'weight_sparsity': 0.3148148148148148, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.09855222702026367, 'target_weight_sparsity': 0.3161111111111111} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']
- epoch 18: `apply_weight_mask` params={'threshold': 0.1196371465921402, 'target_sparsity': 0.39481481481481484} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370372, 'forward_flop_proxy_delta': 0, 'activation_elements_delta': 0, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 68, 'nonzero_parameter_count': 168, 'weight_sparsity': 0.3148148148148148, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.09855222702026367, 'target_weight_sparsity': 0.3161111111111111} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 85, 'nonzero_parameter_count': 151, 'weight_sparsity': 0.39351851851851855, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.1196371465921402, 'target_weight_sparsity': 0.39481481481481484} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']
- epoch 20: `apply_weight_mask` params={'threshold': 0.14212003350257874, 'target_sparsity': 0.47351851851851856} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370366, 'forward_flop_proxy_delta': 0, 'activation_elements_delta': 0, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 85, 'nonzero_parameter_count': 151, 'weight_sparsity': 0.39351851851851855, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.1196371465921402, 'target_weight_sparsity': 0.39481481481481484} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'masked_weight_count': 102, 'nonzero_parameter_count': 134, 'weight_sparsity': 0.4722222222222222, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer'], 'mask_state_names': ['linear_0.weight', 'linear_1.weight'], 'architecture_family': 'mlp', 'parameter_count': 236, 'forward_flop_proxy': 452, 'activation_elements': 20, 'mask_threshold': 0.14212003350257874, 'target_weight_sparsity': 0.47351851851851856} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'prune_neurons', 'remove_hidden_layer']

## Architecture Graphs

### fixed-mlp-spirals10
```mermaid
flowchart LR
    title["fixed-mlp-spirals10"]
    input["Input (10)"]
    hidden_1["Hidden 1 (18)"]
    output["Output (2)"]
    input --> hidden_1
    hidden_1 --> output
```

### gradmax-spirals10
```mermaid
flowchart LR
    title["gradmax-spirals10"]
    input["Input (10)"]
    hidden_1["Hidden 1 (14)"]
    output["Output (2)"]
    input --> hidden_1
    hidden_1 --> output
```

### weights-connections-spirals10
```mermaid
flowchart LR
    title["weights-connections-spirals10"]
    input["Input (10)"]
    hidden_1["Hidden 1 (18)"]
    output["Output (2)"]
    input --> hidden_1
    hidden_1 --> output
```

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    y-axis "Accuracy" 0 --> 1.0
    line "fixed-mlp-spirals10" [0.5733, 0.5560, 0.5613, 0.5240, 0.5413, 0.5520, 0.5800, 0.5840, 0.6027, 0.6333, 0.6480, 0.6480, 0.6347, 0.6293, 0.6173, 0.6053, 0.5960, 0.5933, 0.6040, 0.6147, 0.6147, 0.6133, 0.6133, 0.6120, 0.6053, 0.6107, 0.6067, 0.6013, 0.5973, 0.6000]
    line "gradmax-spirals10" [0.5000, 0.5000, 0.5013, 0.4947, 0.5040, 0.5187, 0.5520, 0.5773, 0.5707, 0.5827, 0.5680, 0.5547, 0.5533, 0.5493, 0.5480, 0.5640, 0.5880, 0.5960, 0.5933, 0.5933, 0.6013, 0.5947, 0.5853, 0.5813, 0.5920, 0.5907, 0.5960, 0.5987, 0.5960, 0.5800]
    line "weights-connections-spirals10" [0.5333, 0.5347, 0.5520, 0.5667, 0.5640, 0.5693, 0.5973, 0.5920, 0.5880, 0.5840, 0.5867, 0.5813, 0.5880, 0.5987, 0.6107, 0.5947, 0.5640, 0.5707, 0.5627, 0.5613, 0.5560, 0.5573, 0.5680, 0.5760, 0.5827, 0.5827, 0.5920, 0.6000, 0.6040, 0.6053]
```
