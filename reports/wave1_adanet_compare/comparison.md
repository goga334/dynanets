# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-spirals10 | baseline | 30 | 0.6194 | 0.6000 | 0.6480 | 0 | - |
| gradmax-spirals10 | dynamic | 30 | 0.6020 | 0.5800 | 0.6013 | 2 | 14 |
| adanet-spirals10 | workflow | 30 | 0.6386 | 0.6333 | 0.6493 | 3 | 18 |
| weights-connections-spirals10 | dynamic | 30 | 0.5906 | 0.6053 | 0.6107 | 6 | 18 |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `fixed-mlp-spirals10`: device=cpu; requested_device=auto; torch=2.10.0+cpu; cuda_available=False
- `gradmax-spirals10`: adaptation=gradmax; device=cpu; requested_device=auto; torch=2.10.0+cpu; cuda_available=False
- `adanet-spirals10`: workflow=adanet_rounds; device=cpu; requested_device=auto; torch=2.10.0+cpu; cuda_available=False
- `weights-connections-spirals10`: adaptation=weights_connections; workflow=scheduled; device=cpu; requested_device=auto; torch=2.10.0+cpu; cuda_available=False

## Workflow Stages

### fixed-mlp-spirals10
- train: epochs=30, range=1..30, adaptation_enabled=False, final_val=0.6000000238418579
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 1}

### gradmax-spirals10
- train: epochs=30, range=1..30, adaptation_enabled=True, final_val=0.5799999833106995
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 1}

### adanet-spirals10
- adanet_warmup: epochs=6, range=1..6, adaptation_enabled=False, final_val=0.518666684627533
- adanet_round_1: epochs=6, range=7..12, adaptation_enabled=False, final_val=0.527999997138977
- adanet_round_2: epochs=6, range=13..18, adaptation_enabled=False, final_val=0.625333309173584
- adanet_round_3: epochs=6, range=19..24, adaptation_enabled=False, final_val=0.5920000076293945
- adanet_consolidate: epochs=6, range=25..30, adaptation_enabled=False, final_val=0.6333333253860474
- workflow_metadata={'workflow_name': 'adanet_rounds', 'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 5, 'round_count': 3, 'total_candidate_evaluations': 5, 'total_candidate_training_epochs': 30, 'rounds': [{'round_index': 1, 'selected_candidate_type': 'wider', 'selected_score': 0.5355151492899114, 'best_metric': 0.5773333311080933, 'parameter_count': 184, 'hidden_dims': [14], 'candidate_count': 2}, {'round_index': 2, 'selected_candidate_type': 'wider', 'selected_score': 0.5868550483040188, 'best_metric': 0.625333309173584, 'parameter_count': 236, 'hidden_dims': [18], 'candidate_count': 2}, {'round_index': 3, 'selected_candidate_type': 'deeper', 'selected_score': 0.5878813416270886, 'best_metric': 0.6399999856948853, 'parameter_count': 410, 'hidden_dims': [18, 10], 'candidate_count': 1}]}

### weights-connections-spirals10
- dense_warmup: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.5920000076293945
- prune: epochs=12, range=9..20, adaptation_enabled=True, final_val=0.5613333582878113
- finetune: epochs=10, range=21..30, adaptation_enabled=False, final_val=0.6053333282470703
- workflow_metadata={'configured_total_epochs': 30, 'executed_total_epochs': 30, 'stage_count': 3}


## Adaptation Timeline

### gradmax-spirals10
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 26, 'nonzero_parameter_count_delta': 26, 'weight_sparsity_delta': 0.0, 'hidden_dims_changed': True} before={'hidden_dim': 10, 'hidden_dims': [10], 'num_hidden_layers': 1, 'parameter_count': 132, 'masked_weight_count': 0, 'nonzero_parameter_count': 132, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} after={'hidden_dim': 12, 'hidden_dims': [12], 'num_hidden_layers': 1, 'parameter_count': 158, 'masked_weight_count': 0, 'nonzero_parameter_count': 158, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 16: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 26, 'nonzero_parameter_count_delta': 26, 'weight_sparsity_delta': 0.0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'num_hidden_layers': 1, 'parameter_count': 158, 'masked_weight_count': 0, 'nonzero_parameter_count': 158, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} after={'hidden_dim': 14, 'hidden_dims': [14], 'num_hidden_layers': 1, 'parameter_count': 184, 'masked_weight_count': 0, 'nonzero_parameter_count': 184, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']

### adanet-spirals10
- epoch 12: `net2wider` params={'amount': 4, 'seed': 42} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 6, 'hidden_dim_delta': 4, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 52, 'nonzero_parameter_count_delta': 52, 'weight_sparsity_delta': 0.0, 'hidden_dims_changed': True} before={'hidden_dim': 10, 'hidden_dims': [10], 'num_hidden_layers': 1, 'parameter_count': 132, 'masked_weight_count': 0, 'nonzero_parameter_count': 132, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} after={'hidden_dim': 14, 'hidden_dims': [14], 'num_hidden_layers': 1, 'parameter_count': 184, 'masked_weight_count': 0, 'nonzero_parameter_count': 184, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 18: `net2wider` params={'amount': 4, 'seed': 42} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 6, 'hidden_dim_delta': 4, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 52, 'nonzero_parameter_count_delta': 52, 'weight_sparsity_delta': 0.0, 'hidden_dims_changed': True} before={'hidden_dim': 14, 'hidden_dims': [14], 'num_hidden_layers': 1, 'parameter_count': 184, 'masked_weight_count': 0, 'nonzero_parameter_count': 184, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 0, 'nonzero_parameter_count': 236, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 24: `insert_hidden_layer` params={'layer_index': 1, 'width': 10, 'init_mode': 'identity'} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 6, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 1, 'parameter_count_delta': 174, 'nonzero_parameter_count_delta': 174, 'weight_sparsity_delta': 0.0, 'hidden_dims_changed': True} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 0, 'nonzero_parameter_count': 236, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} after={'hidden_dim': 18, 'hidden_dims': [18, 10], 'num_hidden_layers': 2, 'parameter_count': 410, 'masked_weight_count': 0, 'nonzero_parameter_count': 410, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']

### weights-connections-spirals10
- epoch 10: `apply_weight_mask` params={'threshold': 0.027810515835881233, 'target_sparsity': 0.08} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.0787037037037037, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 0, 'nonzero_parameter_count': 236, 'weight_sparsity': 0.0, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 17, 'nonzero_parameter_count': 219, 'weight_sparsity': 0.0787037037037037, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.027810515835881233, 'target_weight_sparsity': 0.08} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 12: `apply_weight_mask` params={'threshold': 0.049374934285879135, 'target_sparsity': 0.1587037037037037} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.0787037037037037, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 17, 'nonzero_parameter_count': 219, 'weight_sparsity': 0.0787037037037037, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.027810515835881233, 'target_weight_sparsity': 0.08} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 34, 'nonzero_parameter_count': 202, 'weight_sparsity': 0.1574074074074074, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.049374934285879135, 'target_weight_sparsity': 0.1587037037037037} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 14: `apply_weight_mask` params={'threshold': 0.06899179518222809, 'target_sparsity': 0.2374074074074074} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370369, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 34, 'nonzero_parameter_count': 202, 'weight_sparsity': 0.1574074074074074, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.049374934285879135, 'target_weight_sparsity': 0.1587037037037037} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 51, 'nonzero_parameter_count': 185, 'weight_sparsity': 0.2361111111111111, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.06899179518222809, 'target_weight_sparsity': 0.2374074074074074} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 16: `apply_weight_mask` params={'threshold': 0.09855222702026367, 'target_sparsity': 0.3161111111111111} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370372, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 51, 'nonzero_parameter_count': 185, 'weight_sparsity': 0.2361111111111111, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.06899179518222809, 'target_weight_sparsity': 0.2374074074074074} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 68, 'nonzero_parameter_count': 168, 'weight_sparsity': 0.3148148148148148, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.09855222702026367, 'target_weight_sparsity': 0.3161111111111111} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 18: `apply_weight_mask` params={'threshold': 0.1196371465921402, 'target_sparsity': 0.39481481481481484} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370372, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 68, 'nonzero_parameter_count': 168, 'weight_sparsity': 0.3148148148148148, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.09855222702026367, 'target_weight_sparsity': 0.3161111111111111} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 85, 'nonzero_parameter_count': 151, 'weight_sparsity': 0.39351851851851855, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.1196371465921402, 'target_weight_sparsity': 0.39481481481481484} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']
- epoch 20: `apply_weight_mask` params={'threshold': 0.14212003350257874, 'target_sparsity': 0.47351851851851856} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 0, 'parameter_count_delta': 0, 'nonzero_parameter_count_delta': -17, 'weight_sparsity_delta': 0.07870370370370366, 'hidden_dims_changed': False} before={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 85, 'nonzero_parameter_count': 151, 'weight_sparsity': 0.39351851851851855, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.1196371465921402, 'target_weight_sparsity': 0.39481481481481484} after={'hidden_dim': 18, 'hidden_dims': [18], 'num_hidden_layers': 1, 'parameter_count': 236, 'masked_weight_count': 102, 'nonzero_parameter_count': 134, 'weight_sparsity': 0.4722222222222222, 'device': 'cpu', 'supported_events': ['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer'], 'mask_threshold': 0.14212003350257874, 'target_weight_sparsity': 0.47351851851851856} capabilities=['apply_weight_mask', 'grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden', 'remove_hidden_layer']

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

### adanet-spirals10
```mermaid
flowchart LR
    title["adanet-spirals10"]
    input["Input (10)"]
    hidden_1["Hidden 1 (18)"]
    hidden_2["Hidden 2 (10)"]
    output["Output (2)"]
    input --> hidden_1
    hidden_1 --> hidden_2
    hidden_2 --> output
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
    line "adanet-spirals10" [0.5000, 0.5000, 0.5013, 0.4947, 0.5040, 0.5187, 0.5187, 0.5213, 0.5773, 0.5533, 0.5293, 0.5280, 0.5440, 0.5680, 0.5880, 0.6053, 0.6253, 0.6253, 0.6400, 0.5720, 0.5453, 0.5640, 0.5907, 0.5920, 0.6227, 0.6413, 0.6480, 0.6493, 0.6440, 0.6333]
    line "weights-connections-spirals10" [0.5333, 0.5347, 0.5520, 0.5667, 0.5640, 0.5693, 0.5973, 0.5920, 0.5880, 0.5840, 0.5867, 0.5813, 0.5880, 0.5987, 0.6107, 0.5947, 0.5640, 0.5707, 0.5627, 0.5613, 0.5560, 0.5573, 0.5680, 0.5760, 0.5827, 0.5827, 0.5920, 0.6000, 0.6040, 0.6053]
```
