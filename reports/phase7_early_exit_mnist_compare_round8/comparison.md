# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wide-cnn-mnist-bn | baseline | 8 | 0.8703 | 0.8502 | 0.8502 | 0 | 0 |
| conditional-computation-mnist | workflow | 8 | 0.5947 | 0.5212 | 0.5212 | 0 | - |
| skipnet-mnist | workflow | 8 | 0.6112 | 0.5082 | 0.5082 | 0 | - |
| iamnn-mnist | workflow | 8 | 0.6142 | 0.5378 | 0.5478 | 0 | - |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `wide-cnn-mnist-bn`: device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `conditional-computation-mnist`: workflow=conditional_computation; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.74, 'target_accept_rate': 0.28, 'early_exit_fraction': 0.2794, 'mean_gate_score': 0.014, 'max_gate_score': 0.0957, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7263}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `skipnet-mnist`: workflow=skipnet; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.16, 'target_cost_ratio': 0.74, 'target_accept_rate': 0.34, 'early_exit_fraction': 0.3382, 'mean_gate_score': 0.015, 'max_gate_score': 0.112, 'full_path_fraction': 0.6618, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.6687}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU
- `iamnn-mnist`: workflow=iamnn; route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.16, 'target_cost_ratio': 0.7, 'target_accept_rate': 0.28, 'early_exit_fraction': 0.2794, 'mean_gate_score': 0.0155, 'max_gate_score': 0.1214, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7263}; device=cuda; requested_device=auto; torch=2.11.0+cu128; cuda_available=True; torch_cuda=12.8; cuda_device=NVIDIA GeForce RTX 4070 Laptop GPU

## Routing Details

### conditional-computation-mnist
- route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.74, 'target_accept_rate': 0.28, 'early_exit_fraction': 0.2794, 'mean_gate_score': 0.014, 'max_gate_score': 0.0957, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7263}
- route_trace=[{'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}]

### skipnet-mnist
- route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.16, 'target_cost_ratio': 0.74, 'target_accept_rate': 0.34, 'early_exit_fraction': 0.3382, 'mean_gate_score': 0.015, 'max_gate_score': 0.112, 'full_path_fraction': 0.6618, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.6687}
- route_trace=[{'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}]

### iamnn-mnist
- route_summary={'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.16, 'target_cost_ratio': 0.7, 'target_accept_rate': 0.28, 'early_exit_fraction': 0.2794, 'mean_gate_score': 0.0155, 'max_gate_score': 0.1214, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7263}
- route_trace=[{'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}]


## Constraint Summary

| Experiment | Params | Nonzero params | Weight sparsity | FLOP proxy | Activation elems |
| --- | ---: | ---: | ---: | ---: | ---: |
| wide-cnn-mnist-bn | 16474 | 16474 | 0.0000 | 4505914 | 7210 |
| conditional-computation-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |
| skipnet-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |
| iamnn-mnist | 11146 | 11146 | 0.0000 | 4439194 | 7114 |

## Workflow Stages

### wide-cnn-mnist-bn
- train: epochs=8, range=1..8, adaptation_enabled=False, final_val=0.8501999974250793
- workflow_metadata={'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 1}

### conditional-computation-mnist
- conditional_computation_warmup: epochs=2, range=1..2, adaptation_enabled=False, final_val=0.2678000032901764
- conditional_computation_routing: epochs=6, range=3..8, adaptation_enabled=False, final_val=0.5212000012397766
- workflow_metadata={'workflow_name': 'conditional_computation', 'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 2, 'routing_policy': 'early_exit', 'gate_mode': 'learned', 'warmup_epochs': 2, 'route_summary': {'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.22, 'target_cost_ratio': 0.74, 'target_accept_rate': 0.28, 'early_exit_fraction': 0.2794, 'mean_gate_score': 0.014, 'max_gate_score': 0.0957, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7263}, 'route_trace': [{'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.22, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}]}

### skipnet-mnist
- skipnet_warmup: epochs=3, range=1..3, adaptation_enabled=False, final_val=0.36039999127388
- skipnet_routing: epochs=5, range=4..8, adaptation_enabled=False, final_val=0.5081999897956848
- workflow_metadata={'workflow_name': 'skipnet', 'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 2, 'routing_policy': 'early_exit', 'gate_mode': 'learned', 'warmup_epochs': 3, 'route_summary': {'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.16, 'target_cost_ratio': 0.74, 'target_accept_rate': 0.34, 'early_exit_fraction': 0.3382, 'mean_gate_score': 0.015, 'max_gate_score': 0.112, 'full_path_fraction': 0.6618, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.6687}, 'route_trace': [{'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.34, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}]}

### iamnn-mnist
- iamnn_warmup: epochs=2, range=1..2, adaptation_enabled=False, final_val=0.274399995803833
- iamnn_routing: epochs=4, range=3..6, adaptation_enabled=False, final_val=0.5090000033378601
- iamnn_consolidation: epochs=2, range=7..8, adaptation_enabled=False, final_val=0.5378000140190125
- workflow_metadata={'workflow_name': 'iamnn', 'configured_total_epochs': 8, 'executed_total_epochs': 8, 'stage_count': 3, 'routing_policy': 'early_exit', 'gate_mode': 'learned', 'warmup_epochs': 2, 'consolidation_epochs': 2, 'route_summary': {'policy': 'early_exit', 'mode': 'eval', 'gate_mode': 'learned', 'gate_metric': 'margin', 'confidence_threshold': 0.16, 'target_cost_ratio': 0.7, 'target_accept_rate': 0.28, 'early_exit_fraction': 0.2794, 'mean_gate_score': 0.0155, 'max_gate_score': 0.1214, 'full_path_fraction': 0.7206, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}], 'mean_width': 1.0, 'mean_cost_ratio': 0.7263}, 'route_trace': [{'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'early'}, {'sample': 6, 'path': 'full'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'full'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}, {'policy': 'early_exit', 'mode': 'eval', 'threshold': 0.16, 'target_accept_rate': 0.28, 'trace_samples': [{'sample': 0, 'path': 'full'}, {'sample': 1, 'path': 'full'}, {'sample': 2, 'path': 'early'}, {'sample': 3, 'path': 'full'}, {'sample': 4, 'path': 'full'}, {'sample': 5, 'path': 'full'}, {'sample': 6, 'path': 'early'}, {'sample': 7, 'path': 'full'}]}]}


## Adaptation Timeline

## Architecture Graphs

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

### skipnet-mnist
```mermaid
flowchart LR
    title["skipnet-mnist"]
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

### iamnn-mnist
```mermaid
flowchart LR
    title["iamnn-mnist"]
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

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8]
    y-axis "Accuracy" 0 --> 1.0
    line "wide-cnn-mnist-bn" [0.3566, 0.4694, 0.5874, 0.6740, 0.6346, 0.7842, 0.8458, 0.8502]
    line "conditional-computation-mnist" [0.2168, 0.2678, 0.3360, 0.3712, 0.4244, 0.4674, 0.5096, 0.5212]
    line "skipnet-mnist" [0.2136, 0.2338, 0.3604, 0.3628, 0.4264, 0.4684, 0.5070, 0.5082]
    line "iamnn-mnist" [0.2158, 0.2744, 0.3564, 0.3908, 0.4618, 0.5090, 0.5478, 0.5378]
```
