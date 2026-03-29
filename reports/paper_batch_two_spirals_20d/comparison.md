# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-spirals20 | baseline | 30 | 0.6589 | 0.6533 | 0.6707 | 0 | - |
| net2wider-spirals20 | dynamic | 30 | 0.6217 | 0.6240 | 0.6507 | 2 | 20 |
| gradmax-spirals20 | dynamic | 30 | 0.6574 | 0.6653 | 0.6760 | 3 | 18 |
| den-spirals20 | dynamic | 30 | 0.6280 | 0.6293 | 0.6813 | 2 | 16 |
| nest-spirals20 | dynamic | 30 | 0.6331 | 0.6240 | 0.6587 | 3 | 12 |
| dynamic-nodes-spirals20 | dynamic | 30 | 0.6080 | 0.6320 | 0.6387 | 1 | 12 |
| edge-growth-spirals20 | dynamic | 30 | 0.6726 | 0.6853 | 0.6920 | 3 | 16 |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `net2wider-spirals20`: adaptation=net2wider
- `gradmax-spirals20`: adaptation=gradmax
- `den-spirals20`: adaptation=den
- `nest-spirals20`: adaptation=nest
- `dynamic-nodes-spirals20`: adaptation=dynamic_nodes
- `edge-growth-spirals20`: adaptation=edge_growth

## Adaptation Timeline

### net2wider-spirals20
- epoch 10: `net2wider` params={'amount': 4, 'seed': 51} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 4, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 20: `net2wider` params={'amount': 4, 'seed': 61} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 4, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 20, 'hidden_dims': [20], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### gradmax-spirals20
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 16: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 24: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 18, 'hidden_dims': [18], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### den-spirals20
- epoch 4: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 21: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### nest-spirals20
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 21: `prune_hidden` params={'amount': 1, 'min_width': 12} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': -1, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 13, 'hidden_dims': [13], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 29: `prune_hidden` params={'amount': 1, 'min_width': 12} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': -1, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 13, 'hidden_dims': [13], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### dynamic-nodes-spirals20
- epoch 1: `insert_hidden_layer` params={'layer_index': 1, 'width': 12} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 1, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12, 12], 'output_dim': 2, 'num_hidden_layers': 2, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### edge-growth-spirals20
- epoch 4: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 12: `insert_hidden_layer` params={'layer_index': 1, 'width': 12} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 1, 'hidden_dims_changed': True} before={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16, 12], 'output_dim': 2, 'num_hidden_layers': 2, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

## Architecture Graphs

### fixed-mlp-spirals20
```mermaid
flowchart LR
    title["fixed-mlp-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (24)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### net2wider-spirals20
```mermaid
flowchart LR
    title["net2wider-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (20)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### gradmax-spirals20
```mermaid
flowchart LR
    title["gradmax-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (18)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### den-spirals20
```mermaid
flowchart LR
    title["den-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (16)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### nest-spirals20
```mermaid
flowchart LR
    title["nest-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (12)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### dynamic-nodes-spirals20
```mermaid
flowchart LR
    title["dynamic-nodes-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (12)"]
    hidden2["Hidden 2 (12)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> hidden2
    hidden2 --> output
```

### edge-growth-spirals20
```mermaid
flowchart LR
    title["edge-growth-spirals20"]
    input["Input (20)"]
    hidden1["Hidden 1 (16)"]
    hidden2["Hidden 2 (12)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> hidden2
    hidden2 --> output
```

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    y-axis "Accuracy" 0 --> 1.0
    line "fixed-mlp-spirals20" [0.5120, 0.4960, 0.5573, 0.5867, 0.5920, 0.6120, 0.6427, 0.6093, 0.6000, 0.5840, 0.6107, 0.6293, 0.6640, 0.6707, 0.6627, 0.6520, 0.6547, 0.6413, 0.6360, 0.6333, 0.6360, 0.6280, 0.6347, 0.6387, 0.6387, 0.6413, 0.6507, 0.6547, 0.6547, 0.6533]
    line "net2wider-spirals20" [0.5347, 0.5387, 0.5187, 0.5333, 0.5507, 0.5373, 0.5293, 0.5427, 0.5400, 0.5520, 0.5867, 0.5733, 0.5813, 0.5973, 0.6053, 0.6080, 0.6107, 0.6067, 0.6413, 0.6400, 0.6507, 0.6360, 0.6507, 0.6453, 0.6387, 0.6493, 0.6453, 0.6293, 0.6267, 0.6240]
    line "gradmax-spirals20" [0.5347, 0.5387, 0.5187, 0.5333, 0.5507, 0.5373, 0.5293, 0.5427, 0.5667, 0.5693, 0.5813, 0.5947, 0.5987, 0.6053, 0.6120, 0.6267, 0.6013, 0.6280, 0.6560, 0.6507, 0.6507, 0.6707, 0.6760, 0.6640, 0.6507, 0.6507, 0.6693, 0.6613, 0.6680, 0.6653]
    line "den-spirals20" [0.5347, 0.5387, 0.5187, 0.5333, 0.5240, 0.5240, 0.5267, 0.5480, 0.5547, 0.5680, 0.5787, 0.5907, 0.5973, 0.5987, 0.6040, 0.6120, 0.6133, 0.6493, 0.6547, 0.6480, 0.6427, 0.6680, 0.6813, 0.6627, 0.6347, 0.6227, 0.6200, 0.6320, 0.6320, 0.6293]
    line "nest-spirals20" [0.5347, 0.5387, 0.5187, 0.5333, 0.5507, 0.5373, 0.5293, 0.5427, 0.5667, 0.5693, 0.5813, 0.5947, 0.5987, 0.6053, 0.6120, 0.6267, 0.6293, 0.6347, 0.6507, 0.6493, 0.6520, 0.6533, 0.6587, 0.6360, 0.6347, 0.6373, 0.6333, 0.6400, 0.6400, 0.6240]
    line "dynamic-nodes-spirals20" [0.5347, 0.5000, 0.5160, 0.5213, 0.5293, 0.5680, 0.5280, 0.5093, 0.5013, 0.5027, 0.5160, 0.5347, 0.5400, 0.5480, 0.5653, 0.5640, 0.5653, 0.5627, 0.5707, 0.5747, 0.5813, 0.5920, 0.6213, 0.6293, 0.6173, 0.6240, 0.6293, 0.6347, 0.6387, 0.6320]
    line "edge-growth-spirals20" [0.5347, 0.5387, 0.5187, 0.5333, 0.5240, 0.5240, 0.5267, 0.5480, 0.5747, 0.5907, 0.5773, 0.5787, 0.4853, 0.5027, 0.5453, 0.6213, 0.6640, 0.6587, 0.6533, 0.6667, 0.6813, 0.6787, 0.6773, 0.6840, 0.6867, 0.6880, 0.6853, 0.6893, 0.6920, 0.6853]
```
