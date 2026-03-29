# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-circles | baseline | 30 | 0.9912 | 0.9961 | 0.9961 | 0 | - |
| net2wider-circles | dynamic | 30 | 0.8584 | 0.8906 | 0.8906 | 2 | 16 |
| gradmax-circles | dynamic | 30 | 0.8262 | 0.8516 | 0.8516 | 3 | 14 |
| den-circles | dynamic | 30 | 0.8105 | 0.8203 | 0.8203 | 0 | 8 |
| nest-circles | dynamic | 30 | 0.8857 | 0.9023 | 0.9023 | 3 | 8 |
| dynamic-nodes-circles | dynamic | 30 | 0.9414 | 0.9297 | 0.9297 | 1 | 8 |
| edge-growth-circles | dynamic | 30 | 0.6426 | 0.6406 | 0.6953 | 3 | 12 |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `net2wider-circles`: adaptation=net2wider
- `gradmax-circles`: adaptation=gradmax
- `den-circles`: adaptation=den
- `nest-circles`: adaptation=nest
- `dynamic-nodes-circles`: adaptation=dynamic_nodes
- `edge-growth-circles`: adaptation=edge_growth

## Adaptation Timeline

### net2wider-circles
- epoch 10: `net2wider` params={'amount': 4, 'seed': 51} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 4, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 20: `net2wider` params={'amount': 4, 'seed': 61} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 4, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### gradmax-circles
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 16: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 24: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 14, 'hidden_dims': [14], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### nest-circles
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 21: `prune_hidden` params={'amount': 1, 'min_width': 8} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': -1, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 9, 'hidden_dims': [9], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 29: `prune_hidden` params={'amount': 1, 'min_width': 8} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': -1, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 9, 'hidden_dims': [9], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### dynamic-nodes-circles
- epoch 1: `insert_hidden_layer` params={'layer_index': 1, 'width': 8} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 1, 'hidden_dims_changed': True} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 8, 'hidden_dims': [8, 8], 'output_dim': 2, 'num_hidden_layers': 2, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### edge-growth-circles
- epoch 4: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 8: `net2wider` params={'amount': 2} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 2, 'num_hidden_layers_delta': 0, 'hidden_dims_changed': True} before={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 12: `insert_hidden_layer` params={'layer_index': 1, 'width': 8} effect={'applied': True, 'structural_change': True, 'version_delta': 1, 'step_delta': 0, 'hidden_dim_delta': 0, 'num_hidden_layers_delta': 1, 'hidden_dims_changed': True} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12, 8], 'output_dim': 2, 'num_hidden_layers': 2, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

## Architecture Graphs

### fixed-mlp-circles
```mermaid
flowchart LR
    title["fixed-mlp-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (16)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### net2wider-circles
```mermaid
flowchart LR
    title["net2wider-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (16)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### gradmax-circles
```mermaid
flowchart LR
    title["gradmax-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (14)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### den-circles
```mermaid
flowchart LR
    title["den-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (8)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### nest-circles
```mermaid
flowchart LR
    title["nest-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (8)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> output
```

### dynamic-nodes-circles
```mermaid
flowchart LR
    title["dynamic-nodes-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (8)"]
    hidden2["Hidden 2 (8)"]
    output["Output (2)"]
    input --> hidden1
    hidden1 --> hidden2
    hidden2 --> output
```

### edge-growth-circles
```mermaid
flowchart LR
    title["edge-growth-circles"]
    input["Input (2)"]
    hidden1["Hidden 1 (12)"]
    hidden2["Hidden 2 (8)"]
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
    y-axis "Accuracy" 0 --> 1.05
    line "fixed-mlp-circles" [0.5625, 0.6367, 0.5977, 0.5586, 0.5703, 0.5703, 0.5430, 0.5352, 0.5312, 0.5312, 0.5391, 0.5586, 0.6133, 0.6719, 0.7695, 0.8125, 0.8672, 0.8867, 0.9219, 0.9531, 0.9688, 0.9727, 0.9766, 0.9805, 0.9883, 0.9883, 0.9922, 0.9961, 0.9961, 0.9961]
    line "net2wider-circles" [0.5469, 0.5430, 0.5664, 0.5703, 0.5781, 0.5859, 0.5938, 0.5938, 0.6055, 0.6172, 0.6289, 0.6367, 0.6602, 0.6680, 0.6836, 0.7070, 0.7227, 0.7500, 0.7500, 0.7617, 0.7578, 0.7734, 0.7930, 0.8047, 0.8125, 0.8359, 0.8477, 0.8672, 0.8789, 0.8906]
    line "gradmax-circles" [0.5469, 0.5430, 0.5664, 0.5703, 0.5781, 0.5859, 0.5938, 0.5938, 0.6055, 0.6172, 0.6328, 0.6602, 0.6797, 0.7109, 0.7188, 0.7383, 0.7148, 0.7305, 0.7539, 0.7656, 0.7812, 0.7852, 0.7930, 0.7969, 0.8281, 0.8203, 0.8086, 0.8203, 0.8359, 0.8516]
    line "den-circles" [0.5469, 0.5430, 0.5664, 0.5703, 0.5781, 0.5859, 0.5938, 0.5938, 0.6055, 0.6172, 0.6289, 0.6406, 0.6641, 0.6719, 0.6758, 0.6875, 0.6992, 0.7109, 0.7148, 0.7227, 0.7227, 0.7227, 0.7422, 0.7422, 0.7500, 0.7734, 0.7773, 0.7930, 0.8164, 0.8203]
    line "nest-circles" [0.5469, 0.5430, 0.5664, 0.5703, 0.5781, 0.5859, 0.5938, 0.5938, 0.6055, 0.6172, 0.6328, 0.6602, 0.6797, 0.7109, 0.7188, 0.7383, 0.7500, 0.7500, 0.7617, 0.7695, 0.7812, 0.7852, 0.7930, 0.7930, 0.8008, 0.8086, 0.8203, 0.8359, 0.8477, 0.9023]
    line "dynamic-nodes-circles" [0.5469, 0.6406, 0.7109, 0.7148, 0.7188, 0.7227, 0.7227, 0.7109, 0.7031, 0.7031, 0.7031, 0.7031, 0.7031, 0.7109, 0.7266, 0.7344, 0.7773, 0.8047, 0.8281, 0.8594, 0.8828, 0.9062, 0.9141, 0.9141, 0.9180, 0.9180, 0.9180, 0.9180, 0.9219, 0.9297]
    line "edge-growth-circles" [0.5469, 0.5430, 0.5664, 0.5703, 0.5781, 0.5898, 0.5898, 0.5938, 0.6133, 0.6406, 0.6523, 0.6953, 0.6211, 0.6328, 0.5703, 0.5625, 0.5547, 0.5547, 0.5625, 0.5625, 0.5703, 0.5586, 0.5664, 0.5742, 0.5938, 0.6055, 0.6055, 0.6172, 0.6289, 0.6406]
```
