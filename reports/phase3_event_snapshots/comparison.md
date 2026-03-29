# Baseline Comparison

| Experiment | Type | Epochs | Final train acc | Final val acc | Best val acc | Adaptations | Final hidden dim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-mlp-baseline | baseline | 6 | 0.9805 | 0.9844 | 0.9844 | 0 | - |
| net2wider-dynamic-baseline | dynamic | 6 | 0.9824 | 0.9844 | 0.9844 | 2 | 16 |
| net2deeper-dynamic-baseline | dynamic | 6 | 0.9824 | 0.9922 | 0.9922 | 2 | 8 |
| width-pruning-dynamic-baseline | dynamic | 6 | 0.9629 | 0.9375 | 0.9375 | 3 | 6 |

## Validation Accuracy

![Validation accuracy](validation_accuracy.png)

## Training Accuracy

![Training accuracy](training_accuracy.png)

## Training Loss

![Training loss](training_loss.png)

## Experiment Notes

- `net2wider-dynamic-baseline`: adaptation=net2wider
- `net2deeper-dynamic-baseline`: adaptation=net2deeper
- `width-pruning-dynamic-baseline`: adaptation=prune_hidden

## Adaptation Timeline

### net2wider-dynamic-baseline
- epoch 2: `net2wider` params={'amount': 4, 'seed': 43} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 4: `net2wider` params={'amount': 4, 'seed': 45} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 16, 'hidden_dims': [16], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### net2deeper-dynamic-baseline
- epoch 2: `insert_hidden_layer` params={'layer_index': 1, 'width': 8, 'init_mode': 'identity'} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 8, 'hidden_dims': [8, 8], 'output_dim': 2, 'num_hidden_layers': 2, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 4: `insert_hidden_layer` params={'layer_index': 1, 'width': 8, 'init_mode': 'identity'} before={'hidden_dim': 8, 'hidden_dims': [8, 8], 'output_dim': 2, 'num_hidden_layers': 2, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 8, 'hidden_dims': [8, 8, 8], 'output_dim': 2, 'num_hidden_layers': 3, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

### width-pruning-dynamic-baseline
- epoch 2: `prune_hidden` params={'amount': 2, 'min_width': 6} before={'hidden_dim': 12, 'hidden_dims': [12], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 4: `prune_hidden` params={'amount': 2, 'min_width': 6} before={'hidden_dim': 10, 'hidden_dims': [10], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']
- epoch 6: `prune_hidden` params={'amount': 2, 'min_width': 6} before={'hidden_dim': 8, 'hidden_dims': [8], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} after={'hidden_dim': 6, 'hidden_dims': [6], 'output_dim': 2, 'num_hidden_layers': 1, 'supported_events': ['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']} capabilities=['grow_hidden', 'insert_hidden_layer', 'net2wider', 'prune_hidden']

## Validation Accuracy By Epoch

```mermaid
xychart-beta
    title "Validation Accuracy Comparison"
    x-axis "Epoch" [1, 2, 3, 4, 5, 6]
    y-axis "Accuracy" 0 --> 1.04
    line "fixed-mlp-baseline" [0.9062, 0.9141, 0.9609, 0.9688, 0.9688, 0.9844]
    line "net2wider-dynamic-baseline" [0.9062, 0.9141, 0.9688, 0.9688, 0.9766, 0.9844]
    line "net2deeper-dynamic-baseline" [0.9062, 0.9141, 0.9453, 0.9688, 0.9844, 0.9922]
    line "width-pruning-dynamic-baseline" [0.5391, 0.6562, 0.8438, 0.8984, 0.9141, 0.9375]
```
