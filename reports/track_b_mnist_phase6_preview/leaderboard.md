# Protocol Leaderboard

Protocol: `track_b_mnist_phase6_preview`
Track: `mnist`
Tier: `preview`

## Top Overall

- `channel-pruning-mnist`: acc=0.1150, flop_proxy=2617894, params=10678

## Top Method

- `channel-pruning-mnist`: acc=0.1150, flop_proxy=2617894, params=10678

## Top Baseline

- `wide-cnn-mnist-bn`: acc=0.1106, flop_proxy=4505914, params=16474

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | channel-pruning-mnist | dynamic | 0.1150 | 2617894 | 10678 |
| 2 | weights-connections-cnn-mnist | dynamic | 0.1106 | 4505914 | 16474 |
| 3 | network-slimming-mnist | workflow | 0.1106 | 2617894 | 10678 |
| 4 | wide-cnn-mnist-bn | baseline | 0.1106 | 4505914 | 16474 |

## Accuracy-FLOP Pareto Frontier

- `channel-pruning-mnist`: acc=0.1150, flop_proxy=2617894, params=10678