# Protocol Leaderboard

Protocol: `track_b_mnist_phase7_preview`
Track: `mnist`
Tier: `preview`

## Top Overall

- `network-slimming-mnist`: acc=0.8029, flop_proxy=3119397, params=12187

## Top Method

- `network-slimming-mnist`: acc=0.8029, flop_proxy=3119397, params=12187

## Top Baseline

- `wide-cnn-mnist-bn`: acc=0.7938, flop_proxy=4505914, params=16474

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | network-slimming-mnist | workflow | 0.8029 | 3119397 | 12187 |
| 2 | wide-cnn-mnist-bn | baseline | 0.7938 | 4505914 | 16474 |
| 3 | conditional-computation-mnist | workflow | 0.5926 | 4439194 | 11146 |
| 4 | dynamic-slimmable-mnist | workflow | 0.5826 | 4439194 | 11146 |
| 5 | fixed-cnn-mnist | baseline | 0.4753 | 2061098 | 7562 |

## Accuracy-FLOP Pareto Frontier

- `fixed-cnn-mnist`: acc=0.4753, flop_proxy=2061098, params=7562
- `network-slimming-mnist`: acc=0.8029, flop_proxy=3119397, params=12187