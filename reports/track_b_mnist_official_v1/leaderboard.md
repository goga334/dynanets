# Protocol Leaderboard

Protocol: `track_b_mnist_official_v1`
Track: `mnist`
Tier: `official`

## Top Overall

- `network-slimming-mnist`: acc=0.8398, flop_proxy=3119397, params=12187

## Top Method

- `network-slimming-mnist`: acc=0.8398, flop_proxy=3119397, params=12187

## Top Baseline

- `wide-cnn-mnist-bn`: acc=0.8340, flop_proxy=4505914, params=16474

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | network-slimming-mnist | workflow | 0.8398 | 3119397 | 12187 |
| 2 | wide-cnn-mnist-bn | baseline | 0.8340 | 4505914 | 16474 |
| 3 | prunetrain-mnist | workflow | 0.8316 | 3190786 | 12466 |
| 4 | layermerge-mnist | workflow | 0.8046 | 4502170 | 14586 |
| 5 | runtime-neural-pruning-mnist | dynamic | 0.7975 | 2489228 | 10156 |
| 6 | asfp-mnist | dynamic | 0.7882 | 1929619 | 8305 |
| 7 | weights-connections-cnn-mnist | dynamic | 0.7816 | 4505914 | 16474 |
| 8 | morphnet-mnist | workflow | 0.7473 | 2617894 | 10678 |

## Accuracy-FLOP Pareto Frontier

- `asfp-mnist`: acc=0.7882, flop_proxy=1929619, params=8305
- `runtime-neural-pruning-mnist`: acc=0.7975, flop_proxy=2489228, params=10156
- `network-slimming-mnist`: acc=0.8398, flop_proxy=3119397, params=12187