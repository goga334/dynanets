# Protocol Leaderboard

Protocol: `track_a_cnn_wave1_preview`
Track: `synthetic`
Tier: `preview`

## Top Overall

- `wide-cnn-patterns-bn`: acc=0.5565, flop_proxy=4505914, params=16474

## Top Method

- `network-slimming-patterns`: acc=0.5535, flop_proxy=2352616, params=9838

## Top Baseline

- `wide-cnn-patterns-bn`: acc=0.5565, flop_proxy=4505914, params=16474

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | wide-cnn-patterns-bn | baseline | 0.5565 | 4505914 | 16474 |
| 2 | network-slimming-patterns | workflow | 0.5535 | 2352616 | 9838 |
| 3 | fixed-cnn-patterns | baseline | 0.1525 | 2061098 | 7562 |

## Accuracy-FLOP Pareto Frontier

- `fixed-cnn-patterns`: acc=0.1525, flop_proxy=2061098, params=7562
- `network-slimming-patterns`: acc=0.5535, flop_proxy=2352616, params=9838
- `wide-cnn-patterns-bn`: acc=0.5565, flop_proxy=4505914, params=16474