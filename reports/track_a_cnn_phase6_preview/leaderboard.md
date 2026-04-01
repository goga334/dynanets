# Protocol Leaderboard

Protocol: `track_a_cnn_phase6_preview`
Track: `synthetic`
Tier: `preview`

## Top Overall

- `wide-cnn-patterns-bn`: acc=0.5158, flop_proxy=4505914, params=16474

## Top Method

- `network-slimming-patterns`: acc=0.5092, flop_proxy=2352616, params=9838

## Top Baseline

- `wide-cnn-patterns-bn`: acc=0.5158, flop_proxy=4505914, params=16474

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | wide-cnn-patterns-bn | baseline | 0.5158 | 4505914 | 16474 |
| 2 | network-slimming-patterns | workflow | 0.5092 | 2352616 | 9838 |
| 3 | channel-pruning-patterns | dynamic | 0.4833 | 1194741 | 5971 |
| 4 | fixed-cnn-patterns | baseline | 0.1217 | 2061098 | 7562 |

## Accuracy-FLOP Pareto Frontier

- `channel-pruning-patterns`: acc=0.4833, flop_proxy=1194741, params=5971
- `network-slimming-patterns`: acc=0.5092, flop_proxy=2352616, params=9838
- `wide-cnn-patterns-bn`: acc=0.5158, flop_proxy=4505914, params=16474