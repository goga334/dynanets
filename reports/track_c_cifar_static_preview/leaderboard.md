# Protocol Leaderboard

Protocol: `track_c_cifar_static_preview`
Track: `cifar10`
Tier: `preview`

## Top Overall

- `condensenet-style-cifar10`: acc=0.4702, flop_proxy=11159424, params=29418

## Top Baseline

- `condensenet-style-cifar10`: acc=0.4702, flop_proxy=11159424, params=29418

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | condensenet-style-cifar10 | baseline | 0.4702 | 11159424 | 29418 |
| 2 | fixed-cnn-cifar10 | baseline | 0.3982 | 10772746 | 53186 |
| 3 | squeezenet-style-cifar10 | baseline | 0.3282 | 4317184 | 46594 |

## Accuracy-FLOP Pareto Frontier

- `squeezenet-style-cifar10`: acc=0.3282, flop_proxy=4317184, params=46594
- `fixed-cnn-cifar10`: acc=0.3982, flop_proxy=10772746, params=53186
- `condensenet-style-cifar10`: acc=0.4702, flop_proxy=11159424, params=29418