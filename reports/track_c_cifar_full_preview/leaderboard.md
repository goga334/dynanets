# Protocol Leaderboard

Protocol: `track_c_cifar_full_preview`
Track: `cifar10`
Tier: `preview`

## Top Overall

- `wide-cnn-cifar10-bn`: acc=0.5706, flop_proxy=18498346, params=92298

## Top Method

- `asfp-cifar10`: acc=0.5330, flop_proxy=4525546, params=22625

## Top Baseline

- `wide-cnn-cifar10-bn`: acc=0.5706, flop_proxy=18498346, params=92298

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | wide-cnn-cifar10-bn | baseline | 0.5706 | 18498346 | 92298 |
| 2 | deep-cnn-cifar10 | baseline | 0.5406 | 10818346 | 76066 |
| 3 | asfp-cifar10 | dynamic | 0.5330 | 4525546 | 22625 |
| 4 | channel-pruning-cifar10 | dynamic | 0.5211 | 7640954 | 38307 |
| 5 | prunetrain-cifar10 | workflow | 0.5211 | 7640954 | 38307 |
| 6 | morphnet-cifar10 | workflow | 0.5154 | 6338410 | 32468 |
| 7 | runtime-neural-pruning-cifar10 | dynamic | 0.5081 | 5927546 | 29855 |
| 8 | layermerge-cifar10 | workflow | 0.4788 | 10767466 | 50530 |
| 9 | condensenet-style-cifar10 | baseline | 0.4702 | 11159424 | 29418 |
| 10 | network-slimming-cifar10 | workflow | 0.4620 | 7434602 | 37111 |
| 11 | layerwise-obs-cnn-cifar10 | dynamic | 0.4575 | 10772746 | 53186 |
| 12 | weights-connections-cnn-cifar10 | dynamic | 0.4383 | 10772746 | 53186 |
| 13 | fixed-cnn-cifar10 | baseline | 0.3982 | 10772746 | 53186 |
| 14 | squeezenet-style-cifar10 | baseline | 0.3282 | 4317184 | 46594 |
| 15 | instance-wise-sparsity-cifar10 | workflow | 0.2994 | 11269386 | 20042 |
| 16 | channel-gating-cifar10 | workflow | 0.2992 | 11269386 | 20042 |
| 17 | dynamic-slimmable-cifar10 | workflow | 0.2885 | 11269386 | 20042 |
| 18 | iamnn-cifar10 | workflow | 0.2822 | 11269386 | 20042 |
| 19 | skipnet-cifar10 | workflow | 0.2812 | 11269386 | 20042 |
| 20 | conditional-computation-cifar10 | workflow | 0.2760 | 11269386 | 20042 |

## Accuracy-FLOP Pareto Frontier

- `squeezenet-style-cifar10`: acc=0.3282, flop_proxy=4317184, params=46594
- `asfp-cifar10`: acc=0.5330, flop_proxy=4525546, params=22625
- `deep-cnn-cifar10`: acc=0.5406, flop_proxy=10818346, params=76066
- `wide-cnn-cifar10-bn`: acc=0.5706, flop_proxy=18498346, params=92298