# Protocol Leaderboard

Protocol: `track_c_cifar_official_extended`
Track: `cifar10`
Tier: `official_extended`

## Top Overall

- `wide-cnn-cifar10-bn`: acc=0.6789, flop_proxy=18498346, params=92298

## Top Method

- `weights-connections-cnn-cifar10`: acc=0.6540, flop_proxy=10772746, params=53186

## Top Baseline

- `wide-cnn-cifar10-bn`: acc=0.6789, flop_proxy=18498346, params=92298

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | wide-cnn-cifar10-bn | baseline | 0.6789 | 18498346 | 92298 |
| 2 | weights-connections-cnn-cifar10 | dynamic | 0.6540 | 10772746 | 53186 |
| 3 | runtime-neural-pruning-cifar10 | dynamic | 0.6438 | 6338410 | 32468 |
| 4 | deep-cnn-cifar10 | baseline | 0.6408 | 10818346 | 76066 |
| 5 | channel-pruning-cifar10 | dynamic | 0.6390 | 4753562 | 24353 |
| 6 | network-slimming-cifar10 | workflow | 0.6253 | 7687498 | 38798 |
| 7 | morphnet-cifar10 | workflow | 0.6244 | 6996826 | 35561 |
| 8 | layermerge-cifar10 | workflow | 0.6181 | 10767466 | 50530 |
| 9 | prunetrain-cifar10 | workflow | 0.6027 | 7672938 | 37492 |
| 10 | asfp-cifar10 | dynamic | 0.6007 | 3122058 | 15792 |
| 11 | dynamic-slimmable-cifar10 | workflow | 0.5266 | 52765322 | 221178 |
| 12 | channel-gating-cifar10 | workflow | 0.4944 | 28843530 | 123578 |
| 13 | skipnet-cifar10 | workflow | 0.4806 | 28843530 | 123578 |
| 14 | conditional-computation-cifar10 | workflow | 0.4538 | 39919690 | 168922 |
| 15 | iamnn-cifar10 | workflow | 0.4457 | 33348618 | 138962 |
| 16 | instance-wise-sparsity-cifar10 | workflow | 0.4450 | 18355594 | 75754 |

## Accuracy-FLOP Pareto Frontier

- `asfp-cifar10`: acc=0.6007, flop_proxy=3122058, params=15792
- `channel-pruning-cifar10`: acc=0.6390, flop_proxy=4753562, params=24353
- `runtime-neural-pruning-cifar10`: acc=0.6438, flop_proxy=6338410, params=32468
- `weights-connections-cnn-cifar10`: acc=0.6540, flop_proxy=10772746, params=53186
- `wide-cnn-cifar10-bn`: acc=0.6789, flop_proxy=18498346, params=92298