# Protocol Leaderboard

Protocol: `track_c_cifar_routing_focus_round2`
Track: `cifar10`
Tier: `preview`

## Top Overall

- `deep-cnn-cifar10`: acc=0.5511, flop_proxy=10818346, params=76066

## Top Method

- `network-slimming-cifar10`: acc=0.4627, flop_proxy=7434602, params=37111

## Top Baseline

- `deep-cnn-cifar10`: acc=0.5511, flop_proxy=10818346, params=76066

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | deep-cnn-cifar10 | baseline | 0.5511 | 10818346 | 76066 |
| 2 | wide-cnn-cifar10-bn | baseline | 0.5402 | 18498346 | 92298 |
| 3 | network-slimming-cifar10 | workflow | 0.4627 | 7434602 | 37111 |
| 4 | skipnet-cifar10 | workflow | 0.3277 | 23981962 | 43882 |
| 5 | conditional-computation-cifar10 | workflow | 0.3253 | 23981962 | 43882 |
| 6 | iamnn-cifar10 | workflow | 0.3226 | 23981962 | 43882 |
| 7 | dynamic-slimmable-cifar10 | workflow | 0.3215 | 23981962 | 43882 |
| 8 | channel-gating-cifar10 | workflow | 0.3182 | 23981962 | 43882 |
| 9 | instance-wise-sparsity-cifar10 | workflow | 0.3155 | 23981962 | 43882 |

## Accuracy-FLOP Pareto Frontier

- `network-slimming-cifar10`: acc=0.4627, flop_proxy=7434602, params=37111
- `deep-cnn-cifar10`: acc=0.5511, flop_proxy=10818346, params=76066