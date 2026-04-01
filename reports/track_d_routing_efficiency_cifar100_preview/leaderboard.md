# Protocol Leaderboard

Protocol: `track_d_routing_efficiency_cifar100_preview`
Track: `cifar100`
Tier: `preview`

## Top Overall

- `routed-resnet-largest-cifar100`: acc=0.3654, flop_proxy=38171748, params=166532

## Top Method

- `dynamic-slimmable-cifar100`: acc=0.3289, flop_proxy=38171748, params=166532

## Top Baseline

- `routed-resnet-largest-cifar100`: acc=0.3654, flop_proxy=38171748, params=166532

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | routed-resnet-largest-cifar100 | baseline | 0.3654 | 38171748 | 166532 |
| 2 | wide-cnn-cifar100-bn | baseline | 0.3359 | 18527236 | 106788 |
| 3 | dynamic-slimmable-cifar100 | workflow | 0.3289 | 38171748 | 166532 |
| 4 | conditional-computation-cifar100 | workflow | 0.3249 | 28127140 | 128620 |
| 5 | iamnn-cifar100 | workflow | 0.3222 | 19995940 | 78436 |
| 6 | skipnet-cifar100 | workflow | 0.3153 | 18372964 | 84484 |
| 7 | channel-gating-cifar100 | workflow | 0.2896 | 18372964 | 84484 |
| 8 | instance-wise-sparsity-cifar100 | workflow | 0.2652 | 11127780 | 53828 |

## Accuracy-FLOP Pareto Frontier

- `instance-wise-sparsity-cifar100`: acc=0.2652, flop_proxy=11127780, params=53828
- `skipnet-cifar100`: acc=0.3153, flop_proxy=18372964, params=84484
- `wide-cnn-cifar100-bn`: acc=0.3359, flop_proxy=18527236, params=106788
- `routed-resnet-largest-cifar100`: acc=0.3654, flop_proxy=38171748, params=166532