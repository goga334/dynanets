# Protocol Leaderboard

Protocol: `track_a_synthetic_official_v1`
Track: `synthetic_10d`
Tier: `official`

## Top Overall

- `edge-growth-spirals10`: acc=0.6283, flop_proxy=626, params=326

## Top Method

- `edge-growth-spirals10`: acc=0.6283, flop_proxy=626, params=326

## Top Baseline

- `fixed-mlp-spirals10`: acc=0.6131, flop_proxy=452, params=236

## Accuracy Ranking

| Rank | Method | Type | Mean final val acc | FLOP proxy | Params |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | edge-growth-spirals10 | dynamic | 0.6283 | 626 | 326 |
| 2 | adanet-spirals10 | workflow | 0.6227 | 808 | 420 |
| 3 | dynamic-nodes-spirals10 | dynamic | 0.6139 | 462 | 242 |
| 4 | fixed-mlp-spirals10 | baseline | 0.6131 | 452 | 236 |
| 5 | nest-spirals10 | dynamic | 0.6048 | 302 | 158 |
| 6 | gradmax-spirals10 | dynamic | 0.6045 | 352 | 184 |
| 7 | den-spirals10 | dynamic | 0.5917 | 352 | 184 |
| 8 | weights-connections-spirals10 | dynamic | 0.5861 | 452 | 236 |

## Accuracy-FLOP Pareto Frontier

- `nest-spirals10`: acc=0.6048, flop_proxy=302, params=158
- `fixed-mlp-spirals10`: acc=0.6131, flop_proxy=452, params=236
- `dynamic-nodes-spirals10`: acc=0.6139, flop_proxy=462, params=242
- `edge-growth-spirals10`: acc=0.6283, flop_proxy=626, params=326