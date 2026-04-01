# Protocol Acceptance

Protocol: `track_a_synthetic_official_v1`
Track: `synthetic_10d`
Tier: `official`
Status: PASSED

## Summary

- runs: 40
- experiments: 8
- seeds: 5
- baselines: 1
- methods: 7

## Checks

- [PASS] `min_seeds`: required>=5, observed=5
- [PASS] `required_roles`: required=['baseline', 'method'], missing=none
- [PASS] `required_experiments`: required=['adanet_spirals10', 'den_spirals10', 'dynamic_nodes_spirals10', 'edge_growth_spirals10', 'fixed_mlp_spirals10', 'gradmax_spirals10', 'nest_spirals10', 'weights_connections_spirals10'], missing=none
- [PASS] `minimum_methods`: required>=7, observed=7
- [PASS] `minimum_baselines`: required>=1, observed=1
- [PASS] `require_constraints`: missing=none
- [PASS] `require_runtime_environment`: missing=none
- [PASS] `required_metrics`: missing=none