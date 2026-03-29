from pathlib import Path

from dynanets.protocol import BenchmarkProtocol



def test_benchmark_protocol_loads_and_resolves_configs() -> None:
    protocol_path = Path("benchmarks/track_a_synthetic_seed.yaml")
    protocol = BenchmarkProtocol.from_yaml(protocol_path)
    resolved = protocol.resolve_configs(protocol_path.parent)

    assert protocol.name == "track_a_synthetic_seed"
    assert protocol.track == "synthetic"
    assert protocol.seeds == [7, 11, 23, 42, 99]
    assert len(protocol.experiments) == 7
    assert resolved[0].config.endswith("fixed_mlp_spirals10.yaml")
    assert "accuracy" in protocol.required_metrics
