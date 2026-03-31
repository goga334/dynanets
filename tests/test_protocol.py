from pathlib import Path

from dynanets.protocol import BenchmarkProtocol



def test_all_benchmark_manifests_load_and_resolve_configs() -> None:
    benchmark_dir = Path("benchmarks")
    manifest_paths = sorted(benchmark_dir.glob("*.yaml"))

    assert manifest_paths

    manifest_names = {path.stem for path in manifest_paths}
    assert "track_a_synthetic_seed" in manifest_names
    assert "track_a_wave1_preview" in manifest_names
    assert "track_a_cnn_wave1_preview" in manifest_names
    assert "track_b_mnist_wave1_preview" in manifest_names
    assert "track_b_mnist_phase7_preview" in manifest_names

    for manifest_path in manifest_paths:
        protocol = BenchmarkProtocol.from_yaml(manifest_path)
        resolved = protocol.resolve_configs(manifest_path.parent)

        assert protocol.name == manifest_path.stem
        assert protocol.track
        assert protocol.seeds
        assert protocol.required_metrics
        assert len(resolved) == len(protocol.experiments)
        for experiment in resolved:
            assert Path(experiment.config).exists(), experiment.config

