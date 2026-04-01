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
    assert "track_c_cifar_static_preview" in manifest_names
    assert "track_a_synthetic_official_v1" in manifest_names
    assert "track_b_mnist_official_v1" in manifest_names
    assert "track_c_cifar_official_extended" in manifest_names

    for manifest_path in manifest_paths:
        protocol = BenchmarkProtocol.from_yaml(manifest_path)
        resolved = protocol.resolve_configs(manifest_path.parent)

        assert protocol.name == manifest_path.stem
        assert protocol.track
        assert protocol.tier
        assert protocol.seeds
        assert protocol.required_metrics
        assert len(resolved) == len(protocol.experiments)
        for experiment in resolved:
            assert Path(experiment.config).exists(), experiment.config


def test_official_manifests_define_acceptance_rules() -> None:
    manifest_paths = [
        Path("benchmarks/track_a_synthetic_official_v1.yaml"),
        Path("benchmarks/track_b_mnist_official_v1.yaml"),
        Path("benchmarks/track_c_cifar_official_extended.yaml"),
    ]

    for manifest_path in manifest_paths:
        protocol = BenchmarkProtocol.from_yaml(manifest_path)
        assert protocol.tier.startswith("official")
        assert protocol.acceptance.min_seeds is not None
        assert protocol.acceptance.required_roles
        assert protocol.acceptance.required_experiments
        assert protocol.acceptance.require_constraints is True
        assert protocol.acceptance.require_runtime_environment is True
