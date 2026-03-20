from dynanets.registry import Registry


def test_registry_builds_registered_factory() -> None:
    registry: Registry[str] = Registry()
    registry.register("example", lambda value: f"built:{value}")

    result = registry.build("example", value="ok")

    assert result == "built:ok"
