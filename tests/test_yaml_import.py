import yaml

def test_yaml_points_to_external_package():
    assert yaml.__name__ == "yaml"
    assert hasattr(yaml, "safe_load")
