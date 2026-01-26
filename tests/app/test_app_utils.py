import pytest
from app.utils import collect_class_ids

def test_collect_class_ids():
    # returns empty list when roles are empty
    results = {
        "http://localhost:8000/player-detection/image": {
            "mapping_class": {"1": "forward", "2": "goalkeeper"}
        }
    }
    assert collect_class_ids(results, roles=[]) == []

    # raises error when roles are missing
    results = {
        "http://localhost:8000/player-detection/image": {
            "mapping_class": {"1": "forward", "2": "goalkeeper"}
        }
    }
    with pytest.raises(ValueError, match="roles is missing."):
        collect_class_ids(results)

    # returns sorted ids for valid roles
    results = {
        "http://localhost:8000/player-detection/image": {
            "mapping_class": {"1": "forward", "2": "goalkeeper", "3": "forward"}
        }
    }
    assert collect_class_ids(results, roles=["forward"]) == [1, 3]

    # ignores non-convertible keys in mapping
    results = {
        "http://localhost:8000/player-detection/image": {
            "mapping_class": {"1": "forward", "two": "forward", "3": "goalkeeper"}
        }
    }
    assert collect_class_ids(results, roles=["forward"]) == [1]

    # handles missing endpoint or mapping key gracefully
    results = {}
    assert collect_class_ids(results, roles=["forward"]) == []

    # returns empty list when no matching roles
    results = {
        "http://localhost:8000/player-detection/image": {
            "mapping_class": {"1": "forward", "2": "goalkeeper"}
        }
    }
    assert collect_class_ids(results, roles=["defender"]) == []
