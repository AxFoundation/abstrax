"""Tests for abstrax URLConfig protocols."""

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import yaml


def test_detector_protocol_known_key():
    from strax import URLConfig

    n_tpc_pmts = URLConfig.evaluate_dry("detector://n_tpc_pmts")
    assert isinstance(n_tpc_pmts, int)
    assert n_tpc_pmts > 0


def test_detector_protocol_unknown_key_raises():
    from strax import URLConfig

    with pytest.raises(KeyError):
        URLConfig.evaluate_dry("detector://this_key_does_not_exist")


def test_yaml_protocol(tmp_path):
    from strax import URLConfig

    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"foo": 42, "bar": [1, 2, 3]}))
    val = URLConfig.evaluate_dry(f"yaml://{p}")
    assert val == {"foo": 42, "bar": [1, 2, 3]}


def test_json_file_protocol(tmp_path):
    from strax import URLConfig

    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1}))
    val = URLConfig.evaluate_dry(f"json_file://{p}")
    assert val == {"a": 1}


def test_constant_protocol():
    from strax import URLConfig

    val = URLConfig.evaluate_dry("constant://hello")
    assert val == "hello"


def test_pkg_protocol_missing_raises():
    from strax import URLConfig

    with pytest.raises(FileNotFoundError):
        URLConfig.evaluate_dry("pkg://does_not_exist.yaml")


def test_pkg_protocol_csv():
    from strax import URLConfig

    pmts = URLConfig.evaluate_dry("pkg://pmt_positions.csv")
    assert list(pmts.columns) == ["array", "x", "y"]
    assert len(pmts) > 0


def test_csv_protocol_index_col(tmp_path):
    from strax import URLConfig

    p = tmp_path / "x.csv"
    p.write_text("i,a,b\n0,1,2\n1,3,4\n")
    assert list(URLConfig.evaluate_dry(f"csv://{p}").columns) == ["a", "b"]
    no_index = URLConfig.evaluate_dry(f"csv://{p}?index_col=-1")
    assert list(no_index.columns) == ["i", "a", "b"]


def test_strax_folder_protocol(tmp_path):
    from strax import URLConfig

    folder = tmp_path / "012882-raw_records-abcdefghij"
    folder.mkdir()
    val = URLConfig.evaluate_dry(f"strax_folder://{tmp_path}?run_id=012882&data_type=raw_records")
    assert val == str(folder)


def test_strax_folder_protocol_missing_raises(tmp_path):
    from strax import URLConfig

    with pytest.raises(FileNotFoundError):
        URLConfig.evaluate_dry(f"strax_folder://{tmp_path}?run_id=000001&data_type=raw_records")


def test_dummy_map_protocol():
    import numpy as np
    from strax import URLConfig

    dummy_map = URLConfig.evaluate_dry("dummy_map://2.5")
    np.testing.assert_array_equal(dummy_map(np.zeros((4, 3))), np.full(4, 2.5))


def test_grid_map_protocol_loads_packaged_s1_lce_map():
    import numpy as np
    from strax import URLConfig

    from abstrax.maps import GridMap

    s1_lce_map = URLConfig.evaluate_dry("grid_map://s1_lce_correction_map.pkl?method=linear")
    assert isinstance(s1_lce_map, GridMap)
    assert s1_lce_map.axis_names == ["x", "y", "z"]
    positions = np.array([[0.0, 0.0, -75.0], [np.nan, np.nan, np.nan]])
    values = s1_lce_map(positions)
    assert 0.5 < values[0] < 2
    assert np.isnan(values[1])


def test_grid_map_protocol_absolute_path(tmp_path):
    import numpy as np
    from strax import URLConfig

    path = tmp_path / "map.json"
    path.write_text(
        json.dumps({"coordinate_system": [["x", [0.0, 2.0, 3]]], "map": [1.0, 2.0, 4.0]})
    )
    linear_map = URLConfig.evaluate_dry(f"grid_map://{path}?method=linear")
    nearest_map = URLConfig.evaluate_dry(f"grid_map://{path}?method=nearest")
    np.testing.assert_allclose(linear_map(np.array([[0.5], [1.5]])), [1.5, 3.0])
    np.testing.assert_allclose(nearest_map(np.array([[0.4], [1.6]])), [1.0, 4.0])


def test_grid_map_protocol_missing_file_raises():
    from strax import URLConfig

    with pytest.raises(FileNotFoundError):
        URLConfig.evaluate_dry("grid_map://does_not_exist.pkl")


def test_detector_config_channel_split_is_consistent():
    from abstrax.url_protocols import DETECTOR_CONFIG_FILE, _load_detector_config

    cfg = _load_detector_config(DETECTOR_CONFIG_FILE)
    assert cfg["n_top_pmts"] + cfg["n_bottom_pmts"] == cfg["n_tpc_pmts"]
    assert cfg["top_pmt_indices"] == [0, cfg["n_top_pmts"]]
    assert cfg["bottom_pmt_indices"] == [cfg["n_top_pmts"], cfg["n_tpc_pmts"]]


def test_lookup_detector_value_reads_other_file(tmp_path):
    from abstrax.url_protocols import lookup_detector_value

    path = tmp_path / "other_detector.yaml"
    path.write_text(yaml.safe_dump({"n_tpc_pmts": 7}))
    assert lookup_detector_value(path, "n_tpc_pmts") == 7


def test_lookup_detector_value_unknown_key_raises(tmp_path):
    from abstrax.url_protocols import lookup_detector_value

    path = tmp_path / "other_detector.yaml"
    path.write_text(yaml.safe_dump({"n_tpc_pmts": 7}))
    with pytest.raises(KeyError):
        lookup_detector_value(path, "n_top_pmts")


def test_lookup_detector_value_missing_file_raises(tmp_path):
    from abstrax.url_protocols import lookup_detector_value

    with pytest.raises(FileNotFoundError):
        lookup_detector_value(tmp_path / "missing.yaml", "n_tpc_pmts")


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    content=st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12),
        values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=8)),
        min_size=1,
        max_size=6,
    )
)
def test_lookup_detector_value_round_trip(tmp_path_factory, content):
    from abstrax.url_protocols import lookup_detector_value

    path = tmp_path_factory.mktemp("detector") / "detector.yaml"
    path.write_text(yaml.safe_dump(content))
    for key, value in content.items():
        assert lookup_detector_value(path, key) == value


@pytest.fixture
def restore_active_detector_file():
    from abstrax.url_protocols import active_detector_config_file, use_detector_config_file

    original = active_detector_config_file()
    yield
    use_detector_config_file(original)


def test_detector_protocol_reads_the_active_file(tmp_path, restore_active_detector_file):
    from strax import URLConfig

    from abstrax.url_protocols import active_detector_config_file, use_detector_config_file

    other_file = tmp_path / "other_detector.yaml"
    other_file.write_text("n_tpc_pmts: 7\n")
    use_detector_config_file(other_file)
    assert active_detector_config_file() == other_file
    assert URLConfig.evaluate_dry("detector://n_tpc_pmts") == 7


def test_activating_a_missing_detector_file_raises(tmp_path, restore_active_detector_file):
    from abstrax.url_protocols import use_detector_config_file

    with pytest.raises(FileNotFoundError):
        use_detector_config_file(tmp_path / "missing.yaml")


def test_default_detector_file_is_abstrax_file():
    from abstrax.url_protocols import DETECTOR_CONFIG_FILE, lookup_detector_value

    assert DETECTOR_CONFIG_FILE.parts[-3:] == ("abstrax", "config", "detector.yaml")
    assert lookup_detector_value(DETECTOR_CONFIG_FILE, "n_tpc_pmts") > 0


@pytest.mark.parametrize(
    "key, lower, upper",
    [
        ("tpc_length", 1.0, 1000.0),
        ("tpc_radius", 1.0, 1000.0),
        ("drift_velocity_liquid", 1e-6, 1e-3),
        ("electron_lifetime_liquid", 1e4, 1e10),
        ("pmt_to_pe", 1e-5, 1.0),
    ],
)
def test_detector_values_are_in_cm_and_ns(key, lower, upper):
    from abstrax.url_protocols import DETECTOR_CONFIG_FILE, lookup_detector_value

    assert lower < lookup_detector_value(DETECTOR_CONFIG_FILE, key) < upper
