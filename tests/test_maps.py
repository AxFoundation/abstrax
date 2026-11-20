"""Tests for abstrax.maps (regular-grid correction maps)."""

import gzip
import json
import pickle

import numpy as np
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from abstrax.maps import GridMap, decode_map_values, load_grid_map, load_map_data

LOWER_BOUNDS = np.array([-10.0, -10.0, -30.0])
UPPER_BOUNDS = np.array([10.0, 10.0, 0.0])
N_POINTS = (5, 7, 4)
SLOPES = np.array([0.5, -0.25, 0.1])
OFFSET = 20.0


def linear_function(positions: np.ndarray) -> np.ndarray:
    return OFFSET + positions @ SLOPES


def linear_map_data() -> dict:
    grid = [
        np.linspace(lower, upper, n_points)
        for lower, upper, n_points in zip(LOWER_BOUNDS, UPPER_BOUNDS, N_POINTS)
    ]
    mesh = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1)
    return {
        "coordinate_system": [
            [axis_name, [float(lower), float(upper), n_points]]
            for axis_name, lower, upper, n_points in zip(
                "xyz", LOWER_BOUNDS, UPPER_BOUNDS, N_POINTS
            )
        ],
        "map": linear_function(mesh.reshape(-1, 3)).reshape(N_POINTS),
        "name": "linear test map",
    }


def encoded_map_data(map_data: dict, step: float) -> dict:
    integers = np.round(np.asarray(map_data["map"]) / step).astype(np.uint32)
    encoded = dict(map_data)
    encoded["map"] = strax.io.COMPRESSORS["zstd"]["compress"](integers.tobytes())
    encoded["compressed"] = ("zstd", np.uint32, integers.shape)
    encoded["quantized"] = step
    return encoded


def json_ready(map_data: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in map_data.items()
    }


inside_positions = arrays(
    np.float64,
    st.tuples(st.integers(1, 20), st.just(3)),
    elements=st.floats(0, 1),
).map(lambda unit: LOWER_BOUNDS + unit * (UPPER_BOUNDS - LOWER_BOUNDS))

any_finite_positions = arrays(
    np.float64,
    st.tuples(st.integers(1, 20), st.just(3)),
    elements=st.floats(-1e6, 1e6),
)


def test_map_reproduces_grid_values():
    grid_map = GridMap(linear_map_data())
    corners = np.array([LOWER_BOUNDS, UPPER_BOUNDS])
    np.testing.assert_allclose(grid_map(corners), linear_function(corners))


@settings(deadline=None)
@given(positions=inside_positions)
def test_linear_interpolation_of_linear_function_is_exact(positions):
    grid_map = GridMap(linear_map_data())
    np.testing.assert_allclose(grid_map(positions), linear_function(positions), atol=1e-10)


@settings(deadline=None)
@given(positions=any_finite_positions)
def test_positions_outside_grid_take_boundary_value(positions):
    grid_map = GridMap(linear_map_data())
    clipped = np.clip(positions, LOWER_BOUNDS, UPPER_BOUNDS)
    np.testing.assert_allclose(grid_map(positions), linear_function(clipped), atol=1e-10)


@settings(deadline=None)
@given(
    positions=inside_positions,
    invalid_value=st.sampled_from([np.nan, np.inf, -np.inf]),
    data=st.data(),
)
def test_non_finite_positions_give_nan_and_leave_others_unchanged(positions, invalid_value, data):
    grid_map = GridMap(linear_map_data())
    row_is_invalid = np.array(
        data.draw(st.lists(st.booleans(), min_size=len(positions), max_size=len(positions)))
    )
    column = data.draw(st.integers(0, 2))
    positions_with_invalid = positions.copy()
    positions_with_invalid[row_is_invalid, column] = invalid_value

    result = grid_map(positions_with_invalid)

    assert np.all(np.isnan(result[row_is_invalid]))
    np.testing.assert_allclose(
        result[~row_is_invalid], linear_function(positions[~row_is_invalid]), atol=1e-10
    )


def test_all_positions_nan():
    grid_map = GridMap(linear_map_data())
    result = grid_map(np.full((3, 3), np.nan))
    assert result.shape == (3,)
    assert np.all(np.isnan(result))


def test_empty_positions_give_empty_result():
    grid_map = GridMap(linear_map_data())
    assert grid_map(np.zeros((0, 3))).shape == (0,)


def test_nearest_method_returns_grid_value():
    grid_map = GridMap(linear_map_data(), method="nearest")
    near_lower_corner = LOWER_BOUNDS + 0.1
    np.testing.assert_allclose(
        grid_map(near_lower_corner[np.newaxis]), linear_function(LOWER_BOUNDS[np.newaxis])
    )


def test_vector_map_and_map_names():
    map_data = linear_map_data()
    scalar_values = map_data["map"]
    map_data["map"] = np.stack([scalar_values, 2 * scalar_values], axis=-1)
    map_data["top"] = 3 * scalar_values
    grid_map = GridMap(map_data)
    positions = np.array([[0.0, 0.0, -15.0], [5.0, -5.0, -1.0]])

    expected = linear_function(positions)
    np.testing.assert_allclose(grid_map(positions), np.column_stack([expected, 2 * expected]))
    np.testing.assert_allclose(grid_map(positions, map_name="top"), 3 * expected)
    assert grid_map.map_names == ["map", "top"]


def test_vector_map_with_nan_positions_has_nan_rows():
    map_data = linear_map_data()
    map_data["map"] = np.stack([map_data["map"]] * 4, axis=-1)
    result = GridMap(map_data)(np.array([[0.0, 0.0, -15.0], [np.nan, 0.0, -15.0]]))
    assert result.shape == (2, 4)
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isnan(result[1]))


def test_unknown_map_name_raises():
    with pytest.raises(KeyError, match="no map 'top'"):
        GridMap(linear_map_data())(np.zeros((1, 3)), map_name="top")


@pytest.mark.parametrize("shape", [(3,), (3, 2), (3, 4), (2, 3, 1)])
def test_wrong_position_shape_raises(shape):
    with pytest.raises(ValueError, match="expects positions of shape"):
        GridMap(linear_map_data())(np.zeros(shape))


def test_map_shape_not_matching_grid_raises():
    map_data = linear_map_data()
    map_data["map"] = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="does not match the grid shape"):
        GridMap(map_data)


def test_map_without_values_raises():
    map_data = linear_map_data()
    del map_data["map"]
    with pytest.raises(ValueError, match="contains no map values"):
        GridMap(map_data)


@settings(deadline=None)
@given(
    values=arrays(np.float64, st.just((3, 4)), elements=st.floats(0, 1000)),
    step=st.sampled_from([1e-6, 1e-3, 0.5]),
)
def test_decode_reverts_quantization_and_compression(values, step):
    map_data = {"coordinate_system": [["x", [0, 1, 3]], ["y", [0, 1, 4]]], "map": values}
    decoded = decode_map_values(encoded_map_data(map_data, step))
    assert "compressed" not in decoded
    assert "quantized" not in decoded
    np.testing.assert_allclose(decoded["map"], values, atol=step / 2 * (1 + 1e-9))


def test_decode_leaves_plain_map_unchanged():
    map_data = linear_map_data()
    np.testing.assert_array_equal(decode_map_values(map_data)["map"], map_data["map"])


@pytest.mark.parametrize("file_name", ["map.json", "map.json.gz", "map.pkl", "encoded.pkl"])
def test_load_grid_map_from_file(tmp_path, file_name):
    map_data = linear_map_data()
    path = tmp_path / file_name
    if file_name == "map.json":
        path.write_text(json.dumps(json_ready(map_data)))
    elif file_name == "map.json.gz":
        with gzip.open(path, "wt") as map_file:
            json.dump(json_ready(map_data), map_file)
    elif file_name == "map.pkl":
        path.write_bytes(pickle.dumps(map_data))
    else:
        path.write_bytes(pickle.dumps(encoded_map_data(map_data, 1e-6)))

    grid_map = load_grid_map(path)
    positions = np.array([[1.0, 2.0, -3.0], [-9.0, 9.0, -29.0]])
    np.testing.assert_allclose(grid_map(positions), linear_function(positions), atol=1e-6)
    assert grid_map.name == "linear test map"
    assert grid_map.axis_names == ["x", "y", "z"]


def test_load_map_data_unsupported_format_raises(tmp_path):
    path = tmp_path / "map.npz"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported map file format"):
        load_map_data(path)
