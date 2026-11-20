"""Correction maps interpolated on a regular grid.

A map file holds a dictionary in the format used by fuse, straxen and ember for regular grids::

    {
        "coordinate_system": [["x", [-66.4, 66.4, 20]], ["y", [-66.4, 66.4, 20]],
                              ["z", [-154.6555, 0.0, 50]]],
        "map": <array of shape (20, 20, 50)>,
        "name": "...",
        "description": "..."
    }

Each axis is ``[name, [min, max, n_points]]`` with grid points ``np.linspace(min, max, n_points)``.
A map has the shape ``(n_1, ..., n_d)`` (one value per grid point) or ``(n_1, ..., n_d,
*value_shape)`` (e.g. one value per PMT channel). Every key that is not a metadata key is a map,
so one file can hold several maps on the same grid (e.g. ``"top"`` and ``"bottom"``).

Supported files:

- ``.json`` and ``.json.gz``: the dictionary above, with lists instead of arrays.
- ``.pkl``: the pickled dictionary as written by fuse and ember. ``map`` may be compressed
  (``"compressed": (compressor, dtype, shape)``, one of strax's compressors) and quantized
  (``"quantized": step``, the stored integers times ``step`` give the values). Loading a pickle
  file can run code, so only load files from sources you trust.

The interpolation uses ``scipy.interpolate.RegularGridInterpolator``, as ember's
``RegularGridMap`` and ``straxen.InterpolatingMap(method="RegularGridInterpolator")``. Positions
outside the grid are moved onto the grid boundary first: the map is continued with its boundary
values instead of being extrapolated. Positions with a non-finite coordinate (NaN, e.g. an event
without S2) give NaN.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import strax
from scipy.interpolate import RegularGridInterpolator

METADATA_KEYS = (
    "coordinate_system",
    "name",
    "description",
    "timestamp",
    "compressed",
    "quantized",
)
DEFAULT_MAP_NAME = "map"


def decode_map_values(map_data: dict) -> dict:
    """Return a copy of the map dictionary with ``map`` decompressed and dequantized.

    :param map_data: map dictionary, possibly with the keys ``compressed`` and ``quantized``.
    :return: new dictionary without ``compressed`` and ``quantized``.
    """
    decoded = dict(map_data)
    if "compressed" in decoded:
        compressor, dtype, shape = decoded.pop("compressed")
        raw_bytes = strax.io.COMPRESSORS[compressor]["decompress"](decoded[DEFAULT_MAP_NAME])
        decoded[DEFAULT_MAP_NAME] = np.frombuffer(raw_bytes, dtype=dtype).reshape(*shape)
    if "quantized" in decoded:
        step = decoded.pop("quantized")
        decoded[DEFAULT_MAP_NAME] = step * np.asarray(decoded[DEFAULT_MAP_NAME], dtype=np.float64)
    return decoded


class GridMap:
    """Callable map interpolating values on a regular grid.

    :param map_data: dictionary with ``coordinate_system`` and one or more maps (see module
        docstring). Compressed and quantized maps are decoded.
    :param method: interpolation method, ``"linear"`` or ``"nearest"``.
    """

    def __init__(self, map_data: dict, method: str = "linear"):
        map_data = decode_map_values(map_data)
        self.method = method
        self.name = map_data.get("name", "")
        self.description = map_data.get("description", "")
        self.axis_names = [axis_name for axis_name, _ in map_data["coordinate_system"]]
        self.grid = [
            np.linspace(float(lower), float(upper), int(n_points))
            for _, (lower, upper, n_points) in map_data["coordinate_system"]
        ]
        self.lower_bounds = np.array([axis[0] for axis in self.grid])
        self.upper_bounds = np.array([axis[-1] for axis in self.grid])

        self.map_names = sorted(key for key in map_data if key not in METADATA_KEYS)
        if not self.map_names:
            raise ValueError(f"Map '{self.name}' contains no map values.")
        grid_shape = tuple(len(axis) for axis in self.grid)
        self.maps = {}
        self.interpolators = {}
        for map_name in self.map_names:
            values = np.asarray(map_data[map_name], dtype=np.float64)
            if values.shape[: len(grid_shape)] != grid_shape:
                raise ValueError(
                    f"Map '{self.name}/{map_name}' has shape {values.shape}, which does not match "
                    f"the grid shape {grid_shape}."
                )
            self.maps[map_name] = values
            self.interpolators[map_name] = RegularGridInterpolator(self.grid, values, method=method)

    @property
    def n_dimensions(self) -> int:
        """Number of coordinates of a position."""
        return len(self.grid)

    def __call__(self, positions: np.ndarray, map_name: str = DEFAULT_MAP_NAME) -> np.ndarray:
        """Evaluate the map.

        :param positions: array of shape ``(n, n_dimensions)``.
        :param map_name: name of the map in the file, ``"map"`` by default.
        :return: array of shape ``(n,)`` for maps with one value per grid point, otherwise
            ``(n, *value_shape)``. NaN for positions with a non-finite coordinate.
        """
        if map_name not in self.interpolators:
            raise KeyError(
                f"Map '{self.name}' has no map '{map_name}'. Available: {self.map_names}"
            )
        positions = self._validated_positions(positions)
        n_dimensions = self.n_dimensions
        value_shape = self.maps[map_name].shape[n_dimensions:]
        result = np.full((len(positions), *value_shape), np.nan)
        position_is_finite = np.all(np.isfinite(positions), axis=1)
        if np.any(position_is_finite):
            clipped_positions = np.clip(
                positions[position_is_finite], self.lower_bounds, self.upper_bounds
            )
            result[position_is_finite] = self.interpolators[map_name](clipped_positions)
        return result

    def _validated_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != self.n_dimensions:
            raise ValueError(
                f"Map '{self.name}' expects positions of shape (n, {self.n_dimensions}) with the "
                f"coordinates {self.axis_names}, got shape {positions.shape}."
            )
        return positions


def load_map_data(path: str | Path) -> dict:
    """Read a map dictionary from a ``.json``, ``.json.gz`` or ``.pkl`` file.

    :param path: path of the map file.
    :return: the map dictionary, still compressed and quantized if stored like that.
    """
    path = Path(path)
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt") as map_file:
            return json.load(map_file)
    if path.suffix == ".json":
        with path.open("r") as map_file:
            return json.load(map_file)
    if path.suffix == ".pkl":
        with path.open("rb") as map_file:
            return pickle.load(map_file)
    raise ValueError(f"Unsupported map file format: {path.name}. Use .json, .json.gz or .pkl.")


def load_grid_map(path: str | Path, method: str = "linear") -> GridMap:
    """Load a map from a ``.json``, ``.json.gz`` or ``.pkl`` file.

    :param path: path of the map file.
    :param method: interpolation method, ``"linear"`` or ``"nearest"``.
    :return: the map.
    """
    return GridMap(load_map_data(path), method=method)
