"""Custom URLConfig protocols for abstrax.

This module registers all URL protocols used by abstrax plugins. The protocols
resolve to **local resources only** — no MongoDB, no xedocs, no network calls.

Registered protocols
--------------------
- ``detector://key``       Look up ``key`` in the active detector file: by default
                           ``abstrax/config/detector.yaml``. A simulation package (ember)
                           activates its own file, which contains the same keys with the
                           same values plus its simulation settings
                           (:func:`use_detector_config_file`).
- ``pkg://name``           Load a file from ``abstrax/resources/`` (yaml, json,
                           npy/npz, csv).
- ``csv://path``           Load a CSV file from any path as a pandas DataFrame.
- ``yaml://path``          Load a YAML file from any absolute path.
- ``json_file://path``     Load a JSON file from any absolute path. (Not ``json://``:
                           that name is straxen's JSON-*string* parser.)
- ``numpy://path``         Load a ``.npy`` or ``.npz`` file from any absolute path.
- ``constant://value``     Return the value as-is (useful for hashing scalars).
- ``strax_folder://dir?run_id=...&data_type=...``
                           Resolve the strax storage folder
                           ``dir/<run_id>-<data_type>-<lineage_hash>``.
- ``dummy_map://value``    Return a :class:`DummyMap` callable that returns *value*
                           for any input position array. Used as a placeholder until
                           real LCE/correction maps are available.
- ``grid_map://name?method=linear``
                           Load a correction map on a regular grid (fuse/ember format,
                           ``.json``, ``.json.gz`` or ``.pkl``) as a callable
                           :class:`abstrax.maps.GridMap`. ``name`` is a file in
                           ``abstrax/resources/`` or an absolute path.

Adding a new protocol
---------------------
Register it with ``@URLConfig.register('your-name')``. Document the URL grammar
in the docstring. Add a unit test in ``tests/test_url_protocols.py``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from strax import URLConfig

from abstrax.maps import GridMap, load_grid_map

# Anchor for package-relative resource resolution.
_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _PACKAGE_ROOT / "config"
_RESOURCE_DIR = _PACKAGE_ROOT / "resources"

DETECTOR_CONFIG_FILE = _CONFIG_DIR / "detector.yaml"
_active_detector_config_file = DETECTOR_CONFIG_FILE


def _register(protocol: str):
    """Like @URLConfig.register but skips silently if already registered.

    abstrax no longer imports straxen, but users may import it alongside abstrax
    (in either order). This guard prevents double-registration errors if straxen
    (or another package) already claimed a name. Never reuse a name that straxen
    registers (e.g. ``json``, ``resource``, ``format``, ``take``, ``itp_map``):
    if abstrax registers it first, a later ``import straxen`` raises ValueError.
    """

    def decorator(func):
        if protocol not in URLConfig._LOOKUP:
            URLConfig.register(protocol)(func)
        return func

    return decorator


@lru_cache(maxsize=None)
def _load_detector_config(path: Path = DETECTOR_CONFIG_FILE) -> dict[str, Any]:
    """Read and cache a detector configuration yaml file.

    Defaults to abstrax's canonical ``abstrax/config/detector.yaml``. Other
    packages (e.g. ember) pass the path of their own detector file and wrap
    this loader in their own protocol, so the file location is never
    hardcoded here. The result is cached per path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Detector config not found at {path}. "
            "A detector.yaml describing the TPC geometry is required."
        )
    with path.open("r") as f:
        return yaml.safe_load(f)


def lookup_detector_value(path: Path, key: str) -> Any:
    """Return ``key`` from the detector configuration file at ``path``.

    :param path: path of the detector yaml file.
    :param key: name of the entry.
    :return: the configured value.
    """
    cfg = _load_detector_config(Path(path))
    if key not in cfg:
        raise KeyError(
            f"Detector config {path} has no entry '{key}'. Available keys: {sorted(cfg.keys())}"
        )
    return cfg[key]


def use_detector_config_file(path: Path) -> None:
    """Make ``detector://`` read the detector file at ``path``.

    Used by ember, whose detector file holds all keys of abstrax's file (same names and values)
    plus the simulation settings, so that ember's and abstrax's plugins share one protocol and
    can be registered in the same context. Call it before any plugin is set up: strax may cache
    evaluated URLs.

    :param path: path of the detector yaml file.
    """
    global _active_detector_config_file
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detector config not found at {path}.")
    _active_detector_config_file = path


def active_detector_config_file() -> Path:
    """Return the detector file that abstrax's ``detector://`` protocol currently reads.

    If ember was imported before abstrax, ``detector://`` is ember's protocol and this file is not
    used. Read detector values with ``strax.URLConfig.evaluate_dry("detector://key")`` instead.

    :return: path of the active detector yaml file.
    """
    return _active_detector_config_file


@_register("detector")
def detector_protocol(key: str, **_) -> Any:
    """Resolve ``detector://key`` to a value from the active detector config.

    Examples
    --------
    >>> URLConfig.evaluate_dry("detector://n_tpc_pmts")
    494
    """
    return lookup_detector_value(_active_detector_config_file, key)


@_register("pkg")
def pkg_protocol(name: str, fmt: str = "auto", **_) -> Any:
    """Load a packaged resource from ``abstrax/resources/`` via ``pkg://name``.

    Format is auto-detected from the extension unless overridden
    (yaml/yml, json, npy/npz, csv). CSV files are returned as a pandas
    DataFrame with the first column as index, e.g. ``pkg://pmt_positions.csv``.
    Renamed from ``resource://`` to avoid conflict with straxen's built-in protocol.
    """
    path = _RESOURCE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"No packaged resource named '{name}' at {path}")
    return _load_by_format(path, fmt)


@_register("yaml")
def yaml_protocol(path: str, **_) -> Any:
    """Load a YAML file by absolute or working-directory-relative path."""
    return _load_by_format(Path(path), "yaml")


@_register("json_file")
def json_file_protocol(path: str, **_) -> Any:
    """Load a JSON file by absolute or working-directory-relative path."""
    return _load_by_format(Path(path), "json")


@_register("numpy")
def numpy_protocol(path: str, **_) -> np.ndarray:
    """Load a NumPy array (.npy or .npz) by path."""
    return _load_by_format(Path(path), "numpy")


@_register("csv")
def csv_protocol(path: str, index_col: str = "0", **_):
    """Load a CSV file by path and return a pandas DataFrame.

    Uses ``index_col=0`` by default (stripping the row-number column written
    by ``DataFrame.to_csv()``). Pass ``index_col=-1`` to disable.
    """
    return _read_csv(Path(path), index_col)


def _read_csv(path: Path, index_col: Any = 0):
    """Read a CSV into a pandas DataFrame.

    ``index_col`` is the column number used as the index. A negative value
    means no index column (pandas would otherwise treat -1 as the last column).
    """
    import pandas as pd

    index_col = int(index_col)
    return pd.read_csv(path, index_col=index_col if index_col >= 0 else None)


@_register("constant")
def constant_protocol(value: Any, **_) -> Any:
    """Identity protocol. Useful when a plugin field needs to accept a URL
    syntactically but the value is just a literal."""
    return value


class DummyMap:
    """Placeholder correction map returning a constant for every input position.

    Mirrors the call signature of straxen's interpolating maps:
    ``map(positions)`` with ``positions`` of shape ``(n, ndim)`` returns an
    array of shape ``(n, *shape)`` filled with ``value``.
    """

    def __init__(self, value: float, shape: tuple = ()):
        self.value = float(value)
        self.shape = tuple(shape)

    def __call__(self, positions, **_) -> np.ndarray:
        return np.full((len(positions), *self.shape), self.value)


@_register("dummy_map")
def dummy_map_protocol(value: str, shape: tuple = (), **_) -> DummyMap:
    """Resolve ``dummy_map://value?shape=[n]`` to a :class:`DummyMap`."""
    return DummyMap(value, shape)


@_register("grid_map")
def grid_map_protocol(name: str, method: str = "linear", **_) -> GridMap:
    """Resolve ``grid_map://name?method=linear`` to a :class:`abstrax.maps.GridMap`.

    Example: ``grid_map://s1_lce_correction_map.pkl`` returns the S1 light collection
    efficiency map of ``abstrax/resources/``, a callable taking positions of shape ``(n, 3)``.

    :param name: map file (``.json``, ``.json.gz`` or ``.pkl``) in ``abstrax/resources/`` or an
        absolute path.
    :param method: interpolation method, ``linear`` (default) or ``nearest``.
    :return: the map.
    """
    return load_grid_map(resolve_resource_path(name), method=method)


def resolve_resource_path(name: str) -> Path:
    """Return the path of a file, relative to ``abstrax/resources/`` unless absolute.

    :param name: file name in ``abstrax/resources/`` or an absolute path.
    :return: the path of the existing file.
    """
    path = Path(name)
    if not path.is_absolute():
        path = _RESOURCE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    return path


@_register("strax_folder")
def strax_folder_protocol(base_dir: str, run_id: str, data_type: str, **_) -> str:
    """Resolve ``strax_folder://dir?run_id=...&data_type=...`` to the folder
    ``dir/<run_id>-<data_type>-<lineage_hash>`` written by a strax DataDirectory.

    Use ``run_id=plugin.run_id`` to fill in the run of the plugin being processed.
    The lineage hash is not known in advance, so exactly one matching folder
    must exist.
    """
    run_id = f"{int(run_id):06d}" if str(run_id).isdigit() else str(run_id)
    matches = sorted(Path(base_dir).glob(f"{run_id}-{data_type}-*"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{run_id}-{data_type}-*' folder in {base_dir}, "
            f"found {[m.name for m in matches]}"
        )
    return str(matches[0])


def _load_by_format(path: Path, fmt: str) -> Any:
    """Dispatch on format extension."""
    if fmt == "auto":
        fmt = path.suffix.lstrip(".").lower()
        if fmt in ("yml",):
            fmt = "yaml"
        if fmt in ("npy", "npz"):
            fmt = "numpy"

    if fmt == "yaml":
        with path.open("r") as f:
            return yaml.safe_load(f)
    if fmt == "json":
        with path.open("r") as f:
            return json.load(f)
    if fmt == "numpy":
        return np.load(path)
    if fmt == "csv":
        return _read_csv(path)
    raise ValueError(f"Unsupported format '{fmt}' for {path}")
