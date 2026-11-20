"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def tmp_strax_dir(tmp_path):
    """Per-test strax DataDirectory."""
    strax_dir = tmp_path / "strax_data"
    strax_dir.mkdir()
    return strax_dir
