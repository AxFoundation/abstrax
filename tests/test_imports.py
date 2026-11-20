"""Import smoke tests.

These verify that ``import abstrax`` does not trigger any forbidden behavior:
no MongoDB connections, no XENONnT-credential lookups, no network calls.

If these tests fail or hang, abstrax has accidentally imported part of the
straxen/utilix machinery it's supposed to stay clear of.
"""

import importlib


def test_import_abstrax():
    """The top-level package imports without side effects."""
    abstrax = importlib.import_module("abstrax")
    assert hasattr(abstrax, "__version__")
    assert hasattr(abstrax, "contexts")
    assert hasattr(abstrax, "plugins")
    assert hasattr(abstrax, "url_protocols")


def test_import_plugins():
    """All plugin modules import."""
    for name in ("raw_records", "records", "peaks", "events"):
        importlib.import_module(f"abstrax.plugins.{name}")


def test_print_versions():
    """print_versions runs and reports abstrax."""
    import abstrax

    df = abstrax.print_versions(modules=("abstrax", "not_a_real_module"), print_output=False)
    assert list(df["module"]) == ["python", "abstrax"]
    assert df["version"].iloc[1] == abstrax.__version__
    info = abstrax.print_versions(print_output=False, return_string=True, include_git=False)
    assert "abstrax" in info


def test_import_does_not_pull_in_straxen():
    """abstrax uses strax.URLConfig, so straxen/utilix must not be imported."""
    import subprocess
    import sys

    code = (
        "import sys, abstrax, abstrax.contexts; "
        "print('straxen' in sys.modules, 'utilix' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_straxen_importable_after_abstrax():
    """abstrax must not register protocol names that straxen also registers."""
    import subprocess
    import sys

    import pytest

    pytest.importorskip("straxen")
    subprocess.run([sys.executable, "-c", "import abstrax; import straxen"], check=True)
