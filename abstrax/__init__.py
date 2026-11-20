"""abstrax: abstract strax-based processing for a generic large-scale LXe TPC."""

from abstrax._version import __version__

# Register custom URLConfig protocols on import so they're available everywhere.
# This must happen before any plugin or context construction.
from abstrax import url_protocols  # noqa: F401

from abstrax import contexts, plugins
from abstrax.misc import print_versions

__all__ = [
    "__version__",
    "contexts",
    "plugins",
    "print_versions",
    "url_protocols",
]
