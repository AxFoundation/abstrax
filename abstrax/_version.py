"""Single source of truth for the package version.

Kept separate from __init__.py so it can be read by setup tooling without
importing the full package (which would trigger straxen imports).
"""

__version__ = "0.0.1"
