"""Record-level processing plugins for abstrax.

Plugin chain at this level:

    raw_records  ->  records + pulse_counts   (records.py)
"""

from abstrax.plugins.records.records import PulseProcessing

__all__ = ["PulseProcessing"]
