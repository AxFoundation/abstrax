"""abstrax processing plugins.

Each module here registers strax.Plugin subclasses that together form the
processing chain:


    raw_records  (raw_records/raw_records.py)
        -> records               (records.py)
        -> peaklets              (peaks/peaklets.py)
        -> peaks                 (peaks/peaks.py)
        -> peak_basics           (peaks/peak_basics.py)
        -> peak_positions        (peaks/peak_positions.py)     [placeholder]
        -> events                (events/events.py)
        -> event_basics          (events/event_basics.py)
        -> event_info            (events/event_info.py)

Plugins are registered with a strax.Context in abstrax/contexts.py.
"""

from abstrax.plugins import events  # subpackage
from abstrax.plugins import peaks   # subpackage
from abstrax.plugins import raw_records  # subpackage
from abstrax.plugins import records  # subpackage

__all__ = ["events", "peaks", "records", "raw_records"]
