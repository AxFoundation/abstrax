"""EventInfo: event_basics + corrected_areas + energy_estimates -> event_info.

Merges all event-level data types into a single flat output. Downstream
analysis should load event_info rather than individual types. Add new
event-level data types to depends_on as they are implemented.
"""

from __future__ import annotations

import strax


class EventInfo(strax.MergeOnlyPlugin):
    """Merge all event-level data types into event_info."""

    __version__ = "0.0.2"

    depends_on = ("event_basics", "corrected_areas", "energy_estimates")
    provides = "event_info"
    data_kind = "events"
    save_when = strax.SaveWhen.ALWAYS
