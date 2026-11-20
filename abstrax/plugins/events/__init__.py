"""Event-level processing plugins for abstrax.

Plugin chain at this level:

    peak_basics                   ->  events              (events.py)
    events + peak_basics
          + peak_positions        ->  event_basics        (event_basics.py)
    event_basics                  ->  corrected_areas     (corrected_areas.py)
    corrected_areas               ->  energy_estimates    (energy_estimates.py)
    event_basics
      + corrected_areas
      + energy_estimates          ->  event_info          (event_info.py)
"""

from abstrax.plugins.events.corrected_areas import CorrectedAreas
from abstrax.plugins.events.energy_estimates import EnergyEstimates
from abstrax.plugins.events.event_basics import EventBasics
from abstrax.plugins.events.event_info import EventInfo
from abstrax.plugins.events.events import Events

__all__ = [
    "Events",
    "EventBasics",
    "CorrectedAreas",
    "EnergyEstimates",
    "EventInfo",
]
