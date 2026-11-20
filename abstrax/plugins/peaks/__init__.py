"""Peak-level processing plugins for abstrax.

Plugin chain at this level:

    peaklets + lone_hits      (peaklets.py)
        -> peaklet_classification  (peaklet_classification.py)
        -> merged_s2s              (merged_s2s.py)
        -> peaks                   (peaks.py)
        -> peak_basics             (peak_basics.py)
        -> peak_positions          (peak_positions.py)    [placeholder]
"""

from abstrax.plugins.peaks.merged_s2s import MergedS2s
from abstrax.plugins.peaks.peak_basics import PeakBasics
from abstrax.plugins.peaks.peak_positions import PeakPositions
from abstrax.plugins.peaks.peaklet_classification import PeakletClassification
from abstrax.plugins.peaks.peaklets import Peaklets
from abstrax.plugins.peaks.peaks import Peaks

__all__ = [
    "Peaklets",
    "PeakletClassification",
    "MergedS2s",
    "Peaks",
    "PeakBasics",
    "PeakPositions",
]
