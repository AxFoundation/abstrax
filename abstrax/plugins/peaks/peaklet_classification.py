"""Peaklet classification: peaklets -> peaklet_classification.

Vanilla heuristic classification based on rise time, area, area-fraction-top,
and tight coincidence.  Closely follows straxen's PeakletClassificationVanilla.

S1 criteria
-----------
* Large S1 (area >= 100 PE):  rise_time <= s1_max_rise_time_post100
                               tight_coincidence >= s1_min_coincidence
* Small S1 (area < 100 PE):   rise_time < empirical boundary in (rise_time, area) space
                               rise_time < empirical boundary in (rise_time, AFT) space
                               tight_coincidence >= s1_min_coincidence

S2 criteria
-----------
* n_channels >= s2_min_pmts  AND  not classified as S1
"""

from __future__ import annotations

from typing import Union

import numpy as np
import strax


class PeakletClassification(strax.Plugin):
    """Classify peaklets as unknown (0), S1-like (1), or S2-like (2).

    Outputs a lightweight ``peaklet_classification`` array containing only
    the interval fields and a ``type`` byte.  The downstream ``Peaks`` plugin
    merges this back onto the peaklets.
    """

    __version__ = "0.0.1"

    depends_on = "peaklets"
    provides: Union[str, tuple] = "peaklet_classification"
    data_kind = "peaklets"
    parallel = True

    dtype = strax.peak_interval_dtype + [
        ("type", np.int8, "0 unknown, 1 S1-like, 2 S2-like"),
    ]

    # --- S1 classification parameters ----------------------------------------

    s1_risetime_area_parameters = strax.URLConfig(
        default=(50, 80, 12),
        type=(list, tuple),
        help=(
            "Parameters (norm, const, tau) for the empirical upper boundary "
            "in the (rise_time, area) plane: norm * exp(-area / tau) + const."
        ),
    )

    s1_risetime_aft_parameters = strax.URLConfig(
        default=(-1, 2.6),
        type=(list, tuple),
        help=(
            "Parameters (slope, offset) for the empirical upper boundary "
            "in the (rise_time, area_fraction_top) plane: 10^(slope * AFT + offset)."
        ),
    )

    s1_flatten_threshold_aft = strax.URLConfig(
        default=(0.6, 100),
        type=(tuple, list),
        help=(
            "Two-element tuple (aft_boundary, flat_rise_time): above aft_boundary "
            "the rise-time boundary is flattened to flat_rise_time ns."
        ),
    )

    s1_max_rise_time_post100 = strax.URLConfig(
        default=200,
        type=(int, float),
        help="Maximum rise time (ns) for S1-like peaklets with area >= 100 PE.",
    )

    s1_min_coincidence = strax.URLConfig(
        default=2,
        type=int,
        help="Minimum tight-coincidence channel count required to classify as S1.",
    )

    # --- S2 classification parameters ----------------------------------------

    s2_min_pmts = strax.URLConfig(
        default=4,
        type=int,
        help="Minimum number of PMTs contributing to classify as S2.",
    )

    # -------------------------------------------------------------------------

    @staticmethod
    def upper_rise_time_area_boundary(area, norm, const, tau):
        """Upper rise-time boundary as a function of area: norm * exp(-area / tau) + const."""
        return norm * np.exp(-area / tau) + const

    @staticmethod
    def upper_rise_time_aft_boundary(aft, slope, offset, aft_boundary, flat_threshold):
        """Upper rise-time boundary as a function of area-fraction-top (AFT).

        Returns 10^(slope * AFT + offset), capped at flat_threshold above aft_boundary.
        """
        res = 10 ** (slope * aft + offset)
        res[aft >= aft_boundary] = flat_threshold
        return res

    def compute(self, peaklets):
        ptype = np.zeros(len(peaklets), dtype=np.int8)

        rise_time = -peaklets["area_decile_from_midpoint"][:, 1]
        n_channels = (peaklets["area_per_channel"] > 0).sum(axis=1)

        # Large S1: fast rise + coincidence, area >= 100 PE.
        is_large_s1 = peaklets["area"] >= 100
        is_large_s1 &= rise_time <= self.s1_max_rise_time_post100
        is_large_s1 &= peaklets["tight_coincidence"] >= self.s1_min_coincidence

        # Small S1: area < 100 PE, must satisfy both rise-time boundaries.
        is_small_s1 = peaklets["area"] < 100
        is_small_s1 &= rise_time < self.upper_rise_time_area_boundary(
            peaklets["area"], *self.s1_risetime_area_parameters
        )
        is_small_s1 &= rise_time < self.upper_rise_time_aft_boundary(
            peaklets["area_fraction_top"],
            *self.s1_risetime_aft_parameters,
            *self.s1_flatten_threshold_aft,
        )
        is_small_s1 &= peaklets["tight_coincidence"] >= self.s1_min_coincidence

        ptype[is_large_s1 | is_small_s1] = 1

        # S2: broad channel multiplicity, not already classified as S1.
        is_s2 = n_channels >= self.s2_min_pmts
        is_s2[is_large_s1 | is_small_s1] = False
        ptype[is_s2] = 2

        out = np.zeros(len(peaklets), dtype=self.dtype)
        out["type"] = ptype
        out["time"] = peaklets["time"]
        out["dt"] = peaklets["dt"]
        out["length"] = peaklets["length"]
        out["channel"] = -1
        return out
