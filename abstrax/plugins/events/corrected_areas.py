"""CorrectedAreas: event_basics -> corrected_areas.

Applies three signal corrections to the raw S1 and S2 areas:

  1. S1 xyz light collection efficiency (LCE) correction
  2. S2 xy light collection efficiency correction
  3. Electron lifetime correction for S2

Maps default to ``dummy_map://1`` (see ``abstrax/url_protocols.py``), which returns
a flat correction of 1.0 everywhere. Replace with real interpolating maps
(e.g. ``itp_map://...``) once they are available.

Both the main and alternate S1 are corrected using the interaction position
inferred from the main S2 (same xyz).  The alternate S2 uses its own xy
position and its own drift time for the elife correction.
"""

from __future__ import annotations

import numpy as np
import strax


class CorrectedAreas(strax.Plugin):
    """Apply S1 xyz, S2 xy, and electron lifetime corrections to event areas.

    Outputs corrected areas (cS1, cS2) for main and alternate S1/S2 together
    with the (unchanged) area-fraction-top of the corrected S2.
    """

    __version__ = "0.0.3"

    depends_on = ("event_basics",)
    provides = "corrected_areas"
    data_kind = "events"

    s1_xyz_map = strax.URLConfig(
        default="dummy_map://1",
        cache=True,
        help=(
            "S1 xyz LCE correction map. Callable: (n, 3) positions [cm] → (n,) "
            "correction factors. Divide to apply: cS1 = s1_area / map(xyz)."
        ),
    )
    s2_xy_map = strax.URLConfig(
        default="dummy_map://1",
        cache=True,
        help=(
            "S2 xy LCE correction map. Callable: (n, 2) positions [cm] → (n,) "
            "correction factors. Divide to apply: cS2 = s2_area / map(xy) * elife."
        ),
    )
    electron_lifetime_liquid = strax.URLConfig(
        default="detector://electron_lifetime_liquid",
        type=float,
        track=True,
        help="Electron lifetime [ns].",
    )
    drift_velocity_liquid = strax.URLConfig(
        default="detector://drift_velocity_liquid",
        type=float,
        track=True,
        help="Electron drift velocity [cm/ns], used to convert drift_time to z.",
    )

    def infer_dtype(self):
        dtype = list(strax.time_fields)
        for prefix, peak_name in [("", "main"), ("alt_", "alternate")]:
            dtype += [
                (f"{prefix}cs1", np.float32, f"Corrected area of {peak_name} S1 [PE]"),
                (f"{prefix}cs2", np.float32, f"Corrected area of {peak_name} S2 [PE]"),
                (
                    f"{prefix}cs2_area_fraction_top",
                    np.float32,
                    f"Area fraction top of {peak_name} S2 (unchanged by single LCE map)",
                ),
            ]
        return dtype

    def _drift_time_to_z(self, drift_time_ns):
        """Drift time [ns] → z [cm] below gate (negative convention)."""
        return -drift_time_ns * self.drift_velocity_liquid

    def _elife_correction(self, drift_time_ns):
        """Electron lifetime correction factor exp(t / τ), >1 for longer drift."""
        return np.exp(drift_time_ns / self.electron_lifetime_liquid)

    def compute(self, events):
        result = np.zeros(len(events), dtype=self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        # Interaction position: xy from main S2, z from main drift time.
        # NaN propagates naturally for events missing S1 or S2.
        x = events["s2_x"]
        y = events["s2_y"]
        z = self._drift_time_to_z(events["drift_time"])
        xyz_pos = np.column_stack([x, y, z])

        # S1 xyz correction — same interaction position for both main and alt S1.
        s1_xyz_corr = self.s1_xyz_map(xyz_pos)
        for prefix in ("", "alt_"):
            result[f"{prefix}cs1"] = events[f"{prefix}s1_area"] / s1_xyz_corr

        # Main S2: xy LCE correction + electron lifetime correction.
        s2_xy_corr = self.s2_xy_map(np.column_stack([x, y]))
        elife_corr = self._elife_correction(events["drift_time"])
        result["cs2"] = events["s2_area"] / s2_xy_corr * elife_corr
        # AFT is unchanged when using a single combined LCE map (no separate top/bottom map).
        result["cs2_area_fraction_top"] = events["s2_area_fraction_top"]

        # Alt S2: xy correction at alt S2 position + elife from alt drift time.
        alt_xy_pos = np.column_stack([events["alt_s2_x"], events["alt_s2_y"]])
        alt_elife_corr = self._elife_correction(events["alt_s2_interaction_drift_time"])
        result["alt_cs2"] = events["alt_s2_area"] / self.s2_xy_map(alt_xy_pos) * alt_elife_corr
        result["alt_cs2_area_fraction_top"] = events["alt_s2_area_fraction_top"]

        return result
