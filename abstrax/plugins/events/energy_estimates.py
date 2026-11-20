"""EnergyEstimates: corrected_areas -> energy_estimates.

Converts corrected S1 (cs1) and S2 (cs2) areas from PE to energy [keVee]
using the mean S1 gain g1, mean S2 gain g2, and the LXe work function W.

    e_light  = W × cs1 / g1     (energy seen via scintillation)
    e_charge = W × cs2 / g2     (energy seen via ionisation)
    e_ces    = e_light + e_charge   (combined energy scale)

g1, g2, and W all come from ``abstrax/config/detector.yaml`` via the
``detector://`` URLConfig protocol. Update those values there when better
calibrations become available.
"""

from __future__ import annotations

import numpy as np
import strax


class EnergyEstimates(strax.Plugin):
    """Convert corrected S1/S2 areas to energy estimates in keVee."""

    __version__ = "0.0.2"

    depends_on = ("corrected_areas",)
    provides = "energy_estimates"
    data_kind = "events"
    save_when = strax.SaveWhen.TARGET

    g1 = strax.URLConfig(
        default="detector://g1",
        type=float,
        track=True,
        help="Mean S1 gain [PE / photon produced].",
    )
    g2 = strax.URLConfig(
        default="detector://g2",
        type=float,
        track=True,
        help="Mean S2 gain [PE / electron extracted].",
    )
    w_value_ev = strax.URLConfig(
        default="detector://w_value_ev",
        type=float,
        track=True,
        help="Mean energy per quantum [eV] (LXe work function).",
    )

    dtype = [
        ("e_light", np.float32, "Energy in S1 signal [keVee]"),
        ("e_charge", np.float32, "Energy in S2 signal [keVee]"),
        ("e_ces", np.float32, "Combined energy estimate [keVee]"),
    ] + strax.time_fields

    def compute(self, events):
        w_kev = self.w_value_ev * 1e-3
        e_light = events["cs1"] / self.g1 * w_kev
        e_charge = events["cs2"] / self.g2 * w_kev
        return dict(
            time=events["time"],
            endtime=strax.endtime(events),
            e_light=e_light.astype(np.float32),
            e_charge=e_charge.astype(np.float32),
            e_ces=(e_light + e_charge).astype(np.float32),
        )
