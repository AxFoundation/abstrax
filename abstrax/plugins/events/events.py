"""Events: peak_basics -> events.

Defines event time windows by seeding on large S2-like peaks.
Adapted from straxen.plugins.events.Events but using detector:// configs
instead of xedocs:// and dropping XENONnT-specific features (S3 veto,
online corrections).
"""

from __future__ import annotations

import numpy as np
import strax


class Events(strax.OverlapWindowPlugin):
    """Define (time, endtime) event windows around triggering S2 peaks.

    A peak triggers an event if:
      1. type == 2 (S2-classified)
      2. area > trigger_min_area

    Nearby triggers are joined into a single event window.
    The window extends left by left_event_extension + drift_time_max and
    right by right_event_extension to capture the paired S1.
    """

    __version__ = "0.0.4"

    depends_on = ("peak_basics",)
    provides = "events"
    data_kind = "events"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = [
        ("event_number", np.int64, "Monotonically increasing event index"),
        ("time", np.int64, "Event start time in ns since the unix epoch"),
        ("endtime", np.int64, "Event end time in ns since the unix epoch"),
    ]

    events_seen = 0

    # --- Trigger thresholds --------------------------------------------------

    trigger_min_area = strax.URLConfig(
        default=120,
        type=(int, float),
        track=True,
        help="Minimum S2 area [PE] required to seed an event.",
    )

    # --- Event window extensions ---------------------------------------------

    left_event_extension = strax.URLConfig(
        default=int(0.1e6),
        type=(int, float),
        track=True,
        help=(
            "Extra left extension added on top of the maximum drift time [ns]. "
            "Ensures the paired S1 falls inside the event window."
        ),
    )

    right_event_extension = strax.URLConfig(
        default=int(0.1e6),
        type=(int, float),
        track=True,
        help="Extension to the right of the triggering S2 [ns].",
    )

    # --- Detector parameters (cm, ns) ---------------------------------------

    drift_velocity_liquid = strax.URLConfig(
        default="detector://drift_velocity_liquid",
        type=float,
        track=True,
        help="Electron drift velocity [cm/ns]. Used to compute drift_time_max.",
    )

    tpc_length = strax.URLConfig(
        default="detector://tpc_length",
        type=float,
        track=True,
        help="Cathode-to-gate drift length [cm]. Used to compute drift_time_max.",
    )

    diagnose_overlapping = strax.URLConfig(
        default=True,
        track=False,
        infer_type=False,
        help="Check at runtime that output event windows do not overlap.",
    )

    def setup(self):
        self.drift_time_max = int(self.tpc_length / self.drift_velocity_liquid)
        self.left_extension = self.left_event_extension + self.drift_time_max
        self.right_extension = self.right_event_extension

    def get_window_size(self):
        return 10 * (self.left_event_extension + self.drift_time_max + self.right_event_extension)

    def _is_triggering(self, peaks):
        mask = peaks["type"] == 2
        mask &= peaks["area"] > self.trigger_min_area
        return mask

    def compute(self, peaks, start, end):
        triggers = peaks[self._is_triggering(peaks)]

        t0, t1 = strax.find_peak_groups(
            triggers,
            gap_threshold=self.left_extension + self.right_extension + 1,
            left_extension=self.left_extension,
            right_extension=self.right_extension,
        )

        # Clip to chunk boundaries (overlap window artefact).
        t0 = np.clip(t0, start, end)
        t1 = np.clip(t1, start, end)

        result = np.zeros(len(t0), self.dtype)
        result["time"] = t0
        result["endtime"] = t1
        result["event_number"] = np.arange(len(result)) + self.events_seen
        self.events_seen += len(result)

        if self.diagnose_overlapping and len(result) > 1:
            assert np.all(
                (strax.endtime(result)[:-1] - result["time"][1:]) <= 0
            ), "Events not disjoint"

        return result
