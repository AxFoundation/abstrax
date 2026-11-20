"""EventBasics: events + peak_basics + peak_positions -> event_basics.

Implements vanilla (non-SOM) S1/S2 selection within each event window.
Adapted from straxen.plugins.events.EventBasicsVanilla but:
  - uses detector:// URLConfig instead of xedocs://
  - uses fixed x/y position fields instead of dynamic per-algorithm names
  - strips fields not present in abstrax peak_basics (proximity_score,
    n_competing, merged)
  - includes z/r/theta reconstruction from drift time and S2 xy position
    (no field-distortion or gate corrections)
"""

from __future__ import annotations

import numba
import numpy as np
import strax


class EventBasics(strax.Plugin):
    """Extract main and alternate S1/S2 for each event.

    The main S1 is the largest S1 before the main S2 (with tight
    coincidence >= event_s1_min_coincidence).  The alternate S1 is the
    second largest qualifying S1.  The main S2 is the largest S2; the
    alternate S2 is the largest other S2 within [main S1, main S1 +
    max_drift_time].

    Peak properties are copied directly from the merged peaks array
    (peak_basics + peak_positions).
    """

    __version__ = "0.0.5"

    depends_on = ("events", "peak_basics", "peak_positions")
    provides = "event_basics"
    data_kind = "events"

    # --- Detector parameters -------------------------------------------------

    drift_velocity_liquid = strax.URLConfig(
        default="detector://drift_velocity_liquid",
        type=float,
        track=True,
        help="Electron drift velocity [cm/ns].",
    )

    tpc_length = strax.URLConfig(
        default="detector://tpc_length",
        type=float,
        track=True,
        help="Cathode-to-gate drift length [cm].",
    )

    # --- S1/S2 selection options ---------------------------------------------

    event_s1_min_coincidence = strax.URLConfig(
        default=2,
        infer_type=False,
        track=True,
        help=(
            "Minimum tight_coincidence for a peak to be selected as main or "
            "alternate S1. Should be >= s1_min_coincidence in peaklet classification."
        ),
    )

    allow_posts2_s1s = strax.URLConfig(
        default=False,
        infer_type=False,
        track=True,
        help="Allow S1s after the main S2 to become the main S1.",
    )

    force_main_before_alt = strax.URLConfig(
        default=False,
        infer_type=False,
        track=True,
        help="Swap main/alt S1 (or S2) so the earlier one is always 'main'.",
    )

    force_alt_s2_in_max_drift_time = strax.URLConfig(
        default=True,
        infer_type=False,
        track=True,
        help="Require alt S2 to occur within [main S1, main S1 + drift_time_max].",
    )

    def setup(self):
        self.drift_time_max = int(self.tpc_length / self.drift_velocity_liquid)

    # --- dtype definition ----------------------------------------------------

    # Peak properties copied for each of main/alt S1 and main/alt S2.
    _peak_properties = (
        ("time", np.int64, "start time since unix epoch [ns]"),
        ("center_time", np.int64, "weighted average center time [ns]"),
        ("median_time", np.float32, "weighted relative median time [ns]"),
        ("endtime", np.int64, "end time since unix epoch [ns]"),
        ("area", np.float32, "area, uncorrected [PE]"),
        ("n_channels", np.int16, "count of contributing PMTs"),
        ("top_n_channels", np.int16, "count of contributing top PMTs"),
        ("n_hits", np.int32, "count of hits contributing at least one sample"),
        ("max_pmt", np.int16, "PMT number with the largest contribution"),
        ("max_pmt_area", np.float32, "area in the largest-contributing PMT [PE]"),
        ("range_50p_area", np.float32, "width, 50% area [ns]"),
        ("range_90p_area", np.float32, "width, 90% area [ns]"),
        ("rise_time", np.float32, "time from 10% to 50% area quantile [ns]"),
        ("area_fraction_top", np.float32, "fraction of area seen by the top array"),
        ("tight_coincidence", np.int16, "channels within tight range of mean"),
        ("n_saturated_channels", np.int16, "total number of saturated channels"),
    )

    def infer_dtype(self):
        dtype = strax.time_fields + [
            ("event_number", np.int64, "Event number in this dataset"),
            ("n_peaks", np.int32, "Number of peaks in the event"),
            ("drift_time", np.float32, "Drift time between main S1 and S2 [ns]"),
        ]

        for s_i in [1, 2]:
            dtype += [
                (f"s{s_i}_index", np.int32, f"Main S{s_i} index within the event"),
                (f"alt_s{s_i}_index", np.int32, f"Alternate S{s_i} index within the event"),
            ]
            for name, dt, comment in self._peak_properties:
                dtype += [
                    (f"s{s_i}_{name}", dt, f"Main S{s_i} {comment}"),
                    (f"alt_s{s_i}_{name}", dt, f"Alternate S{s_i} {comment}"),
                ]
            dtype += [
                (
                    f"alt_s{s_i}_interaction_drift_time",
                    np.float32,
                    f"Drift time using alternate S{s_i} [ns]",
                ),
                (f"alt_s{s_i}_delay", np.int32, f"Time between main and alternate S{s_i} [ns]"),
            ]

        # S1-only: hit timing spread
        for prefix in ["s1", "alt_s1"]:
            dtype += [
                (f"{prefix}_max_diff", np.int32, f"{prefix} largest apex-to-apex hit gap [ns]"),
                (f"{prefix}_min_diff", np.int32, f"{prefix} smallest apex-to-apex hit gap [ns]"),
                (f"{prefix}_first_channel", np.int16, f"{prefix} first channel (apex order)"),
                (f"{prefix}_last_channel", np.int16, f"{prefix} last channel (apex order)"),
            ]

        # S2-only: uncorrected position
        for prefix in ["s2", "alt_s2"]:
            dtype += [
                (f"{prefix}_x", np.float32, f"{prefix} reconstructed X [cm], uncorrected"),
                (f"{prefix}_y", np.float32, f"{prefix} reconstructed Y [cm], uncorrected"),
            ]

        # Interaction position derived from main S2 xy and drift time.
        # z convention: gate at z=0, cathode at z=-tpc_length (negative downward).
        # x/y are copies of s2_x/s2_y (no field-distortion correction in abstrax).
        dtype += [
            ("x", np.float32, "Interaction x [cm], copied from main S2 (no FDC)"),
            ("y", np.float32, "Interaction y [cm], copied from main S2 (no FDC)"),
            ("z", np.float32, "Interaction z [cm], from main drift time (gate at z=0)"),
            ("r", np.float32, "Interaction radius [cm], from main S2 xy"),
            ("theta", np.float32, "Interaction azimuthal angle [rad], from main S2 xy"),
            ("alt_s1_z", np.float32, "Interaction z [cm] using alternate S1 drift time"),
            ("alt_s2_z", np.float32, "Interaction z [cm] using alternate S2 drift time"),
        ]

        dtype += [
            ("area_before_main_s2", np.float32, "Sum of peak areas before main S2 [PE]"),
        ]
        for where in ["before", "after"]:
            dtype += [
                (
                    f"large_s2_{where}_main_s2_area",
                    np.float32,
                    f"Area of the largest S2 {where} main S2 [PE]",
                ),
                (
                    f"large_s2_{where}_main_s2_index",
                    np.int32,
                    f"Index of the largest S2 {where} main S2",
                ),
                (
                    f"large_s2_{where}_main_s2_center_time",
                    np.int64,
                    f"Center time of the largest S2 {where} main S2 [ns]",
                ),
            ]

        return dtype

    # --- Computation ---------------------------------------------------------

    def compute(self, events, peaks):
        result = np.zeros(len(events), dtype=self.dtype)
        strax.set_nan_defaults(result)

        result["time"] = events["time"]
        result["endtime"] = events["endtime"]
        result["event_number"] = events["event_number"]

        split_peaks = strax.split_by_containment(peaks, events)
        self._fill_events(result, split_peaks)

        # Interaction position — computed over all events after peak selection.
        v_cm_per_ns = self.drift_velocity_liquid
        result["x"] = result["s2_x"]
        result["y"] = result["s2_y"]
        result["z"] = -result["drift_time"] * v_cm_per_ns
        result["r"] = np.sqrt(result["s2_x"] ** 2 + result["s2_y"] ** 2)
        result["theta"] = np.arctan2(result["s2_y"], result["s2_x"])
        result["alt_s1_z"] = -result["alt_s1_interaction_drift_time"] * v_cm_per_ns
        result["alt_s2_z"] = -result["alt_s2_interaction_drift_time"] * v_cm_per_ns

        return result

    def _fill_events(self, result, split_peaks):
        for i, peaks_in_event in enumerate(split_peaks):
            n = len(peaks_in_event)
            result[i]["n_peaks"] = n
            if not n:
                raise ValueError(f"No peaks inside event {i}?")
            self._fill_result_i(result[i], peaks_in_event)

    def _fill_result_i(self, event, peaks):
        # Find S2s first so allow_posts2_s1s can constrain the S1 search.
        largest_s2s, s2_idx = self._get_largest_sx(peaks, s_i=2, number_of_peaks=0)

        if not self.allow_posts2_s1s and len(largest_s2s):
            s1_latest_time = largest_s2s[0]["time"]
        else:
            s1_latest_time = np.inf

        largest_s1s, s1_idx = self._get_largest_sx(
            peaks,
            s_i=1,
            s1_before_time=s1_latest_time,
            s1_min_coincidence=self.event_s1_min_coincidence,
        )

        if self.force_alt_s2_in_max_drift_time:
            s2_idx, largest_s2s = self._find_main_alt_s2(
                largest_s1s, s2_idx, largest_s2s, self.drift_time_max
            )
        else:
            largest_s2s, s2_idx = largest_s2s[:2], s2_idx[:2]

        if self.force_main_before_alt:
            order = strax.stable_argsort(largest_s2s["time"])
            largest_s2s, s2_idx = largest_s2s[order], s2_idx[order]

        self._set_sx_index(event, s1_idx, s2_idx)
        self._set_event_properties(event, largest_s1s, largest_s2s, peaks)

        for s_i, largest in enumerate([largest_s1s, largest_s2s], 1):
            for li, prefix in enumerate(["s", "alt_s"]):
                peak_fields = [name for name, _, _ in self._peak_properties]
                if s_i == 1:
                    peak_fields += ["max_diff", "min_diff", "first_channel", "last_channel"]
                elif s_i == 2:
                    peak_fields += ["x", "y"]
                ev_fields = [f"{prefix}{s_i}_{f}" for f in peak_fields]
                self._copy_peak_into_event(event, largest, li, ev_fields, peak_fields)

    @staticmethod
    def _get_largest_sx(
        peaks,
        s_i,
        s1_before_time=np.inf,
        s1_min_coincidence=0,
        number_of_peaks=2,
    ):
        mask = peaks["type"] == s_i
        if s_i == 1:
            mask &= peaks["time"] < s1_before_time
            mask &= peaks["tight_coincidence"] >= s1_min_coincidence
        selected = peaks[mask]
        idx = np.arange(len(peaks))[mask]
        order = strax.stable_argsort(selected["area"])[-number_of_peaks:][::-1]
        return selected[order], idx[order]

    @staticmethod
    @numba.njit
    def _find_main_alt_s2(largest_s1s, s2_idx, largest_s2s, drift_time_max):
        if len(largest_s1s) > 0 and len(largest_s2s) > 1:
            after_s1 = largest_s2s["center_time"] > largest_s1s[0]["center_time"]
            within_drift = (
                largest_s2s["center_time"] - largest_s1s[0]["center_time"]
            ) < 1.01 * drift_time_max
            mask = after_s1 & within_drift
            mask[0] = True  # always keep main S2
            s2_idx, largest_s2s = s2_idx[mask], largest_s2s[mask]
        return s2_idx[:2], largest_s2s[:2]

    @staticmethod
    @numba.njit
    def _set_event_properties(result, largest_s1s, largest_s2s, peaks):
        if len(largest_s1s) > 0 and len(largest_s2s) > 0:
            result["drift_time"] = largest_s2s[0]["center_time"] - largest_s1s[0]["center_time"]
            if len(largest_s1s) > 1:
                result["alt_s1_interaction_drift_time"] = (
                    largest_s2s[0]["center_time"] - largest_s1s[1]["center_time"]
                )
                result["alt_s1_delay"] = (
                    largest_s1s[1]["center_time"] - largest_s1s[0]["center_time"]
                )
            if len(largest_s2s) > 1:
                result["alt_s2_interaction_drift_time"] = (
                    largest_s2s[1]["center_time"] - largest_s1s[0]["center_time"]
                )
                result["alt_s2_delay"] = (
                    largest_s2s[1]["center_time"] - largest_s2s[0]["center_time"]
                )

        if len(largest_s2s) > 0:
            main_ct = largest_s2s[0]["center_time"]
            s2_mask = peaks["type"] == 2

            after_mask = s2_mask & (peaks["center_time"] > main_ct)
            before_mask = s2_mask & (peaks["center_time"] < main_ct)

            if np.any(after_mask):
                idxs = np.arange(len(peaks))[after_mask]
                best = idxs[np.argmax(peaks["area"][after_mask])]
                result["large_s2_after_main_s2_area"] = peaks["area"][best]
                result["large_s2_after_main_s2_index"] = best
                result["large_s2_after_main_s2_center_time"] = peaks["center_time"][best]

            if np.any(before_mask):
                idxs = np.arange(len(peaks))[before_mask]
                best = idxs[np.argmax(peaks["area"][before_mask])]
                result["large_s2_before_main_s2_area"] = peaks["area"][best]
                result["large_s2_before_main_s2_index"] = best
                result["large_s2_before_main_s2_center_time"] = peaks["center_time"][best]

            result["area_before_main_s2"] = np.sum(peaks["area"][before_mask])

        return result

    @staticmethod
    def _copy_peak_into_event(result, largest_sx, index, ev_fields, peak_fields):
        if index >= len(largest_sx):
            return
        for ev_f, pk_f in zip(ev_fields, peak_fields):
            result[ev_f] = largest_sx[index][pk_f]

    @staticmethod
    def _set_sx_index(res, s1_idx, s2_idx):
        if len(s1_idx):
            res["s1_index"] = s1_idx[0]
            if len(s1_idx) > 1:
                res["alt_s1_index"] = s1_idx[1]
        if len(s2_idx):
            res["s2_index"] = s2_idx[0]
            if len(s2_idx) > 1:
                res["alt_s2_index"] = s2_idx[1]
