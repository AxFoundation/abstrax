"""MergedS2s: peaklets + peaklet_classification + lone_hits -> merged_s2s.

Merges adjacent S2 peaklets into combined S2 peaks using a gap-threshold
algorithm that is sensitive to the merged area.  Lone hits are added to the
merged waveforms after grouping.

Algorithm (ported from straxen v2.2.7, MergedS2s):
1. Optionally exclude S1-classified peaklets from merging.
2. For each pair of adjacent peaklets, sort gaps by size (smallest first).
3. Merge neighbours if the resulting peak satisfies all of:
   - gap < area-dependent threshold (linearly interpolated from
     s2_merge_gap_thresholds)
   - merged area < max_area (= 10^last_threshold_log_area)
   - merged duration < s2_merge_max_duration
4. After grouping, drop any merge group that does not start with an S2.
5. Lone hits are time-shifted to their integration windows and added to the
   merged waveforms via strax.add_lone_hits.
6. Recompute the peak properties (widths, median and center time, area
   fraction top) of the merged objects with strax.compute_properties.
   (strax.compute_widths returns the widths instead of storing them, so it
   alone left these fields at 0.)
"""

from __future__ import annotations

from typing import Tuple

import numba
import numpy as np
import strax


class MergedS2s(strax.OverlapWindowPlugin):
    """Merge adjacent S2 peaklets into combined merged_s2s peaks."""

    __version__ = "0.0.2"

    depends_on: Tuple[str, ...] = ("peaklets", "peaklet_classification", "lone_hits")
    provides = "merged_s2s"
    data_kind = "merged_s2s"

    # --- Merging parameters --------------------------------------------------

    s2_merge_max_duration = strax.URLConfig(
        default=50_000,
        infer_type=False,
        help="Do not merge peaklets if the result would exceed this duration [ns].",
    )

    s2_merge_gap_thresholds = strax.URLConfig(
        default=((1.7, 2.65e4), (4.0, 2.6e3), (5.0, 0.0)),
        infer_type=False,
        help=(
            "Fix points for area-dependent maximum gap [ns].  Format: "
            "((log10(area), max_gap), ...).  First point gives the overall "
            "maximum gap; last point gives the maximum area to merge."
        ),
    )

    merge_without_s1 = strax.URLConfig(
        default=True,
        infer_type=False,
        help=(
            "If True, exclude S1-classified peaklets from the merge step.  "
            "An S1 may then appear inside a merged S2."
        ),
    )

    merged_s2s_get_window_size_factor = strax.URLConfig(
        default=5,
        type=int,
        track=False,
        help="Multiplicative factor for the OverlapWindowPlugin window size.",
    )

    # --- Detector geometry (single source of truth: detector.yaml) -----------

    n_tpc_pmts = strax.URLConfig(
        default="detector://n_tpc_pmts",
        type=int,
        track=True,
        help="Total number of TPC PMT channels.",
    )

    n_top_pmts = strax.URLConfig(
        default="detector://n_top_pmts",
        type=int,
        track=True,
        help="Number of top-array PMT channels (channels 0 … n_top_pmts-1).",
    )

    # --- Gain model ----------------------------------------------------------
    # No default: the context must supply this (e.g. "pkg://<file in abstrax/resources>").

    gain_model = strax.URLConfig(
        infer_type=False,
        help=(
            "PMT gain model: 1-D array of PE/ADC conversion factors, "
            "one entry per channel, e.g. loaded via the pkg:// protocol."
        ),
    )

    # -------------------------------------------------------------------------

    def setup(self):
        self.to_pe = self.gain_model

    def infer_dtype(self):
        peaklet_classification_dtype = self.deps["peaklet_classification"].dtype_for(
            "peaklet_classification"
        )
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        # Order matters: classification dtype first, then peaklets dtype.
        return strax.merged_dtype((peaklet_classification_dtype, peaklets_dtype))

    def get_window_size(self):
        return self.merged_s2s_get_window_size_factor * (
            int(self.s2_merge_gap_thresholds[0][1]) + self.s2_merge_max_duration
        )

    def compute(self, peaklets, lone_hits):
        if self.merge_without_s1:
            peaklets = peaklets[peaklets["type"] != 1]

        if len(peaklets) <= 1:
            return np.zeros(0, dtype=self.dtype)

        gap_thresholds = self.s2_merge_gap_thresholds
        max_gap = gap_thresholds[0][1]
        max_area = 10 ** gap_thresholds[-1][0]

        if max_gap < 0:
            return np.zeros(0, dtype=self.dtype)

        # strax.merge_peaks (via store_downsampled_waveform) always accesses
        # both data_top and data_start regardless of configuration.  Expand
        # the peaklets array to include both fields if either is absent.
        needs_fields = (
            "data_top" not in peaklets.dtype.names
            or "data_start" not in peaklets.dtype.names
        )
        if needs_fields:
            expanded = np.zeros(
                len(peaklets),
                dtype=strax.peak_dtype(
                    n_channels=self.n_tpc_pmts,
                    store_data_top=True,
                    store_data_start=True,
                ),
            )
            strax.copy_to_buffer(peaklets, expanded, "_add_missing_wf_fields")
            peaklets = expanded

        start_merge_at, end_merge_at = self.get_merge_instructions(
            peaklets["time"],
            strax.endtime(peaklets),
            areas=peaklets["area"],
            types=peaklets["type"],
            gap_thresholds=gap_thresholds,
            max_duration=self.s2_merge_max_duration,
            max_gap=max_gap,
            max_area=max_area,
        )

        merged_s2s = strax.merge_peaks(
            peaklets,
            start_merge_at,
            end_merge_at,
            max_buffer=int(self.s2_merge_max_duration // np.gcd.reduce(peaklets["dt"])),
        )
        merged_s2s["type"] = 2

        # Shift lone hits to their integration windows before adding them.
        lh = np.copy(lone_hits)
        del lone_hits
        lh_time_shift = (lh["left"] - lh["left_integration"]) * lh["dt"]
        lh["time"] = lh["time"] - lh_time_shift
        lh["length"] = lh["right_integration"] - lh["left_integration"]
        lh = strax.sort_by_time(lh)

        n_top_pmts_if_digitize_top = self.n_top_pmts if "data_top" in self.dtype.names else -1
        strax.add_lone_hits(merged_s2s, lh, self.to_pe, n_top_channels=n_top_pmts_if_digitize_top)

        if not len(merged_s2s):
            return merged_s2s

        strax.compute_properties(merged_s2s, n_top_channels=self.n_top_pmts)

        # Cast back to self.dtype to drop any fields that were added solely for
        # strax.merge_peaks (data_top, data_start) and are not in the output dtype.
        if merged_s2s.dtype != self.dtype:
            merged_s2s = _cast_to_dtype(merged_s2s, self.dtype)

        return merged_s2s

    @staticmethod
    @numba.njit(cache=True, nogil=True)
    def get_merge_instructions(
        peaklet_starts,
        peaklet_ends,
        areas,
        types,
        gap_thresholds,
        max_duration,
        max_gap,
        max_area,
        sort_kind="mergesort",
    ):
        """Determine which groups of peaklets to merge.

        Iterates over inter-peaklet gaps from smallest to largest.  Two
        neighbours are merged when the combined peak satisfies the area-
        dependent gap threshold, the area limit, and the duration limit.

        Returns (start_merge_at, end_merge_at) — exclusive end indices ready
        for strax.merge_peaks.
        """
        peaklet_gaps = peaklet_starts[1:] - peaklet_ends[:-1]
        peaklet_start_index = np.arange(len(peaklet_starts))
        peaklet_end_index = np.arange(len(peaklet_starts))

        for gap_i in np.argsort(peaklet_gaps, kind=sort_kind):
            start_idx = peaklet_start_index[gap_i]
            inclusive_end_idx = peaklet_end_index[gap_i + 1]

            sum_area = np.sum(areas[start_idx : inclusive_end_idx + 1])
            this_gap = peaklet_gaps[gap_i]

            if inclusive_end_idx < start_idx:
                raise ValueError("Merge index invariant violated: left > right.")

            if this_gap > max_gap:
                break

            if sum_area > max_area:
                continue

            if (sum_area > 0) and (
                this_gap > _merge_s2_threshold(np.log10(sum_area), gap_thresholds)
            ):
                continue

            peak_duration = peaklet_ends[inclusive_end_idx] - peaklet_starts[start_idx]
            if peak_duration >= max_duration:
                continue

            peaklet_start_index[start_idx : inclusive_end_idx + 1] = peaklet_start_index[
                start_idx
            ]
            peaklet_end_index[start_idx : inclusive_end_idx + 1] = peaklet_end_index[
                inclusive_end_idx
            ]

        start_merge_at = np.unique(peaklet_start_index)
        end_merge_at = np.unique(peaklet_end_index)

        if len(start_merge_at) != len(end_merge_at):
            raise ValueError("Inconsistent start/end merge instructions.")

        return _filter_s1_starts(start_merge_at, types, end_merge_at)


# ---------------------------------------------------------------------------
# Module-level helpers (numba JIT functions must live at module scope)
# ---------------------------------------------------------------------------


@numba.njit(cache=True, nogil=True)
def _filter_s1_starts(start_merge_at, types, end_merge_at):
    """Advance each group's start past any leading S1s, then drop groups that
    do not start with an S2 or span fewer than two peaklets."""
    for i in range(len(start_merge_at)):
        while types[start_merge_at[i]] != 2:
            if end_merge_at[i] - start_merge_at[i] <= 1:
                break
            start_merge_at[i] += 1

    starts_on_s2 = types[start_merge_at] == 2
    merges_two_or_more = end_merge_at - start_merge_at >= 1
    keep = starts_on_s2 & merges_two_or_more
    return start_merge_at[keep], end_merge_at[keep] + 1


@numba.njit(cache=True, nogil=True)
def _merge_s2_threshold(log_area, gap_thresholds):
    """Linearly interpolate the maximum allowed gap for a given log10(area).

    :param log_area: log10 of the merged S2 area [PE].
    :param gap_thresholds: tuple of (log10_area, max_gap) fix points.
    """
    for i, (a1, g1) in enumerate(gap_thresholds):
        if log_area < a1:
            if i == 0:
                return g1
            a0, g0 = gap_thresholds[i - 1]
            return (log_area - a0) * (g1 - g0) / (a1 - a0) + g0
    return gap_thresholds[-1][1]


def _cast_to_dtype(peaks, goal_dtype):
    """Return peaks cast to goal_dtype via strax.copy_to_buffer."""
    out = np.zeros(len(peaks), dtype=goal_dtype)
    strax.copy_to_buffer(peaks, out, "_cast_to_dtype")
    return out
