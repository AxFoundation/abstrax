"""Peaks: peaklets + peaklet_classification + merged_s2s -> peaks.

Replaces S2 peaklets that were merged by MergedS2s with the merged objects,
while keeping all S1 and unclassified peaklets intact.

This step is computationally trivial and should not be cached; set
save_when = strax.SaveWhen.EXPLICIT so strax only writes it on demand.
"""

from __future__ import annotations

import numpy as np
import strax


class Peaks(strax.Plugin):
    """Combine peaklets and merged_s2s into the final peaks array.

    For each group of S2 peaklets that MergedS2s decided to merge, the
    individual peaklets are replaced by the single merged object.  S1 and
    unclassified peaklets pass through unchanged (if merge_without_s1 is
    True, S1s are handled separately so they are never consumed by
    strax.replace_merged).
    """

    __version__ = "0.0.1"

    depends_on = ("peaklets", "peaklet_classification", "merged_s2s")
    provides = "peaks"
    data_kind = "peaks"
    parallel = True
    compressor = "zstd"
    save_when = strax.SaveWhen.EXPLICIT

    merge_without_s1 = strax.URLConfig(
        default=True,
        infer_type=False,
        help=(
            "If True, S1 peaklets are excluded from the replace_merged step "
            "and concatenated back afterwards.  Must match the same setting "
            "in MergedS2s."
        ),
    )

    diagnose_sorting = strax.URLConfig(
        track=False,
        default=False,
        infer_type=False,
        help="Enable runtime assertions for time ordering and disjointness.",
    )

    def infer_dtype(self):
        return self.deps["peaklets"].dtype_for("peaklets")

    def compute(self, peaklets, merged_s2s):
        if self.merge_without_s1:
            is_s1 = peaklets["type"] == 1
            peaks = strax.replace_merged(peaklets[~is_s1], merged_s2s)
            peaks = strax.sort_by_time(np.concatenate([peaklets[is_s1], peaks]))
        else:
            peaks = strax.replace_merged(peaklets, merged_s2s)

        if self.diagnose_sorting:
            assert np.all(np.diff(peaks["time"]) >= 0), "Peaks not sorted"
            non_s1 = peaks["type"] != 1
            assert np.all(
                peaks["time"][non_s1][1:] >= strax.endtime(peaks)[non_s1][:-1]
            ), "Non-S1 peaks not disjoint"

        result = np.zeros(len(peaks), self.dtype)
        strax.copy_to_buffer(peaks, result, "_copy_peak_fields")
        return result
