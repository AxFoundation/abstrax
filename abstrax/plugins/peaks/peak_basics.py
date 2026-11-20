"""PeakBasics: peaks -> peak_basics.

Extracts scalar summary properties from each peak, dropping structured arrays
so the output can be loaded as a flat DataFrame.
"""

from __future__ import annotations

import numpy as np
import strax


class PeakBasics(strax.Plugin):
    """Compute scalar summary properties for each peak.

    All structured-array fields (waveform data, per-channel areas, width
    arrays) are reduced to scalars so the output is pandas-friendly.
    """

    __version__ = "0.0.1"

    depends_on = "peaks"
    provides = "peak_basics"
    data_kind = "peaks"
    parallel = True
    compressor = "zstd"

    n_top_pmts = strax.URLConfig(
        default="detector://n_top_pmts",
        type=int,
        track=True,
        help="Number of top-array PMT channels (channels 0 … n_top_pmts-1).",
    )

    check_peak_sum_area_rtol = strax.URLConfig(
        default=None,
        track=False,
        infer_type=False,
        help=(
            "If set, assert that area_per_channel.sum() matches area within "
            "this relative tolerance (e.g. '1e-4').  None disables the check."
        ),
    )

    def infer_dtype(self):
        return strax.time_fields + [
            (("Weighted average center time of the peak [ns]", "center_time"), np.int64),
            (("Peak integral in PE", "area"), np.float32),
            (("Number of hits contributing at least one sample to the peak", "n_hits"), np.int32),
            (("Number of PMTs contributing to the peak", "n_channels"), np.int16),
            (("Number of top PMTs contributing to the peak", "top_n_channels"), np.int16),
            (("PMT number which contributes the most PE", "max_pmt"), np.int16),
            (("Area of signal in the largest-contributing PMT (PE)", "max_pmt_area"), np.float32),
            (("Total number of saturated channels", "n_saturated_channels"), np.int16),
            (("Width (in ns) of the central 50% area of the peak", "range_50p_area"), np.float32),
            (("Width (in ns) of the central 90% area of the peak", "range_90p_area"), np.float32),
            (("Fraction of area seen by the top array", "area_fraction_top"), np.float32),
            (("Length of the peak waveform in samples", "length"), np.int32),
            (("Time resolution of the peak waveform in ns", "dt"), np.int16),
            (("Weighted relative median time of the peak [ns]", "median_time"), np.float32),
            (("Time between 10% and 50% area quantiles [ns]", "rise_time"), np.float32),
            (("Number of PMTs with hits within tight range of mean", "tight_coincidence"), np.int16),
            (("Classification of the peak(let)", "type"), np.int8),
            (("Largest time difference between apexes of hits inside peak [ns]", "max_diff"), np.int32),
            (("Smallest time difference between apexes of hits inside peak [ns]", "min_diff"), np.int32),
            (("First channel/PMT number inside peak (sorted by apexes of hits)", "first_channel"), np.int16),
            (("Last channel/PMT number inside peak (sorted by apexes of hits)", "last_channel"), np.int16),
        ]

    def compute(self, peaks):
        r = np.zeros(len(peaks), self.dtype)
        if not len(peaks):
            return r

        p = peaks

        # Direct copies from peaks structured array.
        for field in (
            "time",
            "center_time",
            "length",
            "dt",
            "median_time",
            "area",
            "area_fraction_top",
            "type",
            "max_diff",
            "min_diff",
            "first_channel",
            "last_channel",
        ):
            r[field] = p[field]

        r["endtime"] = p["time"] + p["dt"] * p["length"]
        r["n_hits"] = p["n_hits"]
        r["tight_coincidence"] = p["tight_coincidence"]
        r["n_saturated_channels"] = p["n_saturated_channels"]

        r["n_channels"] = (p["area_per_channel"] > 0).sum(axis=1)
        r["top_n_channels"] = (p["area_per_channel"][:, : self.n_top_pmts] > 0).sum(axis=1)

        r["max_pmt"] = np.argmax(p["area_per_channel"], axis=1)
        r["max_pmt_area"] = np.max(p["area_per_channel"], axis=1)

        r["range_50p_area"] = p["width"][:, 5]
        r["range_90p_area"] = p["width"][:, 9]

        r["rise_time"] = -p["area_decile_from_midpoint"][:, 1]

        if self.check_peak_sum_area_rtol is not None:
            self.check_area(p["area_per_channel"].sum(axis=1), p, self.check_peak_sum_area_rtol)

        return r

    @staticmethod
    def check_area(area_per_channel_sum, peaks, rtol) -> None:
        """Raise ValueError if area_per_channel sum disagrees with peak area beyond rtol."""
        positive_area = peaks["area"] > 0
        if not np.sum(positive_area):
            return

        is_close = np.isclose(
            area_per_channel_sum[positive_area],
            peaks[positive_area]["area"],
            rtol=rtol,
        )
        if not is_close.all():
            for peak in peaks[positive_area][~is_close]:
                strax.print_record(peak)
            p_i = np.where(~is_close)[0][0]
            peak = peaks[positive_area][p_i]
            off = 1 - area_per_channel_sum[positive_area][p_i] / peak["area"]
            raise ValueError(
                f"Area not calculated correctly, it's {100 * off:.2f}% off, time: {peak['time']}"
            )
