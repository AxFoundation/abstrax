"""Peaklets: records -> peaklets + lone_hits.

Closely follows the straxen Peaklets plugin but adapted for abstrax:
- URLConfig fields live inside the class body (strax convention).
- Detector parameters (n_tpc_pmts, n_top_pmts) resolved from detector.yaml
  via the ``detector://`` protocol — no hardcoded channel counts.
- No channel_map / xedocs / MonogDB dependencies.
- gain_model and hit_min_amplitude must be set via context config.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numba
import numpy as np
import strax
from strax.dtypes import DIGITAL_SUM_WAVEFORM_CHANNEL
from strax.processing.general import _touching_windows


class Peaklets(strax.Plugin):
    """Split records into peaklets and lone_hits.

    Peaklets are aggressively split peaks so that nearby S1-S2 pairs can be
    resolved. Splitting uses the natural-breaks algorithm on the sum waveform.
    Lone hits are hits that fall outside every peaklet.

    Processing steps:
    1. Hit finding on records.
    2. Initial clustering by gap threshold -> proto-peaklets.
    3. Natural-breaks recursive splitting.
    4. Optional saturation correction on saturated channels.
    5. Tight-coincidence and per-hit timing features.
    """

    depends_on = "records"
    provides: Union[Tuple[str, ...], str] = ("peaklets", "lone_hits")
    data_kind: Union[Dict[str, str], str] = dict(peaklets="peaklets", lone_hits="lone_hits")
    parallel = "process"
    compressor = "zstd"
    rechunk_on_load = True
    chunk_source_size_mb = 100
    chunk_target_size_mb = 2000

    __version__ = "0.0.2"

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

    # --- Gain model -----------------------------------------------------------
    # No default: the context must supply this (e.g. "numpy://<path>").
    # abstrax_context() points this at a flat fake gain array in abstrax/resources/.

    gain_model = strax.URLConfig(
        infer_type=False,
        help=(
            "PMT gain model: 1-D array of PE/ADC conversion factors, "
            "one entry per channel.  Loaded via the numpy:// protocol."
        ),
    )

    # --- Hit finding ----------------------------------------------------------

    hit_min_amplitude = strax.URLConfig(
        default=15,
        infer_type=False,
        track=True,
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Scalar (applied to all channels) or array of length n_tpc_pmts."
        ),
    )

    # --- Clustering / peak finding -------------------------------------------

    peaklet_gap_threshold = strax.URLConfig(
        default=700,
        infer_type=False,
        help="Gap between hits (ns) that triggers a new peaklet.",
    )

    peak_left_extension = strax.URLConfig(
        default=30,
        infer_type=False,
        help="Samples (ns) to extend each peaklet left of the first hit.",
    )

    peak_right_extension = strax.URLConfig(
        default=200,
        infer_type=False,
        help="Samples (ns) to extend each peaklet right of the last hit.",
    )

    peak_min_pmts = strax.URLConfig(
        default=2,
        infer_type=False,
        help="Minimum number of contributing PMTs to define a peaklet.",
    )

    peaklet_max_duration = strax.URLConfig(
        default=int(10e6),
        infer_type=False,
        help="Maximum duration (ns) of a single peaklet.",
    )

    # --- Natural-breaks splitting --------------------------------------------

    peak_split_gof_threshold = strax.URLConfig(
        default=(None, ((0.5, 1.0), (6.0, 0.4)), ((2.5, 1.0), (5.625, 0.4))),
        infer_type=False,
        help=(
            "Natural-breaks goodness-of-fit threshold. "
            "Tuple of (reserved, S1-thresholds, S2-thresholds); each threshold "
            "element is a sequence of (log10(area), gof_threshold) pairs."
        ),
    )

    peak_split_filter_wing_width = strax.URLConfig(
        default=70,
        infer_type=False,
        help="Wing width of the moving-average filter for the low-split natural-breaks step.",
    )

    peak_split_min_area = strax.URLConfig(
        default=40.0,
        infer_type=False,
        help="Minimum area (PE) to evaluate the natural-breaks split criterion.",
    )

    peak_split_iterations = strax.URLConfig(
        default=20,
        infer_type=False,
        help="Maximum number of recursive peak-split iterations.",
    )

    # --- Saturation correction -----------------------------------------------

    saturation_correction_on = strax.URLConfig(
        default=True,
        infer_type=False,
        help="Apply saturation correction using non-saturated channels.",
    )

    saturation_reference_length = strax.URLConfig(
        default=100,
        infer_type=False,
        help="Maximum number of pre-saturation reference samples used for correction.",
    )

    saturation_min_reference_length = strax.URLConfig(
        default=20,
        infer_type=False,
        help="Minimum number of reference samples required to apply saturation correction.",
    )

    # --- Tight coincidence ---------------------------------------------------

    tight_coincidence_window_left = strax.URLConfig(
        default=50,
        infer_type=False,
        help="Left window (ns) around peak maximum to count tight-coincidence hits.",
    )

    tight_coincidence_window_right = strax.URLConfig(
        default=50,
        infer_type=False,
        help="Right window (ns) around peak maximum to count tight-coincidence hits.",
    )

    # --- Sum-waveform storage ------------------------------------------------

    store_data_top = strax.URLConfig(
        default=True,
        type=bool,
        help="Store the separate top-array sum waveform in each peaklet.",
    )

    store_data_start = strax.URLConfig(
        default=True,
        type=bool,
        help="Store the leading samples of the waveform at the minimum dt.",
    )

    # --- Diagnostics ---------------------------------------------------------

    diagnose_sorting = strax.URLConfig(
        track=False,
        default=False,
        infer_type=False,
        help="Enable runtime assertions for time ordering and disjointness.",
    )

    # -------------------------------------------------------------------------

    def infer_dtype(self):
        return dict(
            peaklets=strax.peak_dtype(
                n_channels=self.n_tpc_pmts,
                store_data_top=self.store_data_top,
                store_data_start=self.store_data_start,
            ),
            lone_hits=strax.hit_dtype,
        )

    def setup(self):
        if self.peak_min_pmts > 2:
            # Raising this breaks the lone_hit definition; see straxen/issues/295.
            raise NotImplementedError(
                f"peak_min_pmts={self.peak_min_pmts} interferes with lone_hit definition."
            )

        self.to_pe = self.gain_model
        self.hit_thresholds = self.hit_min_amplitude
        # TPC channel range for tight-coincidence (inclusive both ends).
        self.channel_range = (0, self.n_tpc_pmts - 1)

        if self.peaklet_gap_threshold > strax.DEFAULT_CHUNK_SPLIT_NS:
            raise ValueError(
                f"peaklet_gap_threshold {self.peaklet_gap_threshold} ns exceeds "
                f"strax safe_break_in_pulses {strax.DEFAULT_CHUNK_SPLIT_NS} ns."
            )

    def compute(self, records, start, end):
        hits = strax.find_hits(records, min_amplitude=self.hit_thresholds)

        # Zero-gain channels must not drive clustering.
        hits = hits[self.to_pe[hits["channel"]] != 0]
        hits = strax.sort_by_time(hits)

        # Initial clustering by gap threshold.
        peaklets = strax.find_peaks(
            hits,
            self.to_pe,
            gap_threshold=self.peaklet_gap_threshold,
            left_extension=self.peak_left_extension,
            right_extension=self.peak_right_extension,
            min_channels=self.peak_min_pmts,
            # Always build with data_top + data_start; drop fields later if not storing them.
            result_dtype=strax.peak_dtype(
                n_channels=self.n_tpc_pmts,
                store_data_top=True,
                store_data_start=True,
            ),
            max_duration=self.peaklet_max_duration,
        )

        self.clip_peaklet_times(peaklets, start, end)

        is_lone_hit = strax.fully_contained_in(hits, peaklets) == -1
        lone_hits = hits[is_lone_hit]
        strax.integrate_lone_hits(
            lone_hits,
            records,
            peaklets,
            save_outside_hits=(self.peak_left_extension, self.peak_right_extension),
            n_channels=len(self.to_pe),
        )
        if np.any(lone_hits["right_integration"] - lone_hits["left_integration"] <= 0):
            raise ValueError("Lone hits with non-positive integration length found.")

        # Restrict to in-peaklet hits for subsequent processing.
        hits = hits[~is_lone_hit]
        outside_peaks = self.create_outside_peaks_region(peaklets, start, end)
        strax.find_hit_integration_bounds(
            hits,
            outside_peaks,
            records,
            save_outside_hits=(self.peak_left_extension, self.peak_right_extension),
            n_channels=len(self.to_pe),
            allow_bounds_beyond_records=True,
        )

        # Extend hits into hitlets (include left/right extensions).
        hitlets = hits
        del hits
        hitlets["time"] -= (hitlets["left"] - hitlets["left_integration"]) * hitlets["dt"]
        hitlets["length"] = hitlets["right_integration"] - hitlets["left_integration"]
        hitlets = strax.sort_by_time(hitlets)
        self.clip_peaklet_times(hitlets, start, end)

        rlinks = strax.record_links(records)
        strax.sum_waveform(
            peaklets,
            hitlets,
            records,
            rlinks,
            self.to_pe,
            n_top_channels=self.n_top_pmts,
            store_data_top=self.store_data_top,
            store_data_start=self.store_data_start,
        )
        strax.compute_properties(peaklets, n_top_channels=self.n_top_pmts)

        # Natural-breaks recursive splitting.
        peaklets = strax.split_peaks(
            peaklets,
            hitlets,
            records,
            rlinks,
            self.to_pe,
            algorithm="natural_breaks",
            threshold=self.natural_breaks_threshold,
            split_low=True,
            filter_wing_width=self.peak_split_filter_wing_width,
            min_area=self.peak_split_min_area,
            do_iterations=self.peak_split_iterations,
            n_top_channels=self.n_top_pmts,
            store_data_top=self.store_data_top,
            store_data_start=self.store_data_start,
        )

        if self.saturation_correction_on:
            peak_list = peak_saturation_correction(
                records,
                rlinks,
                peaklets,
                hitlets,
                self.to_pe,
                reference_length=self.saturation_reference_length,
                min_reference_length=self.saturation_min_reference_length,
                n_top_channels=self.n_top_pmts,
                store_data_top=self.store_data_top,
                store_data_start=self.store_data_start,
            )
            strax.compute_properties(
                peaklets,
                n_top_channels=self.n_top_pmts,
                select_peaks_indices=peak_list,
            )

        # Tight coincidence: count distinct channels firing near the peak maximum.
        hit_max_times_argsort = strax.stable_argsort(hitlets["max_time"])
        sorted_hit_max_times = hitlets["max_time"][hit_max_times_argsort]
        sorted_hit_channels = hitlets["channel"][hit_max_times_argsort]
        peaklet_max_times = (
            peaklets["time"] + np.argmax(peaklets["data"], axis=1) * peaklets["dt"]
        )
        peaklets["tight_coincidence"] = get_tight_coin(
            sorted_hit_max_times,
            sorted_hit_channels,
            peaklet_max_times,
            self.tight_coincidence_window_left,
            self.tight_coincidence_window_right,
            self.channel_range,
        )

        self.add_hit_features(hitlets, peaklets)

        if self.diagnose_sorting and len(records):
            assert np.diff(records["time"]).min(initial=1) >= 0, "Records not sorted"
            assert np.diff(hitlets["time"]).min(initial=1) >= 0, "Hitlets not sorted"
            assert np.all(
                peaklets["time"][1:] >= strax.endtime(peaklets)[:-1]
            ), "Peaklets not disjoint"

        counts = strax.touching_windows(hitlets, peaklets)
        counts = np.diff(counts, axis=1).flatten()
        peaklets["n_hits"] = counts

        if not (self.store_data_top and self.store_data_start):
            peaklets = _drop_data_fields(peaklets, self.dtype_for("peaklets"))

        if not np.all(peaklets["n_hits"] >= peaklets["tight_coincidence"]):
            raise ValueError("Found n_hits < tight_coincidence in peaklets.")

        return dict(peaklets=peaklets, lone_hits=lone_hits)

    def natural_breaks_threshold(self, peaks):
        """Per-peak GOF threshold interpolated between S1-like and S2-like curves."""
        rise_time = -peaks["area_decile_from_midpoint"][:, 1]
        f_s2 = 8 * np.log10(rise_time.clip(1, 1e5) / 100)
        f_s2 = 1 / (1 + np.exp(-f_s2))
        log_area = np.log10(peaks["area"].clip(1, 1e7))
        thresholds = self.peak_split_gof_threshold
        return f_s2 * np.interp(log_area, *np.transpose(thresholds[2])) + (
            1 - f_s2
        ) * np.interp(log_area, *np.transpose(thresholds[1]))

    @staticmethod
    @numba.njit(nogil=True, cache=True)
    def clip_peaklet_times(peaklets, start, end):
        for p in peaklets:
            if p["time"] < start:
                p["time"] = start
            if strax.endtime(p) > end:
                p["length"] = (end - p["time"]) // p["dt"]

    @staticmethod
    def create_outside_peaks_region(peaklets, start, end):
        """Return time intervals that lie outside all peaklets."""
        if not len(peaklets):
            return np.zeros(0, dtype=strax.time_fields)
        outside = np.zeros(len(peaklets) + 1, dtype=strax.time_fields)
        outside[0]["time"] = start
        outside[0]["endtime"] = peaklets[0]["time"]
        outside[1:-1]["time"] = strax.endtime(peaklets[:-1])
        outside[1:-1]["endtime"] = peaklets["time"][1:]
        outside[-1]["time"] = strax.endtime(peaklets[-1])
        outside[-1]["endtime"] = end
        return outside

    @staticmethod
    def add_hit_features(hitlets, peaklets):
        """Fill per-peaklet hit-timing features (max_diff, min_diff, first/last channel)."""
        peaklets["max_diff"] = -1
        peaklets["min_diff"] = -1
        peaklets["first_channel"] = DIGITAL_SUM_WAVEFORM_CHANNEL
        peaklets["last_channel"] = DIGITAL_SUM_WAVEFORM_CHANNEL
        for peaklet, _hitlets in zip(peaklets, strax.split_by_containment(hitlets, peaklets)):
            if not len(_hitlets):
                continue
            sorted_hitlets = _hitlets[strax.stable_argsort(_hitlets["max_time"])]
            time_diff = np.diff(sorted_hitlets["max_time"])
            if len(time_diff):
                peaklet["max_diff"] = time_diff.max()
                peaklet["min_diff"] = time_diff.min()
            peaklet["first_channel"] = sorted_hitlets[0]["channel"]
            peaklet["last_channel"] = sorted_hitlets[-1]["channel"]


# ---------------------------------------------------------------------------
# Module-level helpers (numba JIT functions must live at module scope)
# ---------------------------------------------------------------------------


def _drop_data_fields(peaklets, goal_dtype):
    """Return peaklets cast to goal_dtype, dropping any data_top / data_start fields."""
    out = np.zeros(len(peaklets), dtype=goal_dtype)
    strax.copy_to_buffer(peaklets, out, "_drop_data_fields")
    return out


@numba.jit(nopython=True, nogil=True, cache=False)
def peak_saturation_correction(
    records,
    rlinks,
    peaks,
    hitlets,
    to_pe,
    reference_length=100,
    min_reference_length=20,
    use_classification=False,
    n_top_channels=0,
    store_data_top=False,
    store_data_start=False,
):
    """Correct area and per-PMT area of peaks that contain saturated channels.

    For each saturated channel the unsaturated portion of the sum waveform is
    used as a reference to scale the saturated channel's pulse, recovering an
    approximate waveform.  Adapted from straxen / pax.
    """
    if not len(records):
        return
    if not len(peaks):
        return

    mask = peaks["n_saturated_channels"] > 0
    if use_classification:
        mask &= peaks["type"] == 2
    peak_list = np.where(mask)[0]

    record_ranges = _touching_windows(
        records["time"],
        strax.endtime(records),
        peaks[peak_list]["time"],
        strax.endtime(peaks[peak_list]),
    )

    dt = records[0]["dt"]
    n_channels = len(peaks[0]["saturated_channel"])
    len_buffer = np.max(peaks["length"] * peaks["dt"]) // dt + 1
    max_nrecord = len_buffer // len(records[0]["data"]) + 1

    b_sumwf = np.zeros(len_buffer, dtype=np.float32)
    b_pulse = np.zeros((n_channels, len_buffer), dtype=np.int16)
    b_index = np.zeros((n_channels, max_nrecord), dtype=np.int64)

    for ix, peak_i in enumerate(peak_list):
        b_sumwf[:] = 0
        b_pulse[:] = 0
        b_index[:] = -1

        p = peaks[peak_i]
        channel_saturated = p["saturated_channel"] > 0

        for record_i in range(record_ranges[ix][0], record_ranges[ix][1]):
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r["time"] // dt, r["length"], p["time"] // dt, p["length"] * p["dt"] // dt
            )
            ch = r["channel"]
            if channel_saturated[ch]:
                b_pulse[ch, slice(*b_slice)] += r["data"][slice(*r_slice)]
                b_index[ch, np.argmin(b_index[ch])] = record_i
            else:
                b_sumwf[slice(*b_slice)] += r["data"][slice(*r_slice)] * to_pe[ch]

        _peak_saturation_correction_inner(
            channel_saturated,
            records,
            p,
            to_pe,
            b_sumwf,
            b_pulse,
            b_index,
            reference_length,
            min_reference_length,
        )

        peaks[peak_i]["length"] = p["length"] * p["dt"] / dt
        peaks[peak_i]["dt"] = dt

    strax.sum_waveform(
        peaks,
        hitlets,
        records,
        rlinks,
        to_pe,
        n_top_channels=n_top_channels,
        store_data_top=store_data_top,
        store_data_start=store_data_start,
        select_peaks_indices=peak_list,
    )
    return peak_list


@numba.jit(nopython=True, nogil=True, cache=True)
def _peak_saturation_correction_inner(
    channel_saturated,
    records,
    p,
    to_pe,
    b_sumwf,
    b_pulse,
    b_index,
    reference_length=100,
    min_reference_length=20,
):
    dt = records["dt"][0]
    n_channels = len(channel_saturated)

    for ch in range(n_channels):
        if not channel_saturated[ch]:
            continue
        b = b_pulse[ch]

        bl = np.inf
        for record_i in b_index[ch]:
            if record_i == -1:
                break
            bl = min(bl, records["baseline"][record_i])

        s0 = np.argmax(b >= np.int16(bl))
        ref = slice(max(0, s0 - reference_length), s0)

        if (b[ref] * to_pe[ch] > 1).sum() < min_reference_length:
            continue
        if (b_sumwf[ref] > 1).sum() < min_reference_length:
            continue
        if np.sum(b[ref]) * to_pe[ch] / np.sum(b_sumwf[ref]) > 1:
            continue

        scale = np.sum(b[ref]) / np.sum(b_sumwf[ref])

        for record_i in b_index[ch]:
            if record_i == -1:
                break
            r = records[record_i]
            r_slice, b_slice = strax.overlap_indices(
                r["time"] // dt,
                r["length"],
                p["time"] // dt + s0,
                p["length"] * p["dt"] // dt - s0,
            )
            if r_slice[1] == r_slice[0]:
                continue
            b_slice = b_slice[0] + s0, b_slice[1] + s0

            apax = scale * max(b_sumwf[slice(*b_slice)])
            if np.int32(apax) >= 2**15:
                bshift = int(np.floor(np.log2(apax) - 14))
                tmp = r["data"].astype(np.int32)
                tmp[slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale
                r["area"] = np.sum(tmp)
                r["data"][:] = np.right_shift(tmp, bshift)
                r["amplitude_bit_shift"] += bshift
            else:
                r["data"][slice(*r_slice)] = b_sumwf[slice(*b_slice)] * scale
                r["area"] = np.sum(r["data"])


@numba.jit(nopython=True, nogil=True, cache=True)
def get_tight_coin(hit_max_times, hit_channel, peak_max_times, left, right, channels=(0, 1999)):
    """Count distinct PMT channels firing within [peak_max - left, peak_max + right] ns.

    :param hit_max_times: Time of each hit's amplitude maximum (ns), sorted.
    :param hit_channel: PMT channel of each hit (same order as hit_max_times).
    :param peak_max_times: Time of each peaklet's amplitude maximum (ns).
    :param left: Left boundary of the coincidence window (ns).
    :param right: Right boundary of the coincidence window (ns).
    :param channels: (start_ch, end_ch) inclusive channel range for the TPC.
    :return: Array of length len(peak_max_times) with tight-coincidence counts.
    """
    left_hit_i = 0
    n_coin_channel = np.zeros(len(peak_max_times), dtype=np.int16)
    start_ch, end_ch = channels
    channels_seen = np.zeros(end_ch - start_ch + 1, dtype=np.bool_)

    for p_i, p_t in enumerate(peak_max_times):
        channels_seen[:] = 0
        for left_hit_i in range(left_hit_i, len(hit_max_times)):
            d = hit_max_times[left_hit_i] - p_t
            if (-left <= d) & (d <= right):
                channels_seen[hit_channel[left_hit_i] - start_ch] = 1
            if d > right:
                n_coin_channel[p_i] = np.sum(channels_seen)
                break
        n_coin_channel[p_i] = np.sum(channels_seen)

    return n_coin_channel
