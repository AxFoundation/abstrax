"""Pulse processing: raw_records -> records + pulse_counts.

Converts raw digitizer waveforms into baseline-subtracted, hit-found pulse
records and per-channel pulse-count statistics. All detector parameters come
from ``abstrax/config/detector.yaml`` via the ``detector://`` URLConfig
protocol.

Implementation notes
--------------------
- Follows the straxen PulseProcessing pattern, minus the HE veto.
- URLConfig fields live inside the class body (strax convention).
- Channel count is read from config; no hardcoded PMT numbers.
- Bump __version__ whenever the output dtype OR the numeric output changes.
"""

from __future__ import annotations

from typing import Tuple

import numba
import numpy as np
import strax
from immutabledict import immutabledict


class PulseProcessing(strax.Plugin):
    """raw_records -> records + pulse_counts.

    1. Baseline-subtract and integrate the waveform.
    2. Count pulses per channel.
    3. Find hits; zero samples outside hits.
    """

    __version__ = "0.1.0"

    parallel = "process"
    rechunk_on_save = immutabledict(records=False, pulse_counts=True)
    compressor = "zstd"
    depends_on = "raw_records"
    provides: Tuple[str, ...] = ("records", "pulse_counts")
    data_kind = {k: k for k in provides}
    save_when = immutabledict(
        records=strax.SaveWhen.TARGET,
        pulse_counts=strax.SaveWhen.ALWAYS,
    )

    n_tpc_pmts = strax.URLConfig(
        default="detector://n_tpc_pmts",
        type=int,
        track=True,
        help="Total number of TPC PMT channels.",
    )

    baseline_samples = strax.URLConfig(
        default=40,
        infer_type=False,
        help="Number of samples at the start of each pulse used to estimate the baseline.",
    )

    hit_min_amplitude = strax.URLConfig(
        default=15,
        infer_type=False,
        track=True,
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Scalar (applied to all channels) or array of length n_tpc_pmts."
        ),
    )

    save_outside_hits = strax.URLConfig(
        default=(3, 20),
        infer_type=False,
        help="(left, right) samples to preserve on each side of a hit; the rest is zeroed.",
    )

    check_raw_record_overlaps = strax.URLConfig(
        default=True,
        track=False,
        infer_type=False,
        help="Raise if any two pulses in the same channel overlap.",
    )

    allow_sloppy_chunking = strax.URLConfig(
        default=False,
        track=False,
        infer_type=False,
        help=(
            "Accept incorrectly chunked pulse fragments by using a default baseline. "
            "Enable only for data with known chunking issues."
        ),
    )

    def infer_dtype(self):
        self.record_length = strax.record_length_from_dtype(
            self.deps["raw_records"].dtype_for("raw_records")
        )
        return {
            "records": strax.record_dtype(self.record_length),
            "pulse_counts": pulse_count_dtype(self.n_tpc_pmts),
        }

    def compute(self, raw_records, start, end):
        if self.check_raw_record_overlaps:
            check_overlaps(raw_records, n_channels=self.n_tpc_pmts)

        # Drop any out-of-range channels.
        raw_records = raw_records[raw_records["channel"] < self.n_tpc_pmts]

        r = strax.raw_to_records(raw_records)
        del raw_records

        strax.zero_out_of_bounds(r)
        strax.baseline(
            r,
            baseline_samples=self.baseline_samples,
            allow_sloppy_chunking=self.allow_sloppy_chunking,
            flip=True,
        )
        strax.integrate(r)

        pulse_counts = count_pulses(r, self.n_tpc_pmts)
        pulse_counts["time"] = start
        pulse_counts["endtime"] = end

        if len(r):
            hits = strax.find_hits(r, min_amplitude=self.hit_min_amplitude)

            le, re = self.save_outside_hits
            r = strax.cut_outside_hits(r, hits, left_extension=le, right_extension=re)
            strax.zero_out_of_bounds(r)

        return dict(records=r, pulse_counts=pulse_counts)


# ---------------------------------------------------------------------------
# Pulse counting
# ---------------------------------------------------------------------------

NO_PULSE_COUNTS = -9999  # sentinel for channels with zero pulses in the chunk


def pulse_count_dtype(n_channels):
    return [
        (("Start time of the chunk", "time"), np.int64),
        (("End time of the chunk", "endtime"), np.int64),
        (("Number of pulses", "pulse_count"), (np.int64, n_channels)),
        (("Number of lone pulses", "lone_pulse_count"), (np.int64, n_channels)),
        (("Integral of all pulses in ADC_count x samples", "pulse_area"), (np.int64, n_channels)),
        (
            ("Integral of lone pulses in ADC_count x samples", "lone_pulse_area"),
            (np.int64, n_channels),
        ),
        (("Average baseline", "baseline_mean"), (np.int16, n_channels)),
        (("Average baseline rms", "baseline_rms_mean"), (np.float32, n_channels)),
    ]


def count_pulses(records, n_channels):
    """Return single-element array with per-channel pulse statistics."""
    if len(records):
        result = np.zeros(1, dtype=pulse_count_dtype(n_channels))
        _count_pulses(records, n_channels, result)
        return result
    return np.zeros(0, dtype=pulse_count_dtype(n_channels))


@numba.njit(cache=True, nogil=True)
def _count_pulses(records, n_channels, result):
    count = np.zeros(n_channels, dtype=np.int64)
    lone_count = np.zeros(n_channels, dtype=np.int64)
    area = np.zeros(n_channels, dtype=np.int64)
    lone_area = np.zeros(n_channels, dtype=np.int64)

    last_end_seen = 0
    next_start = 0
    in_lone_pulse = np.zeros(n_channels, dtype=np.bool_)
    baseline_buffer = np.zeros(n_channels, dtype=np.float64)
    baseline_rms_buffer = np.zeros(n_channels, dtype=np.float64)

    for r_i, r in enumerate(records):
        if r_i != len(records) - 1:
            next_start = records[r_i + 1]["time"]

        ch = r["channel"]
        if ch >= n_channels:
            print("Channel:", ch)
            raise RuntimeError("Out of bounds channel in count_pulses!")

        area[ch] += r["area"]

        if r["record_i"] == 0:
            count[ch] += 1
            baseline_buffer[ch] += r["baseline"]
            baseline_rms_buffer[ch] += r["baseline_rms"]

            if r["time"] > last_end_seen and r["time"] + r["pulse_length"] * r["dt"] < next_start:
                lone_count[ch] += 1
                in_lone_pulse[ch] = True
                lone_area[ch] += r["area"]
            else:
                in_lone_pulse[ch] = False

            last_end_seen = max(last_end_seen, r["time"] + r["pulse_length"] * r["dt"])

        elif in_lone_pulse[ch]:
            lone_area[ch] += r["area"]

    res = result[0]
    res["pulse_count"][:] = count[:]
    res["lone_pulse_count"][:] = lone_count[:]
    res["pulse_area"][:] = area[:]
    res["lone_pulse_area"][:] = lone_area[:]

    means = baseline_buffer / count
    means[np.isnan(means)] = NO_PULSE_COUNTS
    res["baseline_mean"][:] = means[:]
    res["baseline_rms_mean"][:] = (baseline_rms_buffer / count)[:]


# ---------------------------------------------------------------------------
# Overlap checking
# ---------------------------------------------------------------------------


def check_overlaps(records, n_channels):
    """Raise ValueError if any two records in the same channel overlap."""
    last_end = np.zeros(n_channels, dtype=np.int64)
    channel, time = _check_overlaps(records, last_end)
    if channel != -9999:
        raise ValueError(
            f"Bad data! In channel {channel}, a pulse starts at {time}, "
            "BEFORE the previous pulse in that same channel ended "
            f"(at {last_end[channel]})"
        )


@numba.njit(cache=True, nogil=True)
def _check_overlaps(records, last_end):
    for r in records:
        if r["time"] < last_end[r["channel"]]:
            return r["channel"], r["time"]
        last_end[r["channel"]] = strax.endtime(r)
    return -9999, -9999
