"""Tests of the MergedS2s plugin."""

import numpy as np
import pytest
import strax

from abstrax.contexts import abstrax_context

N_SAMPLES = 50
DT = 10
TOP_FRACTION = 0.7


@pytest.fixture
def merged_s2s_plugin(tmp_strax_dir):
    context = abstrax_context(output_folder=tmp_strax_dir)
    return context.get_single_plugin("0", "merged_s2s")


def s2_peaklets(plugin, start_times, area_per_peaklet):
    """S2 peaklets with a Gaussian waveform; TOP_FRACTION of the area in channel 0 (top)."""
    peaklets = np.zeros(len(start_times), dtype=plugin.dtype)
    samples = np.arange(N_SAMPLES)
    waveform = np.exp(-0.5 * ((samples - N_SAMPLES / 2) / 6) ** 2)
    waveform *= area_per_peaklet / waveform.sum()
    n_top_pmts = plugin.n_top_pmts
    for peaklet, start_time in zip(peaklets, start_times):
        peaklet["time"] = start_time
        peaklet["dt"] = DT
        peaklet["length"] = N_SAMPLES
        peaklet["data"][:N_SAMPLES] = waveform
        peaklet["area"] = area_per_peaklet
        peaklet["area_per_channel"][0] = TOP_FRACTION * area_per_peaklet
        peaklet["area_per_channel"][n_top_pmts] = (1 - TOP_FRACTION) * area_per_peaklet
        peaklet["type"] = 2
    return peaklets


def test_merged_s2_has_its_properties(merged_s2s_plugin):
    start_times = [1_000_000, 1_000_000 + N_SAMPLES * DT + 200]
    peaklets = s2_peaklets(merged_s2s_plugin, start_times, area_per_peaklet=1000.0)
    lone_hits = np.zeros(0, dtype=merged_s2s_plugin.deps["lone_hits"].dtype_for("lone_hits"))

    merged = merged_s2s_plugin.compute(peaklets, lone_hits)

    assert len(merged) == 1
    merged_s2 = merged[0]
    assert merged_s2["area"] == pytest.approx(2000.0, rel=1e-4)
    assert merged_s2["time"] < merged_s2["center_time"] < strax.endtime(merged)[0]
    expected_center = (start_times[0] + start_times[1] + N_SAMPLES * DT) / 2
    assert merged_s2["center_time"] == pytest.approx(expected_center, abs=2 * DT)
    assert merged_s2["median_time"] > 0
    assert merged_s2["width"][5] > 0
    assert merged_s2["area_fraction_top"] == pytest.approx(TOP_FRACTION, rel=1e-4)
