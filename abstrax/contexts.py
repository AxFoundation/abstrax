"""Context builders for abstrax.

The default context wires up:

* a raw data interface plugin providing ``raw_records`` (e.g. from a DAQ or a
  simulation chain)
* abstrax processing plugins (raw_records -> records -> peaks -> events)
* Local-disk storage at ``./strax_data`` by default

No MongoDB, no RunDB, no network. All physics parameters come from
``abstrax/config/detector.yaml`` (via the ``detector://`` URLConfig protocol).

A flat fake gain array is generated once into ``abstrax/resources/`` and
cached there across runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import strax

from abstrax.plugins import events as _events
from abstrax.plugins import peaks as _peaks  # subpackage
from abstrax.plugins import raw_records as _raw_records
from abstrax.plugins import records as _records

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent  # abstrax/ package directory
_RESOURCE_DIR = _HERE / "resources"

# All abstrax-defined plugins in registration order.
ABSTRAX_PLUGINS = [
    _records.PulseProcessing,
    _peaks.Peaklets,
    _peaks.PeakletClassification,
    _peaks.MergedS2s,
    _peaks.Peaks,
    _peaks.PeakBasics,
    _peaks.PeakPositions,
    _events.Events,
    _events.EventBasics,
    _events.CorrectedAreas,
    _events.EnergyEstimates,
    _events.EventInfo,
]


def abstrax_context(
    output_folder: str | Path = "./strax_data",
    **context_kwargs,
) -> strax.Context:
    """Build a strax.Context with the abstrax plugins registered.

    Parameters
    ----------
    output_folder
        Directory for cached strax data. Created if it does not exist.
    **context_kwargs
        Forwarded to ``strax.Context``.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    st = strax.Context(
        storage=[
            strax.DataDirectory(str(output_folder), readonly=False, provide_run_metadata=True)
        ],
        register=[],
        **context_kwargs,
    )

    # Register the interface plugin that provides the raw data, e.g. from a DAQ
    # or from simulations:
    st.register(_raw_records.DAQReader)

    # Register plugins required to process raw_records into events:
    for plugin in ABSTRAX_PLUGINS:
        st.register(plugin)

    gain_file = _ensure_gain_model(
        int(strax.URLConfig.evaluate_dry("detector://n_tpc_pmts")),
        float(strax.URLConfig.evaluate_dry("detector://pmt_to_pe")),
    )
    st.set_config({"gain_model": f"numpy://{gain_file}"})

    return st


# ---------------------------------------------------------------------------
# Fake resource generation
# ---------------------------------------------------------------------------


def _ensure_gain_model(n_tpc_pmts: int, pmt_to_pe: float) -> Path:
    """Generate a flat fake gain array if absent and return its path.

    :param n_tpc_pmts: number of channels.
    :param pmt_to_pe: PE per ADC count and sample of every channel (``pmt_to_pe`` in detector.yaml).
    :return: path of the ``.npy`` file; its name contains both values.
    """
    _RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    gain_file = _RESOURCE_DIR / f"fake_to_pe_{n_tpc_pmts}_{pmt_to_pe:g}.npy"
    if not gain_file.exists():
        gain = np.full(n_tpc_pmts, pmt_to_pe, dtype=np.float32)
        np.save(gain_file, gain)
        log.info("Generated fake gain model: %s", gain_file)
    return gain_file
