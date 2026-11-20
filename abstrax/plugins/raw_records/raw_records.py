"""Source plugin: read pre-existing raw_records from disk.

Interface between abstrax processing and a system that provides raw data, for
example the binary files of a DAQ system (see straxen's DAQReader) or data
written by a simulation chain.

The input folder is resolved per run via the ``strax_folder://`` URLConfig
protocol, i.e. ``<raw_records_dir>/<run_id>-raw_records-<lineage_hash>``.
Chunk boundaries, file names and the compressor are taken from the folder's
``metadata.json``.
"""

from __future__ import annotations
from typing import Tuple

import strax

class DAQReader(strax.Plugin):
    """Load raw_records chunk by chunk from an existing strax data folder."""

    __version__ = "MOCKTESTDATA"

    provides: Tuple[str, ...] = ("raw_records",)
    data_kind = "raw_records"
    depends_on: Tuple = tuple()

    def infer_dtype(self):
        return strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)

    def compute(self, chunk_i):
        pass