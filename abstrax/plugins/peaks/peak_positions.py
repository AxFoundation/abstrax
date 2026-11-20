"""PeakPositions: peaks -> peak_positions.

Placeholder: outputs zeros for (x, y) so the event-building chain runs
end-to-end. A real reconstruction will use peaks["area_per_channel"] on
the top array; replace compute() when that is implemented.
"""

from __future__ import annotations

import numpy as np
import strax


class PeakPositions(strax.Plugin):
    """Reconstruct (x, y) position for each peak.

    Placeholder: all positions returned as (0, 0) until a reconstruction
    algorithm is implemented. Depends on peaks directly so the real
    implementation has access to area_per_channel for top-array pattern
    fitting.
    """

    __version__ = "0.0.1"

    depends_on = "peaks"
    provides = "peak_positions"
    data_kind = "peaks"
    parallel = True
    compressor = "zstd"
    save_when = strax.SaveWhen.EXPLICIT

    dtype = strax.time_fields + [
        (("Reconstructed X position of the peak, uncorrected [cm]", "x"), np.float32),
        (("Reconstructed Y position of the peak, uncorrected [cm]", "y"), np.float32),
    ]

    def compute(self, peaks):
        out = np.zeros(len(peaks), dtype=self.dtype)
        out["time"] = peaks["time"]
        out["endtime"] = strax.endtime(peaks)
        return out
