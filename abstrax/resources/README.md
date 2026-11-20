Files placed here are accessible to plugins via the `pkg://` URLConfig protocol.

Examples of what belongs here:
- PMT position arrays (`.npy`)
- Position reconstruction neural net weights (`.tar.gz`)
- Light yield maps (`.json` or `.npz`)
- Any other small, version-controlled resource that a plugin needs at runtime

Large data (>10 MB) should NOT be committed here — keep them out of the repo
and pull them in at install time, or load them from an external path via the
`yaml://` / `json://` / `numpy://` protocols.
