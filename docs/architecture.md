# abstrax architecture

## Why this exists

Adapting `straxen` to a generic large-scale LXe TPC by forking and deleting the XENONnT-specific code is tempting and wrong: the diff against upstream rapidly becomes unmaintainable, and "removing all XENONnT assumptions" is much harder than it looks because they're scattered across plugin defaults, dataclass fields, hardcoded array shapes, and numba-compiled helpers.

The alternative chosen here: **treat `straxen` as a library, and build fresh plugins on top of `strax`.**

## Dependency layout

```
┌────────────────────────────────────────────────┐
│                    abstrax                      │
│  (this package: contexts, plugins, protocols)   │
└───────────────────────┬────────────────────────┘
                        ▼
                    ┌───────┐
                    │ strax │  ← streaming framework + URLConfig
                    └───────┘
```

`URLConfig` used to live in `straxen`; it now lives in `strax` (`strax.URLConfig`), so `abstrax`
does not import `straxen` at all. Plugin code is a port of straxen's, but nothing from
`straxen` (contexts, plugins, MongoDB, xedocs, corrections, utilix) is imported at runtime.

`straxen` can still be imported alongside `abstrax` in either order. For that to keep working,
abstrax must never register a URL protocol name that straxen also registers (e.g. `json`,
`resource`, `format`, `take`, `itp_map`). That is why abstrax's protocols are `json_file://` and
`pkg://`.

## Single source of truth for detector parameters

All detector-specific values live in [`abstrax/config/detector.yaml`](../abstrax/config/detector.yaml) and are accessed through the `detector://` URLConfig protocol registered in [`abstrax/url_protocols.py`](../abstrax/url_protocols.py).

Why this matters: `abstrax` must work for a generic detector, not the XENONnT defaults baked into straxen. Keeping every parameter in one file means plugins cannot silently drift apart; they all consume the same `detector://` URLs.

## Plugin chain

```
(existing strax folder, e.g. from a DAQ or simulation)
  │
  ▼  DummyDAQReader        [plugins/raw_records/raw_records.py]
raw_records
  │
  ▼  PulseProcessing       [plugins/records/records.py]
records
  │
  ▼  Peaklets              [plugins/peaks.py]
peaklets
  │
  ▼  Peaks                 [plugins/peaks.py]
peaks
  │
  ▼  PeakBasics            [plugins/peaks.py]
peak_basics
  │
  ▼  Events                [plugins/events.py]
events
  │
  ▼  EventBasics           [plugins/events.py]
event_basics
```

The skeleton commits to this dependency graph and the dtype contracts at each step. Algorithm implementations are TODOs to be filled in.

## What's deliberately missing

- **Corrections.** No CMT, no xedocs, no field distortion correction. When corrections become necessary, they should live in a new `abstrax.plugins.corrections` module and consume the same `detector://` / `resource://` protocols.
- **Position reconstruction.** A `PeakPositions` plugin will be needed eventually. When added, the algorithm itself (e.g., a neural net) should be loaded via a `model://` or `tf://` URLConfig protocol so the algorithm choice is a config knob, not a code change.
- **Veto detectors.** None modeled. If needed later, they get their own `*_nv` / `*_mv` plugin trees that depend only on their own raw data, never on TPC plugins.
- **Live monitoring / RunDB.** abstrax is for offline processing of simulated data. Online concerns are out of scope.

## Reproducibility and caching

strax computes a lineage hash from `(plugin __version__, tracked config values, upstream lineages)` and uses it as the cache key. Two things follow:

1. **Bump `__version__` on every PR that changes a plugin's numeric output.** Otherwise stale cached data is silently reused.
2. **Mark every config that affects output with `track=True`.** Untracked configs do not invalidate the cache; that's a footgun for anything but pure cosmetic options.

## End-to-end validation

The `notebooks/` directory holds a reference notebook that runs the full chain: raw_records → abstrax processing → plots. CI runs this notebook on every PR. If unit tests pass but the notebook breaks, the failure is almost always at the raw_records↔abstrax boundary (config schema mismatch, dtype field rename in the input data, etc.) — exactly what the notebook is there to catch.
