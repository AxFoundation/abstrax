"""Tests that the abstrax context can be constructed."""


def test_context_builds(tmp_strax_dir):
    """The default context must build with no external deps."""
    from abstrax.contexts import abstrax_context

    st = abstrax_context(output_folder=tmp_strax_dir)
    # The processing plugins should all be registered.
    for dtype in (
        "raw_records",
        "records",
        "peaklets",
        "peaks",
        "peak_basics",
        "events",
        "event_basics",
    ):
        assert dtype in st._plugin_class_registry, f"{dtype} not registered"


def test_gain_model_is_built_from_pmt_to_pe(tmp_strax_dir):
    import numpy as np
    import strax

    from abstrax.contexts import abstrax_context

    st = abstrax_context(output_folder=tmp_strax_dir)
    gain_model = strax.URLConfig.evaluate_dry(st.config["gain_model"])
    assert len(gain_model) == strax.URLConfig.evaluate_dry("detector://n_tpc_pmts")
    np.testing.assert_allclose(gain_model, strax.URLConfig.evaluate_dry("detector://pmt_to_pe"))


def test_gain_model_uses_the_registered_detector_protocol(tmp_strax_dir, monkeypatch):
    """ember may own ``detector://``; the gain model must use the same values as the plugins."""
    import strax

    from abstrax.contexts import abstrax_context

    registered_protocol = strax.URLConfig._LOOKUP["detector"]
    requested_keys = []

    def recording_detector_protocol(key: str, **kwargs):
        requested_keys.append(key)
        return registered_protocol(key, **kwargs)

    monkeypatch.setitem(strax.URLConfig._LOOKUP, "detector", recording_detector_protocol)
    abstrax_context(output_folder=tmp_strax_dir)
    assert {"n_tpc_pmts", "pmt_to_pe"} <= set(requested_keys)


def test_maximal_drift_time_from_tpc_length_and_drift_velocity(tmp_strax_dir):
    import strax

    from abstrax.contexts import abstrax_context

    st = abstrax_context(output_folder=tmp_strax_dir)
    tpc_length = strax.URLConfig.evaluate_dry("detector://tpc_length")
    drift_velocity = strax.URLConfig.evaluate_dry("detector://drift_velocity_liquid")
    for data_type in ["events", "event_basics"]:
        plugin = st.get_single_plugin("0", data_type)
        assert plugin.drift_time_max == int(tpc_length / drift_velocity)
