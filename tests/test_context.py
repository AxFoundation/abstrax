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


def test_gain_model_is_referenced_by_file_name(tmp_strax_dir):
    import strax

    from abstrax.contexts import abstrax_context

    st = abstrax_context(output_folder=tmp_strax_dir)
    n_tpc_pmts = strax.URLConfig.evaluate_dry("detector://n_tpc_pmts")
    pmt_to_pe = strax.URLConfig.evaluate_dry("detector://pmt_to_pe")
    assert st.config["gain_model"] == f"pkg://fake_to_pe_{n_tpc_pmts}_{pmt_to_pe:g}.npy"


def test_lineage_does_not_depend_on_the_install_location(tmp_path, monkeypatch):
    """Moving abstrax/resources/ must not change the hashes of the shipped test data."""
    import numpy as np
    import strax

    import abstrax.contexts
    import abstrax.url_protocols
    from abstrax.contexts import abstrax_context

    data_types = ["peaklets", "lone_hits", "merged_s2s", "event_info"]
    default_keys = [
        str(abstrax_context(output_folder=tmp_path / "default").key_for("012882", data_type))
        for data_type in data_types
    ]

    moved_resource_dir = tmp_path / "elsewhere" / "resources"
    monkeypatch.setattr(abstrax.contexts, "_RESOURCE_DIR", moved_resource_dir)
    monkeypatch.setattr(abstrax.url_protocols, "_RESOURCE_DIR", moved_resource_dir)
    st = abstrax_context(output_folder=tmp_path / "moved")

    assert [str(st.key_for("012882", data_type)) for data_type in data_types] == default_keys
    gain_model = strax.URLConfig.evaluate_dry(st.config["gain_model"])
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
