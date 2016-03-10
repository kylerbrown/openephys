
_prb_group = """
            {i}: {{
                "channels": [{i}],
                "graph": [],
                "geometry": {{{i}: [0, {i}]}}
                }},

"""

_prm_file = """
prb_file = '{prb}'

traces = dict(
    raw_data_files=[{dat_files}],
    voltage_gain=10.,
    sample_rate={sampling_rate},
    n_channels={n_channels},
)

spikedetekt = dict(
    filter_low=500.,  # Low pass frequency (Hz)
    filter_high_factor=0.95 * .5,
    filter_butter_order=3,  # Order of Butterworth filter.

    filter_lfp_low=0,  # LFP filter low-pass frequency
    filter_lfp_high=300,  # LFP filter high-pass frequency

    chunk_size_seconds=1,
    chunk_overlap_seconds=.015,

    n_excerpts=50,
    excerpt_size_seconds=1,
    threshold_strong_std_factor=4.5,
    threshold_weak_std_factor=2.,
    detect_spikes='negative',

    connected_component_join_size=1,

    extract_s_before=16,
    extract_s_after=16,

    n_features_per_channel=3,  # Number of features per channel.
    pca_n_waveforms_max=10000,
)

klustakwik2 = dict(
    num_starting_clusters=100,
)
"""

def dummy_prb_file(filename, n_channels):
    """
    Creates a dummy .prb file, assumes each channel is independent
    """
    with open(filename, "w") as f:
        f.write("channel_groups = {\n")
        for i in range(n_channels):
            f.write(_prb_group.format(i=i))
        f.write("    }")


def dummy_prm_file(filename, prb_file, dat_files, sampling_rate, n_channels):
    dat_file_string = "".join(("'{}',".format(x) for x in dat_files))
    with open(filename, "w") as f:
        f.write(_prm_file.format(filename=filename,
                                 prb=prb_file,
                                 dat_files=dat_file_string,
                                 sampling_rate=sampling_rate,
                                 n_channels=n_channels))

