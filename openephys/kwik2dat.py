#!/usr/bin/python

# Converts open-ephys .kwd files to raw binary, a format ready
# for spike sorting with the Phy spike sorting program

from __future__ import division, print_function, unicode_literals
import os.path
import numpy as np
from kwik import load, load_all

# number of data points to write at a time, prevents excess memory usage
BUFFERSIZE = 131072

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


def write_binary(filename, data, channels=()):
    if not data.dtype == np.int16:
        raise TypeError("data should be of type int16")
    if not channels:
        channels = range(data.shape[1])
    n_rows = int(BUFFERSIZE / len(channels))
    with open(filename, "wb") as f:
        for i in range(int(data.shape[0] / n_rows) + 1):
            index = i * n_rows
            buffer = data[index:index + n_rows, channels]
            f.write(buffer.tobytes())


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


def main(filename, channels=(), recordings=()):
    if recordings:
        all_data = [load(filename, x) for x in recordings]
    else:
        all_data = load_all(filename)
    sampling_rate = all_data[0]["info"]["sample_rate"]
    if channels:
        n_channels = len(channels)
    else:
        n_channels = all_data[0]["data"].shape[1]
    root_filename = os.path.splitext(filename)[0]
    dat_filenames = []
    for group_i, data in enumerate(all_data):
        binary_filename = "{fname}_{i}.dat".format(fname=root_filename,
                                                   i=group_i)
        dat_filenames.append(binary_filename)
        write_binary(binary_filename, data["data"], channels)

    prb_filename = root_filename + ".prb"
    dummy_prb_file(prb_filename, n_channels)
    prm_filename = root_filename + ".prm"
    dummy_prm_file(prm_filename, prb_filename, dat_filenames, sampling_rate,
                   n_channels)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        prog="kwik2dat",
        description="""Converts Open-Ephys .kwd files to raw binary
        for use with Phy or Neuroscope""")
    p.add_argument("kwdfile", help="input .kwd file")
    p.add_argument("-c",
                   "--channels",
                   help="Channels to extract (0-indexed), all by default",
                   nargs="+",
                   type=int)
    p.add_argument("-r",
                   "--recordings",
                   help="Recordings to extract, all by default",
                   nargs="+",
                   type=int)
    options = p.parse_args()
    main(options.kwdfile, options.channels, options.recordings)
