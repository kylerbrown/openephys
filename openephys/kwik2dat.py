#!/usr/bin/python

# Converts open-ephys .kwd files to raw binary, a format ready
# for spike sorting with the Phy spike sorting program

from __future__ import division, print_function, unicode_literals
import os.path
import numpy as np
from openephys.kwik import load, load_all
from openephys.klustafiles import _prm_file, _prb_group

# number of data points to write at a time, prevents excess memory usage
BUFFERSIZE = 131072


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
    p.add_argument("--klustafiles",
                   help="creates template files for klustakwik/phy")
    options = p.parse_args()
    main(options.kwdfile, options.channels, options.recordings,
         options.klustafiles)
