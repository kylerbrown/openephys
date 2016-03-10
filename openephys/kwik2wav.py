#!/usr/bin/python

from __future__ import division, print_function, unicode_literals
import numpy as np
from openephys.kwik import load
from scipy.io import wavfile
import argparse


def split_file_to_wav(data, sampling_rate, n_channel, max_length,
                      base_file_name):
    """
    Splits data into smaller subfiles of same length.
    Files use the following convention for naming:
    BASEFILENAME + START_SAMPLE_NUMBER + .wav
    where START_SAMPLE_NUMBER is the location of the first sample in the original
    data set

    returns: a list of start, stop, label, where start is the first sample, stop is the last
    and label is the .wav file name.
    """
    if max_length == np.inf:
        max_length = len(data)
    n_segments = len(data) // max_length
    intervals = [{"start": start_sample,
                  "stop": start_sample + max_length,
                  "label":
                  "{}_{:013d}.wav".format(base_file_name, start_sample)}
                 for start_sample in np.arange(n_segments) * max_length]
    for x in intervals:
        subdata = data[x["start"]:x["start"] + max_length, n_channel]
        wavfile.write(x["label"], sampling_rate,
                      subdata - int(np.mean(subdata)))
    return intervals


def minutes_to_samples(minutes, sampling_rate):
    """
     converts a floating point minute value into an
     integer number of samples, unless infinity, then returns float inf
     """
    if minutes < np.inf:
        return int(minutes * 60. * sampling_rate)
    else:
        return np.inf


def save_intervals(filename, intervals):
    """
    Write .wav file intervals to file

    Parameters
    ----------
    filename : string
        outputs filename (.csv file)
    intervals : a list of dictionaries
         with the fields: start, stop, label
    """
    with open(filename, "w") as f:
        f.write("start,stop,label\n")  # header
        [f.write("{},{},{}\n".format(x["start"], x["stop"], x["label"]))
         for x in intervals]


def main(kwikfiles, n_channel, max_minutes, verbose=False):
    for kwikfile in kwikfiles:
        verbose and print(kwikfile)
        dfile = load(kwikfile)
        verbose and print("loaded\t{}".format(kwikfile))
        if n_channel == -1:
            n_channel = dfile["data"].shape[1] - 1
        basename = "{}_channel_{}".format(kwikfile, n_channel)
        sampling_rate = int(dfile["info"]["sample_rate"])
        verbose and print("shape:{}\tsampling rate:{}".format(dfile[
            "data"].shape, sampling_rate))
        max_length = minutes_to_samples(max_minutes, sampling_rate)
        intervals = split_file_to_wav(dfile["data"], sampling_rate, n_channel,
                                      max_length, basename)
        save_intervals(basename + "_intervals.csv", intervals)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="kwik2wav",
        description="Copies data from one channel to a .wav file")
    p.add_argument("kwikfiles", help="input kwikfile file[s]", nargs="+")
    p.add_argument("-c",
                   "--channel",
                   help="channel to extract, last channel by default",
                   default=-1,
                   type=int)
    p.add_argument("-m",
                   "--minutes",
                   type=float,
                   default=np.inf,
                   help="maximum number of minutes per wav file, \
                   default is {}".format(np.inf))
    p.add_argument("-v", "--verbose", action="store_true", default=False)
    options = p.parse_args()
    main(options.kwikfiles, options.channel, options.minutes, options.verbose)
