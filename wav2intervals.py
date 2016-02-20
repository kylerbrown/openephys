#/usr/bin/python

from __future__ import division, print_function, unicode_literals
from scipy.io import wavfile
from kwik2wav import save_intervals


def create_intervals(wavfiles):
    intervals = []
    cur_sample = 0
    for wav in wavfiles:
        sampling_rate, data = wavfile.read(wav, mmap=True)
        intervals.append({"start": cur_sample, "stop": cur_sample + len(data), "label": wav})
        cur_sample = len(data)
    return intervals


def main(wavfiles, outfile):
    intervals = create_intervals(wavfiles)
    save_intervals(outfile, intervals)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        prog='wav2intervals',
        description="creates a sample interval file for a list of ordered wav files")
    p.add_argument("wavfiles", help="input wav files", nargs="+")
    p.add_argument("-o", "--out", help="name of file to save the intervals in",
                   default="intervals.csv")
    options = p.parse_args()
    main(options.wavfiles, options.out)
