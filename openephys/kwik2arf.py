#/usr/bin/python

# kwik2arf
# converts .kwik files created by open-ephys to .arf files
# Kyler Brown (kjbrown@uchicago.edu) 2016

import os.path
import numpy as np
import h5py
import arf
import openephys.kwik as kwik

# number of data points to write at a time, prevents excess memory usage
BUFFERSIZE = 131072

def copy(kfile, afile, datatypes=0):
    """
    copies the contents of .kwd hdf5 file to a .arf hdf5 file
    """

    # copy top level attributes
    for k,v in kfile.attrs.items():
        afile.attrs[k] = v

    # descend into /recordings
    recordings = kfile["recordings"]
    # recordings holds the "entries", which have names "0", "1", "2", etc
    for kname, kentry in recordings.items():
        timestamp = 0  # TODO determine correct timestamp
        e = arf.create_entry(afile, kname, timestamp)
        for k,v in kentry.attrs.items():
            e.attrs[k] = v
        kdata = kentry["data"]
        if len(datatypes) == 1:
            datatypes = datatypes * kdata.shape[1]
        else:
            assert len(datatypes) == kdata.shape[1]
        channel_bit_volts = kentry["application_data"].attrs["channel_bit_volts"]
        channel_sample_rates = kentry["application_data"].attrs["channel_sample_rates"]
        # kwik files are SxN datasets, while in arf it's N datasets of length S
        for i in range(kdata.shape[1]):
            dset = arf.create_dataset(e, name=str(i), data=np.array([],dtype=np.int16),
                                      maxshape=(kdata.shape[0],),
                                      sampling_rate=channel_sample_rates[i],
                                      units='samples', datatype=datatypes[i],
                                      compression=6)
            dset.attrs["bit_volts"] = channel_bit_volts[i]
            for j in range(int(kdata.shape[0]/BUFFERSIZE) + 1):
                index = j*BUFFERSIZE
                arf.append_data(dset, kdata[index:index + BUFFERSIZE, i])




def main(kwikfile, datatypes, arf_name):
    if not  datatypes:
        datatypes = [0]
    if not arf_name:
        arf_name = os.path.splitext(kwikfile)[0] + ".arf"
    with h5py.File(kwikfile, "r") as kfile, arf.open_file(arf_name, "w") as afile:
        copy(kfile, afile, datatypes)




if __name__ == "__main__":
    import argparse
    from argparse import RawTextHelpFormatter
    p = argparse.ArgumentParser(
        prog="kwik2arf",
        description="Copies data from a .kwik file to a .arf file",
        formatter_class=RawTextHelpFormatter)
    p.add_argument("kwikfile", help="input .kwik file")
    p.add_argument("-d", "--datatypes",
           help="""integer codes for the datatype of each channel, see
                   https://github.com/melizalab/arf/blob/master/specification.md#datatypes

                   0    UNDEFINED   undefined or unknown
                   1    ACOUSTIC    acoustic
                   2    EXTRAC_HP   extracellular, high-pass (single-unit or multi-unit)
                   3    EXTRAC_LF   extracellular, local-field
                   4    EXTRAC_EEG  extracellular, EEG
                   5    INTRAC_CC   intracellular, current-clamp
                   6    INTRAC_VC   intracellular, voltage-clamp
                  23    EXTRAC_RAW  extracellular, wide-band
                1000    EVENT       generic event times
                1001    SPIKET      spike event times
                1002    BEHAVET     behavioral event times
                2000    INTERVAL    generic intervals
                2001    STIMI       stimulus presentation intervals
                2002    COMPONENTL  component (e.g. motif) labels

               If more than one code is given, the number of codes must match the number of channels""",
               type=int, nargs="+")
    p.add_argument("-o", "--outfile", help="name of output file")

    options = p.parse_args()
    main(options.kwikfile, options.datatypes, options.outfile)
