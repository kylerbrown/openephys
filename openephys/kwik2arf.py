#/usr/bin/python

# kwik2arf
# converts .kwik files created by open-ephys to .arf files
# Kyler Brown (kjbrown@uchicago.edu) 2016

import os.path
import numpy as np
import h5py
import arf
import openephy.kwik as kwik


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
            dset = arf.create_dataset(e, name=str(i),
                                      data=np.ravel(kdata[:,i]), compression=6,
                                      sampling_rate=channel_sample_rates[i],
                                      units='samples', datatype=datatypes[i])
            dset.attrs["bit_volts"] = channel_bit_volts[i]




def main(kwikfiles, datatypes):
    if not  datatypes:
        datatypes = [0]
    for kwikfile in kwikfiles:
        arf_name = os.path.splitext(kwikfile)[0] + ".arf"
        with h5py.File(kwikfile, "r") as kfile, arf.open_file(arf_name, "w") as afile:
            copy(kfile, afile, datatypes)




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        prog="kwik2arf",
        description="Copies data from a .kwik file to a .arf file")
    p.add_argument("kwikfiles", help="input .kwik files", nargs="+")
    p.add_argument("-d", "--datatypes",
                   help="""integer codes for the datatype of each channel, see
                   https://github.com/melizalab/arf/blob/master/specification.md#datatypes
                   If more than one code is given, the number of codes must match the number of channels""",
                   type=int, nargs="+")

    options = p.parse_args()
    main(options.kwikfiles, options.datatypes)
