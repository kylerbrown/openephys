#/usr/bin/python

# kwik2arf
# converts .kwik files created by open-ephys to .arf files
# Kyler Brown (kjbrown@uchicago.edu) 2016

import os.path
import arf
import kwik


def copy(kfile, afile):
    """
    copies the contents of .kwd hdf5 file to a .ard hdf5 file
    """

    # copy top level attributes
    for k,v in kwd.attrs.items():
        afile.attrs[k] = v

    # descend into /recordings
    recordings = kfile["recordings"]
    # recordings holds the "entries", which have names "0", "1", "2", etc
    for kname, kentry in recordings.items():
        timestamp = 0  # TODO determine correct timestamp
        e = arf.create_entry(afile, kname, timestamp)
        for k,v in kentry.attrs.items():
            e[k] = v
        kdata = kentry["data"]
        channel_bit_volts = kentry["application_data"].attrs["channel_bit_volts"]
        channel_sample_rates = kentry["application_data"].attrs["channel_sample_rates"]
        # kwik files are SxN datasets, while in arf it's N datasets of length S
        for i in kdata.shape[1]:
            dset = arf.create_dataset(e, name=str(i),
                                      data=np.ravel(kdata[:,i]), compression=6,
                                      sampling_rate=channel_sample_rates[i],
                                      units='samples')
            dset.attrs["bit_volts"] = channel_bit_volts[i]




def main(kwikfiles):
    for kwikfile in kwikfiles:
        arf_name = os.path.splitext(kwikfile)[0] + ".arf"
        with h5py.File(kwikfile, "r") as kfile, arf.open_file(arf_name, "w") as afile:
            copy(kfile, afile)




if __name__ == "__main__":
    import argparse
    p = arparse.ArgumentParser(
        prog="kwik2arf",
        description="Copies data from a .kwik file to a .arf file")
    p.add_argument("kwikfiles", help="input .kwik files", nargs="+")

    options = p.parse_args()
    main(options.kwikfiles)
