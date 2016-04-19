#openephys

A python library and scripts for loading and converting data saved by the Open Ephys GUI.

Python 2/3 compatible.

More info on the Open Ephys data format can be found at https://open-ephys.atlassian.net/wiki/display/OEW/Data+format

More info on the Kwik data format (implemented in the GUI as of October 2014) can be found at https://github.com/klusta-team/kwiklib/wiki/Kwik-format
## install

    git clone https://github.com/kylerbrown/openephys.git
    pip install -e openephys

## scripts
+ kwik2arf.py: converts .kwd files to [.arf](https://github.com/melizalab/arf/)
+ kwik2dat.py: converts .kwd files to raw binary, which is used by programs such as [phy](http://phy.readthedocs.org/en/latest/), [neuroscope](http://neurosuite.sourceforge.net/information.html).
+ kwik2wav.py: converts a selected channel to a .wav file for acoustic analysis.


For Python:
- use the 'OpenEphys.py' module for .continuous files, .spikes, and .events files
- use the 'kwik.py' module for .kwd files

# Spike sorting with Spyking Circus
* Obtain source code from [here](http://spyking-circus.readthedocs.org/en/latest/introduction/download.html). Follow the install instructions. Spyking circus has good documentation, so you may want to spend some time reading through it.
* Convert raw.kwd file to a raw binary, for example: `kwik2dat.py experiment1_100.raw.kwd`. If only some channels contain neural data, run `kwik2dat.py experiement1_100.raw.kwd -c 6 7 8`, where `6 7 8` are the zero indexed channel numbers. (You can use `h5ls -r experiement1_100.raw.kwd` to see the total number of recorded channels. Or you can export all the channels by leaving out the `-c` flag and view the raw data first in neuroscope)
* Run `spyking-circus *.dat`, spyking circus will ask if you want to create a parameter file. Say yes. [y ENTER]
* edit the `.params` file. At a minimum you must change these lines:


        data_offset    = 0                    # Length of the header ('MCS' is auto for MCS file)
        mapping        = 3SU.prb     # Mapping of the electrode (see http://spyking-circus.rtfd.ord)
        data_dtype     = int16                 # Type of the data

* Here the file 3SU.prb is an example probe file for 3 independent electrodes. You're going to want to use a different .prb file, one that matches your array. Here's the contents of file, placed in the same directory:


        total_nb_channels = 3
        radius = 00
        channel_groups = {
            0: {
                "channels": [0, 1, 2],
                "graph" : [[0, 1],[1, 2], [0, 2]],
                "geometry": {
                    0: [0, 0],
                    1: [0, 200],
                    2: [200, 200],
                }
            }
        }

Note that the channel names are now 0, 1 and 2, instead of 6, 7, 8, as extracted from the .kwd file. For more information on designing a .prb file see the [spyking circus documentation](http://spyking-circus.readthedocs.org/en/latest/code/probe.html).

* finally, re-run `spyking-circus *.dat`.
* Make coffee :coffee:
