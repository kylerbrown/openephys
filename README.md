#openephys

A python library and scripts for loading and converting data saved by the Open Ephys GUI

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
