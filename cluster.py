#/usr/bin/python

# cluster
# Kyler Brown (kjbrown@uchicago.edu)
#
# Takes a segmented output from segment.py and clusters the segments based on chosen features

# goals: plot segments in a low dimensional space,
# by clicking on a point (segment), the full spectrogram should be visible.
# Ideally there could be some hand labeling of the syllables as a seed for automatic clustering

from __future__ import division, print_function, unicode_literals
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.io import wavfile


def cluster(X, labels, n_clusters=7):
    """
    Clusters X

    X: features
    labels: An array of the same length as X, or a number of clusters
            Cluster labels to use as seed for clustering, `None`s are ignored.
            The number of clusters is dependent on the number of unique labels
            in the label array. If the label array is None or an array of None
            n_clusters is used, otherwise n_clusters is ignored.
            K-means is then seeded by the means of the points of each cluster.
    n_clusters: The number of clusters

    returns
    -------
    IDs = new labels
    """
    pass


def plot_segment(ax, data, sampling_rate, plot_type, **kwargs):
    """
    plots data for a segment

    ax: axis on which to plot
    data:
    sampling_rate:
    plot_type: either "oscillogram" or "spectrogram"

    returns
    -------
    ax
    """
    if plot_type == "oscillogram":
        t = np.arange(len(data)) / sampling_rate
        ax.plot(t, data)
    elif plot_type == "spectrogram":
        # parameters from SAP
        NFFT = 1024
        FFT_step = 10
        FFT_size = 400
        frequency_range = 0.5
        plt.sca(ax)
        plt.specgram(data, NFFT=NFFT, Fs=sampling_rate, noverlap=FFT_step)
        plt.ylim(0, sampling_rate / 2. * frequency_range)
    return ax


def get_raw_data(sample_start, duration, filename, filetype="wav", pad=0):
    """
    sample_start: sample number in file
    duration: length of segment in samples
    filename: file to load

    returns
    -------
    rate : int
        Sample rate of file
    data : numpy array
        data from file
    """
    if filetype == "wav":
        sampling_rate, fulldata = wavfile.read(filename, mmap=True)
    data = fulldata[sample_start:sample_start + duration]
    return sampling_rate, data


def find_file(sample, filemap):
    """
    input: a master sample and filemap dataframe
    returns the file sample and file
    """
    print(sample)
    sample = sample
    row = filemap[(filemap.start <= sample) & (filemap.stop > sample)]
    print(row)
    return sample - row.start.iloc[0], row.label.iloc[0]


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, fig, ax, ax2, xs, ys, X, line, filemap, wavpath):
        self.fig = fig
        self.plot_type = "spectrogram"  # "oscillogram" or "spetrogram"
        self.line = line
        self.ax = ax
        self.ax2 = ax2
        self.lastind = 0
        self.xs = xs
        self.ys = ys
        self.X = X
        self.filemap = filemap
        self.wavpath = wavpath
        self.text = ax.text(0.05,
                            0.95,
                            'selected: none',
                            transform=ax.transAxes,
                            va='top')
        self.selected, = ax.plot([xs.iloc[0]],
                                 [ys.iloc[0]],
                                 'o',
                                 ms=12,
                                 alpha=0.4,
                                 color='yellow',
                                 visible=False)
        self.label_mode = False

    def onpress(self, event):
        if self.lastind is None:
            return
        elif self.label_mode:
            print("label set to {}".format(event.key))
            self.X.loc[self.lastind, "label"] = event.key
            self.label_mode = False
            self.update()
        elif event.key in ('n', 'p'):
            if event.key == 'n':
                inc = 1
            else:
                inc = -1
            self.lastind += inc
            self.lastind = np.clip(self.lastind, 0, len(self.xs) - 1)
            self.update()
        elif event.key in ('a', ):
            self.label_mode = True
        elif event.key == 'u':
            #cluster
            self.X = cluster(self.X)
        elif event.key == 'h':
            help_fig = plt.figure(2)
            help_ax = help_fig.add_subplot(111)
            help_ax.set_title('help')
            help_ax.text(0.01,
                         0.95,
                         """Keyboard commands:
            n - Next point
            p - Previous point
            a - lAbel point
            u - cluster (label points first)
            s - Save cluster file
            h - Help, show this message""",
                         verticalalignment='top',
                         horizontalalignment='left',
                         transform=help_ax.transAxes,
                         fontsize=15)
            plt.show()

    def onpick(self, event):

        if event.artist != self.line:
            print('not line')
            return True
        print(event)
        print(event.ind)
        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        distances = np.array(np.hypot(x - self.xs.iloc[event.ind], y -
                                      self.ys.iloc[event.ind]))
        print(distances)
        indmin = distances.argmin()
        print(indmin)
        dataind = event.ind[indmin]
        self.lastind = dataind
        self.update()
        return event

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind
        master_sample_start = int(self.X["sample"].iloc[dataind])
        file_sample, wavfilename = find_file(master_sample_start, self.filemap)
        ax2 = self.ax2
        ax2.cla()
        sampling_rate, data = get_raw_data(
            file_sample, int(self.X["duration"].iloc[dataind]),
            self.wavpath + wavfilename)
        plot_segment(ax2, data, sampling_rate, self.plot_type)
        #ax2.plot(data)
        ax2.text(0.05,
                 0.9,
                 'label: {}'.format(self.X["label"].iloc[dataind]),
                 transform=ax2.transAxes,
                 va='top')
        #ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(self.xs.iloc[dataind], self.ys.iloc[dataind])

        self.text.set_text('selected: %d' % dataind)
        self.fig.canvas.draw()
        self.label_mode = False


def find_PCAs(df, n=2, whiten=True, drop=("sample")):
    """
    Adds additional columns to a dataframe, corresponding to the principle components, using names PCAX wher X is
    the component number.
    drop: columns to drop before taking PCA, though they are still returned
    """

    pca = PCA(n_components=n, whiten=whiten)
    X = pca.fit_transform(df.drop(drop, 1))
    for i in range(n):
        df["PCA{}".format(i)] = X[:, i]
    return df


def main(segment_file, interval_file, wavpath):
    filemap = pd.read_csv(interval_file)
    data = pd.read_csv(segment_file)
    datapca = find_PCAs(data, n=4)
    datapca["label"] = None
    X = datapca
    xs = datapca["PCA1"]
    ys = datapca["PCA2"]
    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title("Kyler's most amazing cluster program")
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 point tolerance
    browser = PointBrowser(fig, ax, ax2, xs, ys, X, line, filemap, wavpath)
    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)
    plt.show(block=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        prog="cluster",
        description="A semi-automatic clustering engine")
    p.add_argument("segments",
                   help="a segment csv file, created by segment.py")
    p.add_argument("-i",
                   "--intervals",
                   required=True,
                   help="an intervals file, which maps samples to wav files")
    p.add_argument("--wavpath",
                   help="path to wav files, if not in current directory",
                   default="")
    options = p.parse_args()
    main(options.segments, options.intervals, options.wavpath)
