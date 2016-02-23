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
import seaborn as sns
from sklearn.cluster import KMeans


def cluster(x, labels, **kwargs):
    """
    Clusters X

    X: features
    labels: An array of the same length as X, or a number of clusters
            Cluster labels to use as seed for clustering, `no-label`s are ignored.
            The number of clusters is dependent on the number of unique labels
            in the label array. If the label array is None or an array of "no-label"
            the kmeans defaults are used. K-means is then seeded by the
            means of the points of each cluster.
    **kwargs: passed to sklearn.cluster.KMeans
    returns
    -------
    IDs = new labels
    """
    if labels is None or len(set(labels)) == 1:
        kmeans = KMeans(n_clusters=10, **kwargs)
        kmeans.fit(x)
        return kmeans.predict(x)
    else:
       ordered_label_set = list(set(labels))
       if "no-label" in ordered_label_set:
           ordered_label_set.remove("no-label")
       n_clusters = len(ordered_label_set)
       start_means = [np.mean(x[labels==label], 0) for label in ordered_label_set]
       nd_start_means = np.row_stack(start_means)
       print(nd_start_means)
       kmeans = KMeans(n_clusters=len(ordered_label_set),
                       init=nd_start_means)
       kmeans_labels = kmeans.fit_predict(x)
       # map kmeans labels onto original labels
       for i, label in enumerate(ordered_label_set):
           labels[kmeans_labels == i] = label
    return labels




def create_palette(x):
    print(x["label"].unique())
    return sns.husl_palette(len(x["label"].unique()))


def plot_scatter(ax, data, old_lines, x_feature="duration", y_feature="pitch"):
    """
    creates a scatter plot from a dataframe, each unique entry in the "label" column gets
    a unique color
    """
    palette = create_palette(data)
    lines = []
    for l in old_lines:
        l.remove()
        del l
    x = data[x_feature]
    y = data[y_feature]
    for color, label in zip(palette, data["label"].unique()):
        x1 =x[data["label"]==label]
        y1 = y[data["label"]==label]
        l, = ax.plot(x1, y1, 'o', color=color, alpha=.7, picker=5)
        lines.append(l)
    plt.sca(ax)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    return lines, data[x_feature], data[y_feature]



def plot_segment(ax, data, pad, sampling_rate, plot_type, **kwargs):
    """
    plots data for a segment

    ax: axis on which to plot
    data:
    sampling_rate:
    plot_type: either "oscillogram" or "spectrogram"
    pad: number of extra samples on each side of the data

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
        FFT_step = 40
        #FFT_size = 400
        frequency_range = 0.5
        plt.sca(ax)
        plt.specgram(data, NFFT=NFFT, Fs=sampling_rate, noverlap=NFFT-FFT_step)
        plt.ylim(0, sampling_rate / 2. * frequency_range)
    return ax


def get_raw_data(sample_start, duration, filename, filetype="wav", pad=10000):
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
    data = fulldata[sample_start - pad: sample_start + duration + pad]
    return sampling_rate, data, pad


def find_file(sample, filemap):
    """
    input: a master sample and filemap dataframe
    returns the file sample and file
    """
    sample = sample
    row = filemap[(filemap.start <= sample) & (filemap.stop > sample)]
    return sample - row.start.iloc[0], row.label.iloc[0]


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, fig, ax, ax2, xs, ys, X, lines, filemap, wavpath):
        self.fig = fig
        self.plot_type = "spectrogram"  # "oscillogram" or "spetrogram"
        self.lines = lines
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

    def _cluster(self):
        X = self.X
        cluster_features = X
        for col_name in X:
            if not col_name.startswith("PCA"):
                cluster_features = cluster_features.drop(col_name, 1)
        print(cluster_features.columns)
        new_labels = cluster(cluster_features, X['label'])
        self.X["label"] = new_labels

    def onpress(self, event):
        if self.lastind is None:
            return
        elif self.label_mode:
            print("label set to {}".format(event.key))
            self.X.loc[self.lastind, "label"] = event.key
            self.label_mode = False
            self.update()
            self.lines, self.xs, self.ys = plot_scatter(self.ax, self.X, self.lines)
        elif event.key == ">":
            self.save_labels()
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
            self._cluster()
            self.lines, self.xs, self.ys = plot_scatter(self.ax, self.X, self.lines)
        elif event.key == 't':
            if self.plot_type == "oscillogram":
                self.plot_type = "spectrogram"
            elif self.plot_type == "spectrogram":
                self.plot_type = "oscillogram"
            self.update()
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
            > - Save cluster file
            t - cycle plot type
            h - Help, show this message""",
                         verticalalignment='top',
                         horizontalalignment='left',
                         transform=help_ax.transAxes,
                         fontsize=15)
            plt.show()

    def save_labels(self):
        """
        saves labels to csv file, keeping the "sample" and "duration" columns
        """
        saved_columns = ("sample", "duration", "label")
        out_data = self.X
        for n in self.X.columns:
            if n not in saved_columns:
                out_data = out_data.drop(n, 1)
        out_data.to_csv("labels.csv", index=False)
        print("labels saved")
    def onpick(self, event):
        N = len(event.ind)
        if not N:
            return True
        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        if x == None or y == None:
            return True
        print("on pick debug")
        print("x: {}, y: {}".format(x, y))
        print("event.ind: {}".format(event.ind))
        distances = np.array(np.hypot(x - self.xs, y -
                                      self.ys))
        print(distances)
        indmin = distances.argmin()
        print(indmin)
        dataind = indmin
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
        sampling_rate, data, pad = get_raw_data(
            file_sample, int(self.X["duration"].iloc[dataind]),
            self.wavpath + wavfilename)
        plot_segment(ax2, data, pad, sampling_rate, self.plot_type)
        ax2.text(0.05,
                 0.9,
                 'label: {}'.format(self.X["label"].iloc[dataind]),
                 transform=ax2.transAxes,
                 va='top')
        self.selected.set_visible(True)
        self.selected.set_data(self.xs.iloc[dataind], self.ys.iloc[dataind])
        print("selected x: {}".format(self.xs.iloc[dataind]))
        print("selected y: {}".format(self.ys.iloc[dataind]))

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
    data = data.drop("aperiodicity", 1)
    data = data.drop("goodness", 1)
    data = data.drop("AM", 1)
    data = data.drop("FM", 1)
    datapca = find_PCAs(data, n=3)
    datapca["label"] = "no-label"
    X = datapca
    xs = datapca["duration"]
    ys = datapca["pitch"]
    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title("Kyler's most amazing cluster program")
    plt.sca(ax)
    plt.xlim(np.min(xs), np.max(xs))
    plt.ylim(np.min(ys), np.max(ys))
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 point tolerance
    browser = PointBrowser(fig, ax, ax2, xs, ys, X, (line,), filemap, wavpath)
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
