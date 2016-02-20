#/usr/bin/python

# segment.py
# Kyler Brown (kjbrown@uchicago.edu)
#
# Segments features of acoustic data
# == Input
#  * a csv of SAP features
#  * a segmentation feature (such as amplitude)
#  * a threshold and threshold direction
# == output
# a two column csv file
# first column, sample number of segment start
# second column, duration of segment in samples
# remainder of columns: mean of each feature for duration of segment

from __future__ import division, unicode_literals, print_function #py2 compatibility
import csv


def open_csvfile(csvfile):
   f = open(csvfile, "r")
   return f, csv.DictReader(f)


def open_writefile(csvfile, fieldnames):
    f = open(csvfile, "w")
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    return f, writer


def thresh_met(x, threshold, less_than):
    """
    x -- number
       input
    threshold -- number
    less_than -- boolean

    Returns true if LESS_THAN is False and X > THRESHOLD or
                    LESS_THAN is True and X < THRESHOLD
    """
    if less_than:
        return x < threshold
    else:
        return x > threshold


def main(csvfile, outfile, feature, less_than, threshold, minimum_size,
         sap_segment=None):
    f1, reader = open_csvfile(csvfile)
    writer_fieldnames = reader.fieldnames.copy()
    writer_fieldnames.insert(1, "duration")
    f2, writer = open_writefile(outfile, writer_fieldnames)
    if sap_segment:
        f3, thresh_reader = open_csvfile(sap_segment)
    in_segment = False
    feature_average_names = set(reader.fieldnames.copy())
    feature_average_names.remove("sample")
    for row in reader:
        if sap_segment:
            thresh = int(next(thresh_reader)["threshold"])
        else:
            thresh = thresh_met(float(row[feature]), threshold, less_than)
        if thresh:
            if not in_segment:
                # start a new segment
                in_segment = True
                feature_averages = {x: [] for x in feature_average_names}
                start_sample = int(row["sample"])
            else:
                # continue adding data to current segment
                pass
            [feature_averages[x].append(float(row[x]))
             for x in feature_averages.keys()]
        else:
            if in_segment:
                # end segment
                in_segment = False
                # compute duration
                duration = int(row["sample"]) - start_sample
                if duration > minimum_size:
                    # compute feature means
                    features = {k: sum(v)/len(v) for k, v in feature_averages.items()}
                    features["sample"] = start_sample
                    features["duration"] = duration
                    # write data to file
                    writer.writerow(features)
            else:
                # thresh not met and not in segment, nothing to do
                pass

    f1.close()
    f2.close()
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        prog="segment",
        description="segmentation based on SAP feature")
    p.add_argument("csvfile", help="A .csv file for analysis")
    p.add_argument("-f", "--feature", default="amplitude",
                   help="feature for segmentation")
    p.add_argument("-l", "--less-than", action="store_true")
    p.add_argument("-t", "--threshold", default=0.25, type=float,
                   help="threshold for feature, must match a column from the header \
                   of the csv file")
    p.add_argument("-o", "--out", help="name of output file", default="segments.csv")
    p.add_argument("--minimum-size", help="minimum size of a segment in samples",
                   type=int, default=1500)
    p.add_argument("--sap-segment", help="use an SAP segmentation file")
    options = p.parse_args()
    main(options.csvfile, options.out, options.feature, options.less_than, options.threshold,
         options.minimum_size, options.sap_segment)
