"""
Invoke this script to evaluate performances on all substacks that have GT.
It will output on stdout performances for every threshold and it will produce
the precision-recall graph and others.
"""

import pandas as pd
import sys
import utils
from ClassificationScores import ClassificationScores
from EuclideanMetric import EuclideanMetric
from sklearn.metrics import roc_curve
import re
import ujson
import os
import numpy as np
import parameters


def usage():
    print "python " + sys.argv[0] + " indir outdir"
    print "indir: must contain indir/info.json and GT markers files"
    print "pred_dir: must contain substacks with ms.marker predictions"
    print "outdir: where to save plots"

if len(sys.argv) != 4:
    usage()
    sys.exit(1)

indir = utils.add_trailing_slash(sys.argv[1])
pred_dir = utils.add_trailing_slash(sys.argv[2])
outdir = utils.add_trailing_slash(sys.argv[3])

if os.path.exists(utils.add_trailing_slash(indir) + 'info.json'):
    plist = ujson.loads(open(indir + 'info.json').read())
else:
    raise RuntimeError("File info.json needed! Aborting...")

margin = plist['Margin']

regex = re.compile('^\d\d\d\d\d\d-GT.marker$')
files = utils.get_filenames(indir)
gt_filenames = list()
gt_substacks = list()
for f in files:
    if regex.match(f) is not None:
        gt_filenames.append(regex.match(f).string)
        gt_substacks.append(utils.add_trailing_slash(regex.match(f).string[:6]))

gt_data_matrices = dict()
predictions_data_matrices = dict()
thresholds = set()
empty_substacks = set()
empty_filenames = set()

for index in xrange(len(gt_filenames)):
    substack = gt_substacks[index]
    print "Processing substack " + substack
    try:
        predictions_data_matrices[substack] = pd.read_csv(pred_dir + substack +
                                                          'ms.marker').as_matrix([parameters.x_col,
                                                                                  parameters.y_col,
                                                                                  parameters.z_col,
                                                                                  parameters.distance_col])
        gt_data_matrices[substack] = pd.read_csv(indir +
                                                 gt_filenames[index]).as_matrix([parameters.gt_x_col,
                                                                                 parameters.gt_y_col,
                                                                                 parameters.gt_z_col])
    except IOError:
        print indir + substack + " does not contain ms.marker file..."
    if (substack not in gt_data_matrices) and (substack not in predictions_data_matrices):
        print substack + " not in both dictionaries, adding it to the empty dictionaries..."
        empty_substacks.add(substack)
        empty_filenames.add(gt_filenames[index])
        continue

for empty_substack in empty_substacks:
    if empty_substack in gt_substacks:
        gt_substacks.remove(empty_substack)

for empty_filename in empty_filenames:
    if empty_filename in gt_filenames:
        gt_filenames.remove(empty_filename)

for substack, matrix in predictions_data_matrices.iteritems():
    width = plist['SubStacks'][utils.remove_trailing_slash(substack)]['Width']
    height = plist['SubStacks'][utils.remove_trailing_slash(substack)]['Height']
    depth = plist['SubStacks'][utils.remove_trailing_slash(substack)]['Depth']

    for index in xrange(matrix.shape[0]):
        point = np.array([matrix[index, 0], matrix[index, 1], matrix[index, 2]])
        threshold = matrix[index, 3]
        if ClassificationScores.is_inside(point, margin, width, height, depth) is True:
            thresholds.add(threshold)

thresholds = list(thresholds)
thresholds.sort()

precision = np.empty((len(thresholds), ))
recall = np.empty((len(thresholds), ))
F1 = np.empty((len(thresholds), ))
true_positives = np.empty((len(thresholds), ))
false_negatives = np.empty((len(thresholds), ))
false_positives = np.empty((len(thresholds), ))
metric = EuclideanMetric()

last_perf = dict()
last_shape = dict()

tested = 0

print "threshold,tp,fn,fp"
for threshold in thresholds:
    thre_tp = 0
    thre_fn = 0
    thre_fp = 0
    for substack in gt_substacks:
        gt_points_matrix = gt_data_matrices[substack]
        pred_points_matrix = predictions_data_matrices[substack]

        X_pred = pred_points_matrix[pred_points_matrix[:, 3] <= threshold, :3]

        width = plist['SubStacks'][utils.remove_trailing_slash(substack)]['Width']
        height = plist['SubStacks'][utils.remove_trailing_slash(substack)]['Height']
        depth = plist['SubStacks'][utils.remove_trailing_slash(substack)]['Depth']

        if substack in last_shape:
            if X_pred.shape[0] > last_shape[substack]:
                class_score = ClassificationScores(gt_points_matrix, X_pred, metric, margin, width, height, depth)
                p, r, f1, tp, fn, fp, tp_pred_in, fp_pred_in = class_score.graph_based_performances()

                last_perf[substack] = [p, r, f1, tp, fn, fp, tp_pred_in, fp_pred_in]
                last_shape[substack] = X_pred.shape[0]
            elif X_pred.shape[0] == last_shape[substack]:
                p, r, f1, tp, fn, fp, tp_pred_in, fp_pred_in = last_perf[substack]
            else:
                raise RuntimeError("Something unexpected happened! Debug me...")
        else:
            class_score = ClassificationScores(gt_points_matrix, X_pred, metric, margin, width, height, depth)
            p, r, f1, tp, fn, fp, tp_pred_in, fp_pred_in = class_score.graph_based_performances()

            last_perf[substack] = [p, r, f1, tp, fn, fp, tp_pred_in, fp_pred_in]
            last_shape[substack] = X_pred.shape[0]

        thre_tp += tp
        thre_fn += fn
        thre_fp += fp
    p = ClassificationScores.precision(thre_tp, thre_fp)
    r = ClassificationScores.recall(thre_tp, thre_fn)
    f1 = ClassificationScores.f1(p, r)
    precision[tested] = p
    recall[tested] = r
    F1[tested] = f1
    true_positives[tested] = thre_tp
    false_negatives[tested] = thre_fn
    false_positives[tested] = thre_fp
    tested += 1
    print ','.join([repr(threshold), repr(thre_tp), repr(thre_fn), repr(thre_fp)])

y_true = np.empty(0)
y_score = np.empty(0)
for substack, perf in last_perf.iteritems():
    tp_pred_in = np.array(list(perf[6]), dtype=np.int_)
    fp_pred_in = np.array(list(perf[7]), dtype=np.int_)

    pred_points_matrix = predictions_data_matrices[substack]
    tp_pred_thresh = pred_points_matrix[tp_pred_in, 3]
    fp_pred_thresh = pred_points_matrix[fp_pred_in, 3]

    y_true = np.append(y_true, np.ones(tp_pred_in.shape))
    y_score = np.append(y_score, -tp_pred_thresh)

    y_true = np.append(y_true, np.zeros(fp_pred_in.shape))
    y_score = np.append(y_score, -fp_pred_thresh)

fpr, tpr, threshs = roc_curve(y_true, y_score)

utils.plot_curve(recall, precision, 'Recall', 'Precision', outdir)
utils.plot_curve(fpr, tpr, 'FPR', 'TPR', outdir)
utils.plot_curve(thresholds, F1, 'Threshold', 'F1', outdir)
utils.plot_curve(thresholds, precision, 'Threshold', 'Precision', outdir)
utils.plot_curve(thresholds, recall, 'Threshold', 'Recall', outdir)
