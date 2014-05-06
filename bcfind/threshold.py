"""
Functions for image thresholding
"""
from __future__ import print_function
import math
from bcfind import timer

multi_kapur_timer = timer.Timer('Multi Kapur')


def _info(p):
    if p > 0:
        return -p * math.log(p)
    else:
        return 0


def kapur(histogram):
    """Computes binarization threshold using the maximum entropy approach

    Parameters
    ----------
    histogram : array-like
        Image histogram

    Returns
    -------
    max_entropy_threshold : int

    References
    ----------
    J. N. Kapur, P. K. Sahoo, and A. K. C. Wong, A new method for
    gray-level picture thresholding using the entropy of the
    histogram, Comput. Vision Graphics Image Process. 29, 1985,
    273--285.
    """
    s = sum(histogram)
    histogram = [float(h)/float(s) for h in histogram]
    N = len(histogram)
    prob_t = [histogram[0]]
    for i in xrange(1, N):
        prob_t.append(prob_t[i - 1] + histogram[i])

    h_background, h_foreground = [], []

    for t in xrange(N):
        if prob_t[t] > 0:
            h_background.append(sum([_info(histogram[i] / prob_t[t]) for i in xrange(t+1)]))
        else:
            h_background.append(0)

        if 1 - prob_t[t] > 0:
            h_foreground.append(sum([_info(histogram[i] / (1-prob_t[t])) for i in xrange(t+1, N)]))
        else:
            h_foreground.append(0)

    # Maximize entropy
    h_max = 0
    argmax = 0
    for t in xrange(N):
        # print('Kapur: current entropy, t=',t,'H_b+H_f=',h_background[t] + h_foreground[t])
        if h_background[t] + h_foreground[t] > h_max:
            h_max = h_background[t] + h_foreground[t]
            argmax = t
    print('Kapur: best entropy=', h_max, 'argmax (threshold)=', argmax)
    return argmax


##################################################
@multi_kapur_timer.timed
def multi_kapur(histogram, n_divisions):
    """multi_kapur(histogram, n_divisions)

    Multi-thresholding using maximum entropy.

    Rather than just dividing into background and foreground,
    determine `n_divisions` bins such that :math:`\sum_i=1^n H(bin_i)`
    is maximized. Since the algorithm tries all possible divisions
    by brute force, it is only practical for small values of `n_divisions`.

    Parameters
    ----------
    histogram : array-like
        Image histogram

    n_divisions : int
        Number of bins

    Returns
    -------
    thresholds : list
        The `n_divisions` max entropy thresholds
    """
    interval_entropy = []
    N = 257
    for i in xrange(N):
        interval_entropy.append([None]*N)
    s = sum(histogram)
    histogram = [float(h)/float(s) for h in histogram]
    # histogram = histogram[0:10]

    min_t = 0
    max_t = len(histogram)
    # Create all intervals
    intervals = _compute_intervals(n_divisions, 0, len(histogram))
    # print(intervals)

    best_entropy = -1e50
    best_interval = None

    # For each division, compute the entropies and sum them;
    # Determine the interval with maximum sum of entropies
    for i in xrange(len(intervals)):
        interval = intervals[i]
        # print('debug',i,intervals[i])
        entropy = 0.0
        last_t = min_t
        for t in interval:
            entropy += _compute_interval_entropy(histogram,interval_entropy,last_t, t)
            # print(interval_entropy)
            last_t = t
        entropy += _compute_interval_entropy(histogram,interval_entropy,last_t, max_t)
        # print('Multi_Kapur: i-th entropy i=',i,'e=',entropy,'Interval (thresholds)=',interval)
        if best_entropy < entropy:
            best_entropy = entropy
            best_interval = interval
    # print('Multi_Kapur: best entropy=',best_entropy,'best_interval (thresholds)=',best_interval)
    return best_interval


def _compute_intervals(n_divisions, min_t, max_t):
    """
    Find all possible intervals between min_t and max_t
    Only practical if n_divisions is less (or equal) than 3
    """
    intervals = []
    if n_divisions == 1:
        for n in xrange(min_t + 1, max_t):
            intervals.append([n])
    else:
        for n in xrange(min_t + 1, max_t - n_divisions + 1+1):
            subIntervals = _compute_intervals(n_divisions - 1, n, max_t)
            # combine
            for subInterval in subIntervals:
                interval = [n]+subInterval
                intervals.append(interval)
    return intervals


def _compute_interval_entropy(histogram, interval_entropy, begin, end):
    """
    Compute interval entropy from begin (included) to end (excluded)
    """
    ie = interval_entropy[begin][end]
    if ie is None:
        hSum = sum(histogram[begin:end])
        entropy = 0
        for i in xrange(begin, end):
            h = histogram[i]
            if h > 0:
                a = histogram[i] / hSum
                entropy -= a * math.log(a)
        ie = entropy
        interval_entropy[begin][end] = ie
    return ie
