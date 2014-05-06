#!/usr/bin/env python
"""
Summarizes several runs varying r (seed sphere) and R (mean-shift bandwidth)
"""
from __future__ import print_function
import glob
import argparse
import results_table
import pandas as pd
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prefix', metavar='prefix', type=str,
                        help='prefix for the pathname pattern glob /fast/armonia_d/Brain/V_000_stitched_new/results-model_selection-SUP_WNEG_V3-')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='Name of the output spreadsheet file')
    return parser

def expand(res):
    rvals = np.unique(res['r'])
    precision = {}
    recall = {}
    F1 = {}
    for r in rvals:
        subtable = res[res['r']==r]
        precision['R'] = ['R=%.1f'%R for R in subtable['R'].values]
        recall['R'] = ['R=%.1f'%R for R in subtable['R'].values]
        F1['R'] = ['R=%.1f'%R for R in subtable['R'].values]
        precision['r=%.1f'%r] = subtable['precision'].values
        recall['r=%.1f'%r] = subtable['recall'].values
        F1['r=%.1f'%r] = subtable['F1'].values
    precision=pd.DataFrame(precision)
    recall=pd.DataFrame(recall)
    F1=pd.DataFrame(F1)
    return precision, recall, F1

def main(args):
    dirs = glob.glob(args.prefix+'*')
    tables = {}
    for d in dirs:
        parser = results_table.get_parser()
        rt_args = parser.parse_args([d])
        l = d.split('-')
        seed_r = l[-2] # e.g. 'r4.5'
        seed_r = float(seed_r[1:])
        ms_bandwidth = l[-1] # e.g. 'R5.5'
        ms_bandwidth = float(ms_bandwidth[1:])
        _,tables[(seed_r,ms_bandwidth)] = results_table.main(rt_args)
    res = []
    for params, table in sorted(tables.iteritems()):
        total=table.tail(1)
        res.append([params[0], params[1], total.precision[0], total.recall[0], total.F1[0]])
    index = ['r','R','precision','recall','F1']
    res = pd.DataFrame(res,columns=index)
    precision, recall, F1 = expand(res)
    writer = pd.ExcelWriter(args.outfile+'.xlsx')
    res.to_excel(writer,'Overall')
    precision.to_excel(writer,'Precision')
    recall.to_excel(writer,'Recall')
    F1.to_excel(writer,'F1')
    precision.transpose().to_excel(writer,'T Precision')
    recall.transpose().to_excel(writer,'T Recall')
    F1.transpose().to_excel(writer,'T F1')
    writer.save()
    return res

    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    res = main(args)
    pd.set_option('precision',2)
    pd.set_option('display.height', 500)
    pd.set_option('display.max_rows', 500)
    print(res)
