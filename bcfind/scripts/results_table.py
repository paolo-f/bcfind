#!/usr/bin/env python
"""Tabulate results of one or more experiment directories
"""
from __future__ import print_function
import argparse
import os
import pandas as pd
import uuid

def parse_results(resdir):
    # command = ["awk", "FNR==1 && NR!=1{next;}{print}", "%s/*/eval.log" % resdir]
    command = ["awk", "FNR==1 && NR!=1{next;}{print}", "%s/*/eval.log" % resdir]
    tempname = '/tmp/'+str(uuid.uuid4())+'.csv'
    os.system("awk 'FNR==1 && NR!=1{next;}{print}' %s/*/eval.log > %s" % (resdir,tempname))
    try:
        table = pd.read_csv(tempname, converters={'substack': str})
    except pd.parser.CParserError: # maybe all empty?
        with open(tempname,'w') as ostream:
            print('substack,method,parameter,precision,recall,F1,TP,FP,FN,|true|,|pred|', file=ostream)
            print('010608,unfiltered,0,1.0,0.0,0.0,0,0,4138,4138,0', file=ostream)
        table = pd.read_csv(tempname, converters={'substack': str})
    table = table.drop('method',1).drop('parameter',1)
    TP = table.TP.sum()
    FP = table.FP.sum()
    FN = table.FN.sum()
    true = table['|true|'].sum()
    pred = table['|pred|'].sum()
    if TP+FP>0:
        precision = float(TP)/(TP+FP)
    else:
        precision = 1.0
    recall = float(TP)/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    macroF1 = table.F1.mean()
    precision *= 100.0
    recall *= 100.0
    F1 *= 100.0
    totals = [{'substack':'TOTAL', 'precision':precision, 'recall':recall,
               'F1':F1, 'TP':TP, 'FP':FP, 'FN':FN, '|true|':true, '|pred|':pred }]
    res=table.append(totals)
    return totals,res

def main(args):
    pd.set_option('precision',3)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.show_dimensions', False)
    totals_table = pd.DataFrame(columns=['substack','precision','recall','F1','TP','FP','FN','|true|','|pred|'])
    for resdir in args.resdir:
        totals,res = parse_results(resdir)
        if args.totals_only:
            # print(pd.DataFrame(totals))
            # print(r.tail(1))
            totals[0]['substack'] = resdir
            totals_table = totals_table.append(totals)
        else:
            print('='*len(resdir))
            print(resdir)
            print('='*len(resdir))
            print(res)
    if args.totals_only:
        print(totals_table)

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('resdir', metavar='resdir', type=str, nargs='+',
                        help='must contain resdir/??????/eval.log')
    parser.add_argument('-t', '--totals_only', dest='totals_only', action='store_true',
                        help='Show totals only for each experiment')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
# df2=r[['precision','recall','F1']]
# df2.plot(kind='bar')
