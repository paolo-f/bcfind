#!/usr/bin/env python
"""
Summarizes results on several substacks by showing a table
"""
from __future__ import print_function
import argparse
import os
import pandas as pd
import uuid

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('resdir', metavar='resdir', type=str,
                        help='expected to contain results in resdir/*/eval.log')
    return parser

def main(args):
    # command = ["awk", "FNR==1 && NR!=1{next;}{print}", "%s/*/eval.log" % args.resdir]
    command = ["awk", "FNR==1 && NR!=1{next;}{print}", "%s/*/eval.log" % args.resdir]
    tempname = '/tmp/'+str(uuid.uuid4())+'.csv'
    os.system("awk 'FNR==1 && NR!=1{next;}{print}' %s/*/eval.log > %s" % (args.resdir,tempname))
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
    totals = [{'substack':'TOTAL', 'precision':precision, 'recall':recall,
               'F1':F1, 'TP':TP, 'FP':FP, 'FN':FN, '|true|':true, '|pred|':pred }]
    res=table.append(totals)
    res.precision *= 100
    res.recall *= 100
    res.F1 *= 100
    return table,res

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    t,r=main(args)
    pd.set_option('precision',2)
    pd.set_option('display.height', 500)
    pd.set_option('display.max_rows', 500)
    print(r)
# df2=r[['precision','recall','F1']]
# df2.plot(kind='bar')
