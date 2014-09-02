#!/usr/bin/env python
"""
Compares predicted markers against ground truth. Reports precision, recall, F1 measure.
"""
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import sys
import itertools
import argparse
import pylab

from bcfind.volume import *
from bcfind.markers import distance, match_markers


def inside(c,substack):
    m = substack.plist['Margin']/2
    if c.x<m or c.y<m or c.z<m or c.x>substack.info['Width']-m or c.y>substack.info['Height']-m or c.z>substack.info['Depth']-m:
        return False
    return True


def eval_perf(substack,C_true,C_pred,verbose=True,errors_marker_file=None,rp_file=None, max_cell_diameter=None):
    # max-cardinality bipartite matching
    C_rejected = [c for c in C_pred if c.rejected]
    C_pred = [c for c in C_pred if not c.rejected]

    true_positives_true = set()  # subset of C_true that are true positives
    true_positives_pred = set()  # subset of C_pred that are true positives
    TP = 0
    TP_inside = []
    G,mate,node2center = match_markers(C_true,C_pred, 2*max_cell_diameter)
    kw = 0
    for k1,k2 in mate.iteritems():
        if k1[0] == 'p':  # mate is symmetric
            continue
        c1 = node2center[k1]
        c2 = node2center[k2]
        d = distance((c1.x,c1.y,c1.z),(c2.x,c2.y,c2.z))
        if d < max_cell_diameter/2:
            true_positives_pred.add(c2)
            true_positives_true.add(c1)
            TP += 1
            if inside(c1,substack):
                if verbose:
                    print(' TP:', k2, c2.name,c2.x,c2.y,c2.z,c2, k1, c1.name,c1.x,c1.y,c1.z,d)
                TP_inside.append(c2)
                kw += 1
            else:
                if verbose:
                    print('OTP:', k2, c2.name,c2.x,c2.y,c2.z,c2, k1, c1.name,c1.x,c1.y,c1.z,'d:',d)
        else:
            if verbose:
                print('---> too far', c2.name,c2.x,c2.y,c2.z,c2,c1.name,c1.x,c1.y,c1.z,d)

    FP_inside,FN_inside = [],[]
    if errors_marker_file is not None:
        ostream = open(errors_marker_file,'w')
        print('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b',file=ostream)

    for i,c in enumerate(C_true):
        if c not in true_positives_true:
            if inside(c,substack):
                r,g,b = 255,0,255
                name = 'FN_%03d (%s)' % (i+1,c.name)
                cx,cy,cz = int(round(c.x)),int(round(c.y)),int(round(c.z))
                comment = ':'.join(map(str,[cx,cy,cz,c]))
                if errors_marker_file is not None:
                    print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)
                FN_inside.append(c)
                if verbose:
                    print('FN: ', c.name,c.x,c.y,c.z,c)
    for i,c in enumerate(C_pred):
        c.is_false_positive = False
        if c not in true_positives_pred:
            if inside(c,substack):
                r,g,b = 255,0,0
                name = 'FP_%03d (%s)' % (i+1,c.name)
                c.is_false_positive = True
                cx,cy,cz = int(round(c.x)),int(round(c.y)),int(round(c.z))
                comment = ':'.join(map(str,[cx,cy,cz,c]))
                if errors_marker_file is not None:
                    print(','.join(map(str,[1+cx,1+cy,1+cz,0,1,name,comment,r,g,b])), file=ostream)
                FP_inside.append(c)
                if verbose:
                    print('FP: ', c.name,c.x,c.y,c.z,c)
    # Also print predicted TP in error marker file (helps debugging)
    for i,c in enumerate(C_pred):
        if c in true_positives_pred:
            if inside(c,substack):
                r,g,b = 0,255,0
                name = 'TP_%03d (%s)' % (i+1,c.name)
                cx,cy,cz = int(round(c.x)),int(round(c.y)),int(round(c.z))
                comment = ':'.join(map(str,[cx,cy,cz,c]))
                if errors_marker_file is not None:
                    print(','.join(map(str,[1+cx,1+cy,1+cz,0,1,name,comment,r,g,b])), file=ostream)

    # Also print true TP in error marker file (helps debugging)
    for i,c in enumerate(C_true):
        if c in true_positives_true:
            if inside(c,substack):
                r,g,b = 0,255,255
                name = 'TP_%03d (%s)' % (i+1,c.name)
                cx,cy,cz = int(round(c.x)),int(round(c.y)),int(round(c.z))
                comment = ':'.join(map(str,[cx,cy,cz,c]))
                if errors_marker_file is not None:
                    print(','.join(map(str,[1+cx,1+cy,1+cz,0,1,name,comment,r,g,b])), file=ostream)

    # Finally, print rejected predictions error marker file (to show the benefit of the filter)
    for i,c in enumerate(C_rejected):
        if inside(c,substack):
            r,g,b = 255,128,0
            name = 'REJ_%03d (%s)' % (i+1,c.name)
            cx,cy,cz = int(round(c.x)),int(round(c.y)),int(round(c.z))
            comment = ':'.join(map(str,[cx,cy,cz,c]))
            if errors_marker_file is not None:
                print(','.join(map(str,[1+cx,1+cy,1+cz,0,1,name,comment,r,g,b])), file=ostream)

    if errors_marker_file is not None:
        ostream.close()

    # This is for the recall-precision and ROC curves according to manifold distance
    if hasattr(C_pred[0],'distance') and rp_file is not None:
        with open(rp_file, 'w') as ostream:
            for i,c in enumerate(C_pred):
                if inside(c,substack):
                    if c in true_positives_pred:
                        print(-c.distance, '1', file=ostream)
                    else:
                        print(-c.distance, '0', file=ostream)
            # Add also the false negatives with infinite distance so they will always be rejected
            for i,c in enumerate(C_true):
                if c not in true_positives_true:
                    if inside(c,substack):
                        print(-1000, '1', file=ostream)

    if len(TP_inside) > 0:
        precision = float(len(TP_inside))/float(len(TP_inside)+len(FP_inside))
        recall = float(len(TP_inside))/float(len(TP_inside)+len(FN_inside))
    else:
        precision = int(len(FP_inside) == 0)
        recall = 1.0
    F1 = 2*precision*recall/(precision+recall)

    C_pred_inside = [c for c in C_pred if inside(c,substack)]
    C_true_inside = [c for c in C_true if inside(c,substack)]
    print('|pred|=%d |true|=%d  P: %.2f / R: %.2f / F1: %.2f ==== TP: %d / FP: %d / FN: %d' % (len(C_pred_inside),len(C_true_inside),precision*100,recall*100,F1*100,len(TP_inside),len(FP_inside),len(FN_inside)))

    return precision,recall,F1,TP_inside,FP_inside,FN_inside


def rp_curve_on_attr(attr,substack,C_true,C_pred,subdir):
    values = [v for v,_ in itertools.groupby(sorted([c.__dict__[attr] for c in C_pred if inside(c,substack)]))]
    curve = []
    best_F1 = 0
    best_precision = 0
    best_recall = 0
    best_value = None
    for v in values:
        print(attr,v)
        C_pred1 = [c for c in C_pred if c.__dict__[attr] > v]
        if len([c for c in C_pred1 if inside(c,substack)]) > 10:
            precision,recall,F1,TP_inside,FP_inside,FN_inside = eval_perf(substack,C_true,C_pred1,verbose=False)
            curve.append([v,len(TP_inside),precision,recall,F1])
            if F1 > best_F1:
                best_F1 = F1
                best_precision = precision
                best_recall = recall
                best_value = v
    print('Tried all values, best is',best_value)
    C_pred1 = [c for c in C_pred if c.__dict__[attr] > best_value]
    eval_perf(substack,C_true,C_pred1,verbose=False,
              errors_marker_file=subdir+'/errors_at_best_%s_threshold.marker' % attr)
    ostream = open(subdir+'/'+attr+'.rp','w')
    print('#best P/R/F1:\t%.2f\t%.2f\t%.2f' % (best_precision*100,best_recall*100,best_F1*100),file=ostream)
    print('#thresh\t#TP\tprec\trec\tF1',file=ostream)
    print('\n'.join(['\t'.join(map(str,a)) for a in curve]),file=ostream)
    ostream.close()

    # try to predict by max entropy
    if len(values) < 3:
        maxe_prec,maxe_rec,maxe_F1 = 0,0,0
    else:
        maxe = max_entropy(values)
        print('best '+attr+':',best_value,'max entropy '+attr+':',maxe)
        C_pred1 = [c for c in C_pred if c.__dict__[attr] > maxe]
        if len([c for c in C_pred1 if inside(c,substack)]) > 10:
            maxe_prec,maxe_rec,maxe_F1,TP_inside,FP_inside,FN_inside = eval_perf(substack,C_true,C_pred1,verbose=False)
        else:
            maxe_prec,maxe_rec,maxe_F1 = 0,0,0
    return ([c[2] for c in curve],[c[3] for c in curve],attr,best_value,best_precision,best_recall,best_F1,maxe_prec,maxe_rec,maxe_F1)


def recall_precision_curve(substack,C_true,C_pred,subdir):
    rpdata = []
    # rpdata.append(rp_curve_on_attr('volume',substack,C_true,C_pred,subdir))
    rpdata.append(rp_curve_on_attr('mass',substack,C_true,C_pred,subdir))
    # if hasattr(C_pred[0],'EVR'):
    #     rpdata.append(rp_curve_on_attr('last_variance',substack,C_true,C_pred,subdir))
    # if hasattr(C_pred[0],'radius'):
    #     rpdata.append(rp_curve_on_attr('radius',substack,C_true,C_pred,subdir))
    rpcurves(rpdata,'Substack: %s (|GT|=%d, |Pred|=%d)' % (substack.substack_id,len(C_true),len(C_pred)),subdir)
    return rpdata


def max_entropy(data):
    from bcfind.threshold import multi_kapur
    N = 256
    histogram = [0]*N
    Min = min(data)
    Max = max(data)
    norm_data = [(d-Min)/(Max-Min) for d in data]
    for d in norm_data:
        histogram[int(d*(N-1))] += 1
    t = multi_kapur(histogram,2)[0]
    return Min+(t/float(N-1))*(Max-Min)


def rpcurves(rpdata,title,subdir,subplot=111):
    if len(rpdata[0][0]) == 0:
        return  # method does not support RP curve
    fig = pylab.figure()
    ax = fig.add_subplot(subplot)
    fig.subplots_adjust(hspace=0.6)
    ax.grid(True)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    min_rp = 0.9
    for i,(prec,rec,desc,best_th,best_precision,best_recall,best_F1,maxe_prec,maxe_rec,maxe_F1) in enumerate(rpdata):
        if best_th is not None:
            tit = '%s ($t^\star$=%.2f, $F_1^\star$=%.2f)' % (desc,best_th,best_F1*100)
        else:
            tit = '%s ($t^\star$=None, $F_1^\star$=None)' % (desc)
        ax.plot(rec, prec, label=tit)
#       ax.plot([maxe_rec], [maxe_prec], 'o', label='%s (maxEntropy: $F_1^{ME}$=%.2f)'%(desc,maxe_F1*100))
        min_rp = min(min_rp,min(rec))
        min_rp = min(min_rp,min(prec))
    min_rp = max(min_rp,0.6)
    ax.plot([min_rp,1.1],[min_rp,1.1],ls='dashed',label='breakeven line')
    ax.legend(loc='lower left')
    ax.set_title(title)
    # fig.show()
    pylab.savefig(subdir+'/rp.png',dpi=300)


def rp(prec,rec,title,subplot=111):
    fig = pylab.figure()
    ax = fig.add_subplot(subplot)
    fig.subplots_adjust(hspace=0.6)
    ax.plot(rec, prec, c='blue')
    ax.plot([0.6,1.1],[0.6,1.1],c='red')
    ax.grid(True)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    pylab.legend(loc='upper right')
    ax.set_title(title)
    fig.show()


def scatter3(C,title):
    fig = pylab.figure()
    ax = fig.add_subplot(3,1,1)
    fig.subplots_adjust(hspace=0.6)
    ax.plot([c.volume for c in C],[c.mass for c in C], 'bo')
    ax.grid(True)
    ax.set_xlabel('Volume')
    ax.set_ylabel('Mass')
    ax.set_title(title)

    ax = fig.add_subplot(3,1,2)
    ax.plot([c.volume for c in C if hasattr(c,'EVR')],[c.EVR[2] for c in C if hasattr(c,'EVR')], 'ro')
    ax.grid(True)
    ax.set_xlabel('Volume')
    ax.set_ylabel('EVR[2]')
    ax.set_title(title)

    ax = fig.add_subplot(3,1,3)
    ax.plot([c.mass for c in C if hasattr(c,'EVR')],[c.EVR[2] for c in C if hasattr(c,'EVR')], 'ro')
    ax.grid(True)
    ax.set_xlabel('Mass')
    ax.set_ylabel('EVR[2]')
    ax.set_title(title)

    fig.show()


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str,
                        help="""needs indir/info.plist, substacks, e.g. indir/100905,
                        and GT files e.g. indir/100905-GT.marker (unless a different folder
                        is specified with the --ground_truth_folder option)""")
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='substack identifier, e.g. 100905')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help="""where prediction results were saved, e.g. outdir/100905/ms.marker.
                        Errors are saved in outdir/100905/errors.marker
                        """)
    parser.add_argument('-D', '--max_cell_diameter', dest='max_cell_diameter', type=float, default=16.0,
                        help='Maximum diameter of a cell')
    parser.add_argument('-d', '--manifold-distance', dest='manifold_distance', type=float, default=None,
                        help='Maximum distance from estimated manifold to be included as a prediction')
    parser.add_argument('-c', '--curve', dest='curve', action='store_true', help='Make a recall-precision curve.')
    parser.add_argument('-g', '--ground_truth_folder', dest='ground_truth_folder', type=str, default=None,
                        help='folder containing merged marker files (for multiview images)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Verbose output.')
    return parser


def main(args):
    substack = SubStack(args.indir,args.substack_id)

    if args.ground_truth_folder:
        gt_markers = args.ground_truth_folder+'/'+args.substack_id+'-GT.marker'
    else:
        gt_markers = args.indir+'/'+args.substack_id+'-GT.marker'

    print('Loading ground truth markers from',gt_markers)
    try:
        C_true = substack.load_markers(gt_markers,from_vaa3d=True)
    except IOError:
        print('Ground truth file',gt_markers,'not found. Bailing out')
        sys.exit(1)
    pred_markers = args.outdir+'/'+args.substack_id+'/ms.marker'
    print('Loading predicted markers from',pred_markers)
    try:
        C_pred = substack.load_markers(pred_markers,from_vaa3d=False)
    except IOError:
        print('Warning: prediction marker file',pred_markers,'not found. Assuming empty volume')
        C_pred = []
    if args.manifold_distance:
        try:
            for c in C_pred:
                c.rejected = c.distance >= args.manifold_distance
        except AttributeError:
            print('You specified a manifold distance',args.manifold_distance,'however markers file',
                  pred_markers, 'is not annotated with a distance column')
    else:
        for c in C_pred:
            c.rejected = False
    errors_marker_file = args.outdir+'/'+args.substack_id+'/errors.marker'
    rp_file = args.outdir+'/'+args.substack_id+'/curve.rp'
    precision,recall,F1,TP_inside,FP_inside,FN_inside = eval_perf(substack,C_true,C_pred,
                                                                  errors_marker_file=errors_marker_file,
                                                                  rp_file=rp_file,
                                                                  verbose=args.verbose,
                                                                  max_cell_diameter=args.max_cell_diameter)

    with open(args.outdir+'/'+args.substack_id+'/eval.log','w') as ostream:
        print('substack,method,parameter,precision,recall,F1,TP,FP,FN,|true|,|pred|',file=ostream)
        print(','.join(map(str,([args.substack_id,'unfiltered',0,precision,recall,F1,
                                 repr(len(TP_inside)),repr(len(FP_inside)),
                                 repr(len(FN_inside)),repr(len(TP_inside)+len(FN_inside)),
                                 repr(len(TP_inside)+len(FP_inside))]))),file=ostream)
        # scatter3(TP_inside,'TP inside')
        # scatter3(FP_inside,'FP inside')
        # scatter3(FN_inside,'FN inside')

        if not args.curve:
            return
        rpdata = recall_precision_curve(substack,C_true,C_pred,args.outdir+'/'+args.substack_id)

        #Uncomment here for filtermass and max_entropymass
        #for (prec,rec,desc,best_th,best_precision,best_recall,best_F1,maxe_prec,maxe_rec,maxe_F1) in rpdata:
        #    print(','.join(map(str,(['filter'+desc,best_th,best_precision,best_recall,best_F1]))),file=ostream)
        #    print(','.join(map(str,(['max_entropy'+desc,0,maxe_prec,maxe_rec,maxe_F1]))),file=ostream)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
