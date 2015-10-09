#!/usr/bin/env python
"""
Script that fuses marker files of different aligned views of the same substack
"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import argparse

from bcfind.markers import distance, match_markers, match_markers_with_icp
from bcfind.volume import m_load_markers,SubStack
from bcfind.utils import mkdir_p

def save_fused_markers(substack, C_merged, C_onlyfirstview, C_onlysecondview, output_marker_file, first_view_id, second_view_id, verbose=False):
    
    if output_marker_file is not None:
        ostream = open(output_marker_file,'w')
        print('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b',file=ostream)

    for i,c in enumerate(C_onlyfirstview):
            r,g,b = 255,0,255
            name = first_view_id+'_single('+c.name+')'
            cx,cy,cz = c.x,c.y,c.z
            comment = ''
            if output_marker_file is not None:
                print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)
            if verbose:
                print('ONLY-FIRST-VIEW: ', c.name,c.x,c.y,c.z,c)

    for i,c in enumerate(C_onlysecondview):
            r,g,b = 255,0,0
            name = second_view_id+'_single('+c.name+')'
            cx,cy,cz = c.x,c.y,c.z
            comment = ''
            if output_marker_file is not None:
                print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)
            if verbose:
                print('ONLY-SECOND-VIEW: ', c.name,c.x,c.y,c.z,c)

    for i,c in enumerate(C_merged):
        r,g,b = 255,255,0
        name = first_view_id+'_'+second_view_id+'_merged_'+str(i)
        cx,cy,cz = c.x,c.y,c.z
        comment = ''
        if output_marker_file is not None:
            print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)

    if output_marker_file is not None:
        ostream.close()

def do_fuse_with_icp(substack,C_firstview,C_secondview,max_distance,match_distance=None,num_iterations = 100, eps=1e-8, verbose=False):

    if match_distance is None:
        match_distance = max_distance

    if len(C_firstview)==0 or len(C_secondview)==0:
        if verbose:
            print('total=%d merged=%d only_firstview=%d only_secondview=%d'%(len(C_firstview+C_secondview),len([]),len(C_firstview),len(C_secondview)))
        return [],C_firstview,C_secondview,C_firstview+C_secondview

    C_secondview,good_firstview,good_secondview,_,_ = match_markers_with_icp(C_firstview,C_secondview, match_distance, num_iterations, eps) 

    c_firstview_matched = []
    c_secondview_matched = []
    for gi,gj in zip(good_firstview,good_secondview):
        c1=C_firstview[gi]
        c2=C_secondview[gj]
        d = distance((c1.x,c1.y,c1.z),(c2.x,c2.y,c2.z))
        if d < max_distance:
            c_firstview_matched.append(c1)
            c_secondview_matched.append(c2)
    true_positives_firstview = set(c_firstview_matched) 
    true_positives_secondview = set(c_secondview_matched)
    
    C_merged = []
    for c1,c2 in zip(c_firstview_matched,c_secondview_matched):
        c_merged = c1
        c_merged.x = (c_merged.x + c2.x)/2
        c_merged.y = (c_merged.y + c2.y)/2
        c_merged.z = (c_merged.z + c2.z)/2
        if c_merged.x<0 or c_merged.y<0 or c_merged.z<0 or c_merged.x>substack.info['Width'] or c_merged.y>substack.info['Height'] or c_merged.z>substack.info['Depth']:
            continue
        C_merged.append(c_merged)


    C_onlyfirstview=[]
    for i,c in enumerate(C_firstview):
        if c not in true_positives_firstview:
            cx,cy,cz = c.x,c.y,c.z
            if cx<0 or cy<0 or cz<0 or cx>substack.info['Width'] or cy>substack.info['Height'] or cz>substack.info['Depth']:
                continue
            else:
                C_onlyfirstview.append(c)

    C_onlysecondview=[]
    for i,c in enumerate(C_secondview):
        if c not in true_positives_secondview:
            cx,cy,cz = c.x,c.y,c.z
            if cx<0 or cy<0 or cz<0 or cx>substack.info['Width'] or cy>substack.info['Height'] or cz>substack.info['Depth']:
                continue
            else:
                C_onlysecondview.append(c)

    C_total=C_merged+C_onlyfirstview+C_onlysecondview
    if verbose:
        print('total=%d merged=%d only_firstview=%d only_secondview=%d'%(len(C_total),len(C_merged),len(C_onlyfirstview),len(C_onlysecondview)))

    return C_merged,C_onlyfirstview,C_onlysecondview,C_total




def do_fuse(substack,C_firstview,C_secondview,max_distance,verbose=False):

    # ============ max-cardinality bipartite matching
    if len(C_firstview)==0 or len(C_secondview)==0:
        if verbose:
            print('total=%d merged=%d only_firstview=%d only_secondview=%d'%(len(C_firstview+C_secondview),len([]),len(C_firstview),len(C_secondview)))
        return [],C_firstview,C_secondview,C_firstview+C_secondview

    true_positives_firstview = set()  # subset of C_true that are true positives
    true_positives_secondview = set()  # subset of C_pred that are true positives
    G,mate,node2center = match_markers(C_firstview,C_secondview,max_distance*2)
    C_merged = []
    for k1,k2 in mate.iteritems():
        if k1[0] == 'p':  # mate is symmetric
            continue
        c1 = node2center[k1]
        c2 = node2center[k2]
        d = distance((c1.x,c1.y,c1.z),(c2.x,c2.y,c2.z))
        if d < (max_distance):  # a constant criterion is needed!

            true_positives_firstview.add(c1)
            true_positives_secondview.add(c2)

            c_merged = c1
            c_merged.x = (c_merged.x + c2.x)/2
            c_merged.y = (c_merged.y + c2.y)/2
            c_merged.z = (c_merged.z + c2.z)/2

            if c_merged.x<0 or c_merged.y<0 or c_merged.z<0 or c_merged.x>substack.info['Width'] or c_merged.y>substack.info['Height'] or c_merged.z>substack.info['Depth']:
                continue
            C_merged.append(c_merged)
    
    C_onlyfirstview=[]
    for i,c in enumerate(C_firstview):
        if c not in true_positives_firstview:
            cx,cy,cz = c.x,c.y,c.z
            if cx<0 or cy<0 or cz<0 or cx>substack.info['Width'] or cy>substack.info['Height'] or cz>substack.info['Depth']:
                continue
            else:
                C_onlyfirstview.append(c)

    C_onlysecondview=[]
    for i,c in enumerate(C_secondview):
        if c not in true_positives_secondview:
            cx,cy,cz = c.x,c.y,c.z
            if cx<0 or cy<0 or cz<0 or cx>substack.info['Width'] or cy>substack.info['Height'] or cz>substack.info['Depth']:
                continue
            else:
                C_onlysecondview.append(c)

    C_total=C_merged+C_onlyfirstview+C_onlysecondview
    if verbose:
        print('total=%d merged=%d only_firstview=%d only_secondview=%d'%(len(C_total),len(C_merged),len(C_onlyfirstview),len(C_onlysecondview)))

    return C_merged,C_onlyfirstview,C_onlysecondview,C_total


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str,
                        help="""needs indir/info.plist, substacks, e.g. indir/100905,
                        and GT files e.g. indir/100905-GT.marker (unless a different folder
                        is specified with the --ground_truth_folder option)""")
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='substack identifier, e.g. 100905')
    parser.add_argument('first_view', metavar='first_view', type=str,
                        help='Markers of the 1st view')
    parser.add_argument('second_view', metavar='second_view', type=str,
                        help='Markers of the 2nd view')
    parser.add_argument('output_marker_file', metavar='output_marker_file', type=str,
                        help='Output file of fused markers')

    parser.add_argument('--first_view_id', metavar='first_view_id', type=str, action='store', default='first_view_marker_',
                        help='id of the 1st view')
    parser.add_argument('--second_view_id', metavar='second_view_id', type=str, action='store', default='second_view_marker_',
                        help='id of the 2nd view')

    parser.add_argument('--max_distance', metavar='max_distance', dest='max_distance',
                        action='store', type=float, default=2.0,
                        help='maximum distance beyond which two neurons of different views are no longer considered the same element')
    parser.add_argument('--match_distance', metavar='match_distance', dest='match_distance',
                        action='store', type=float, default=2.0,
                        help='maximum distance beyond which a pair of markers is matched in the Iterative Closest Point procedure')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.add_argument('--do_icp', dest='do_icp', action='store_true', help='do the Iterative Closest point procedure to align the second view markers to the first view markers')

    return parser


def main(args):
    try:
        C_firstview = m_load_markers(args.first_view,from_vaa3d=True)
    except IOError:
        print('Warning: first view marker file',args.first_view,'not found.')
        C_firstview = []
    try:
        C_secondview = m_load_markers(args.second_view,from_vaa3d=True)
    except IOError:
        print('Warning: second view marker file',args.second_view,'not found.')
        C_secondview = []

    mkdir_p(os.path.dirname(args.output_marker_file))
    substack = SubStack(args.indir,args.substack_id)
    if args.do_icp:
        C_merged, C_onlyfirstview, C_onlyfirstview, _ = do_fuse_with_icp(substack,C_firstview,C_secondview,args.max_distance,match_distance=args.match_distance,verbose=args.verbose)
    else:
        C_merged, C_onlyfirstview, C_onlyfirstview, _ = do_fuse(substack,C_firstview,C_secondview,args.max_distance, args.verbose)
    
    save_fused_markers(substack,C_merged,C_onlyfirstview,C_onlysecondview,output_marker_file,first_view_id,second_view_id,verbose)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
