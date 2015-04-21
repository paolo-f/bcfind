#!/usr/bin/env python
"""
Scripts that fuse two marker files corresponding to the same substack
"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import argparse

from bcfind.markers import distance, match_markers
from bcfind.volume import m_load_markers
from bcfind.utils import mkdir_p


def do_fuse(C_firstview,C_secondview,max_distance,output_marker_file,first_view_id,second_view_id,verbose):

    # ============ max-cardinality bipartite matching

    true_positives_firstview = set()  # subset of C_true that are true positives
    true_positives_secondview = set()  # subset of C_pred that are true positives
    TP = 0
    TP_list = []
    G,mate,node2center = match_markers(C_firstview,C_secondview,max_distance*2)
    # Debug
    # for k1,k2 in mate.iteritems():
    #     c1=node2center[k1]
    #     c2=node2center[k2]
    #     print(c1.name,k1,[c1.x,c1.y,c1.z],c2.name,k2,[c2.x,c2.y,c2.z],G.get_edge_data(k1,k2))
    # crepa
    # end debug
    kw = 0

    merged_centers = []
    for k1,k2 in mate.iteritems():
        if k1[0] == 'p':  # mate is symmetric
            continue
        c1 = node2center[k1]
        c2 = node2center[k2]
        d = distance((c1.x,c1.y,c1.z),(c2.x,c2.y,c2.z))
        # print(k1,c1.name,c1.x,c1.y,c1.z,k2,c2.name,c2.x,c2.y,c2.z,d,end='')
        if d < (max_distance):  # a constant criterion is needed!

            true_positives_firstview.add(c1)
            true_positives_secondview.add(c2)

            c_merged = c1
            c_merged.x = (c_merged.x + c2.x)/2
            c_merged.y = (c_merged.y + c2.y)/2
            c_merged.z = (c_merged.z + c2.z)/2

            merged_centers.append(c_merged)

            TP += 1
            if verbose:
                print('BOTH-VIEWS:', k2, c2.name,c2.x,c2.y,c2.z,c2, k1, c1.name,c1.x,c1.y,c1.z,d)
            TP_list.append(c2)
            kw += 1
        else:
            if verbose:
                print('---> too far', c2.name,c2.x,c2.y,c2.z,c2,c1.name,c1.x,c1.y,c1.z,d)

    FP_list,FN_list = [],[]
    if output_marker_file is not None:
        ostream = open(output_marker_file,'w')
        print('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b',file=ostream)

    for i,c in enumerate(C_firstview):
        if c not in true_positives_firstview:
            r,g,b = 255,0,255
            #name = 'first_view_marker_%03d (%s)' % (i,c.name)
            name = first_view_id+'_single('+c.name+')'
            #cx,cy,cz=int(round(c.x)),int(round(c.y)),int(round(c.z))
            cx,cy,cz = c.x,c.y,c.z
            #comment=':'.join(map(str,[cx,cy,cz,c]))
            comment = ''
            if output_marker_file is not None:
                print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)
            FN_list.append(c)
            if verbose:
                print('ONLY-FIRST-VIEW: ', c.name,c.x,c.y,c.z,c)

    for i,c in enumerate(C_secondview):
        c.is_false_positive = False
        if c not in true_positives_secondview:
            r,g,b = 255,0,0
            #name = 'second_view_marker_%03d (%s)' % (i,c.name)
            name = second_view_id+'_single('+c.name+')'
            c.is_false_positive = True
            #cx,cy,cz=int(round(c.x)),int(round(c.y)),int(round(c.z))
            cx,cy,cz = c.x,c.y,c.z
            #comment=':'.join(map(str,[cx,cy,cz,c]))
            comment = ''
            if output_marker_file is not None:
                #print(','.join(map(str,[1+cx,1+cy,1+cz,0,1,name,comment,r,g,b])), file=ostream)
                print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)
            FP_list.append(c)
            if verbose:
                print('ONLY-SECOND-VIEW: ', c.name,c.x,c.y,c.z,c)

    for i,c in enumerate(merged_centers):
        r,g,b = 255,255,0
        #name = 'merged_marker_%03d (%s)' % (i,c.name)
        name = first_view_id+'_'+second_view_id+'_merged_'+str(i)#('+c.name+')'
        cx,cy,cz = c.x,c.y,c.z
        #comment=':'.join(map(str,[cx,cy,cz,c]))
        comment = ''
        if output_marker_file is not None:
            print(','.join(map(str,[cx,cy,cz,0,1,name,comment,r,g,b])), file=ostream)

    if output_marker_file is not None:
        ostream.close()


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('first_view', metavar='first_view', type=str,
                        help='Markers of the 1st view')
    parser.add_argument('second_view', metavar='second_view', type=str,
                        help='Markers of the 2nd view')
    parser.add_argument('fused_file', metavar='fused_file', type=str,
                        help='Output file of fused markers')

    parser.add_argument('--first_view_id', metavar='first_view_id', type=str, action='store', default='first_view_marker_',
                        help='id of the 1st view')
    parser.add_argument('--second_view_id', metavar='second_view_id', type=str, action='store', default='second_view_marker_',
                        help='id of the 2nd view')

    parser.add_argument('--max_distance', metavar='max_distance', dest='max_distance',
                        action='store', type=float, default=2.0,
                        help='maximum distance beyond which two neurons of different views are no longer considered the same element')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')

    return parser


def main(args):
    try:
        C_firstmarkers = m_load_markers(args.first_view,from_vaa3d=True)
    except IOError:
        print('Warning: first view marker file',args.first_view,'not found.')
        C_firstmarkers = []
    try:
        C_secondmarkers = m_load_markers(args.second_view,from_vaa3d=True)
    except IOError:
        print('Warning: second view marker file',args.second_view,'not found.')
        C_secondmarkers = []

    mkdir_p(os.path.dirname(args.fused_file))
    do_fuse(C_firstmarkers,C_secondmarkers,args.max_distance, args.fused_file,args.first_view_id,args.second_view_id,args.verbose)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
