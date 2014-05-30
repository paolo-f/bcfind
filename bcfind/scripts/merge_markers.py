#!/usr/bin/env python
"""
Merges marker files from several substacks into a single file.
"""
from __future__ import print_function
import sys,os,string,copy
import argparse

from bcfind.volume import SubStack

from scipy.spatial import cKDTree
import numpy as np

def inside_margin(c,substack):
    m=substack.plist['Margin']/2
    return min(c.x-m,c.y-m,c.z-m,substack.info['Width']-m-c.x,substack.info['Height']-m-c.y,substack.info['Depth']-m-c.z)

def merge(C_final,C,substack, hue, verbose=False):
    # if this is the first substack no need to check close markers
    if len(C_final)>0:
        if verbose:
            print('Making KD-tree for', len(C_final), 'previous points')
        data = np.array([[c.x,c.y,c.z] for c in C_final])
        kdtree=cKDTree(data)
    else:
        kdtree = None
    C_prev = [c for c in C_final]
    if verbose:
        print('Processing', len(C), 'new points')
    for c in C:
        if (c.x>substack.info['Width'] or c.y>substack.info['Height'] or c.z>substack.info['Depth']):
            raise Exception('Coordinates in marker file are out of range',(c.x,c.y,c.z),substack.substack_id)
        if inside_margin(c,substack)<0:
            continue
        if c.name.find('(landmark') >=0:
            continue
        c1 = copy.deepcopy(c)
        c1.x,c1.y,c1.z = c.x+substack.info['X0'],c.y+substack.info['Y0'],c.z+substack.info['Z0']
        c1.name = c.name+'('+substack.substack_id+')'
        c1.inside_margin = inside_margin(c,substack)
        # c1.hue = min(1.0, float(len(C))/1000.0)
        c1.hue = hue
        C_final.add(c1)
        if kdtree is not None:
            d,index = kdtree.query(np.array([c1.x,c1.y,c1.z]), k=1)
            # d = math.sqrt((n.point[0]-c1.x)**2 + (n.point[1]-c1.y)**2 + (n.point[2]-c1.z)**2)
            n = C_prev[index]
            if d<8:
                if verbose:
                    print('mmh.., near duplicate',(c1.x,c1.y,c1.z,c1.inside_margin),
                          'of',(n.x,n.y,n.z,n.inside_margin))
                if c1.inside_margin < n.inside_margin:
                    if verbose:
                        print('Removing first')
                    C_final.remove(c1)
                else:
                    if n in C_final:
                        if verbose:
                            print('Removing second')
                        C_final.remove(n)
                    else:
                        if verbose:
                            print('n not in C_final')

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/100905, and GT files e.g. indir/100905-GT.marker')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help="""where prediction results were saved, e.g. outdir/100905/ms.marker.
                        As a special case, if outdir=="GT" then ground truth files are merged
                        """)
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='where to save the merged .marker file, e.g. merged.marker')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.set_defaults(verbose=False)
    return parser

def main(args):
    if args.outdir=='GT':
        marker_files = map(string.strip,os.popen('ls '+args.indir+'*-GT.marker').readlines())
        substack_ids = [f.split('/')[-1].split('-')[0] for f in marker_files]
    else:
        marker_files = map(string.strip,os.popen('ls '+args.outdir+'/*/ms.marker').readlines())
        substack_ids = [f.split('/')[-2] for f in marker_files]

    C_final=set()
    plist = None
    hue = 0.0
    # FIXME: This is way too slow, should use divide & conquer using a binary tree
    for marker_file,substack_id in zip(marker_files,substack_ids):
        substack=SubStack(args.indir,substack_id, plist)
        plist = substack.plist
        print('Merging', marker_file, substack_id)
        C=substack.load_markers(marker_file,args.outdir=='GT')
        # I had forgotten that markers are shifted by 1 in save_markers()
        # that was because Vaa3D coordinates starts from 1 rather than 0
        # ==> repair original values
        for c in C:
            c.x -= 1
            c.y -= 1
            c.z -= 1
        hue = hue + 0.31
        if hue > 1:
            hue = hue -1
        merge(C_final, C, substack, hue, args.verbose)
    substack.save_markers(args.outfile, C_final)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

