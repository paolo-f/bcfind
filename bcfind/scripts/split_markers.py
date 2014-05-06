#!/usr/bin/env python
"""Splits a large marker file into substack-based marker
files. Relies on the substacks arrangments described into an info.json
file and point coordinates.  The same point will endup in different
local .marker files if within the overlapping region.

run split_markers.py ~/BTMP/fake/ ~/mmm.csv ~/BTMP/SPLIT/
"""
from __future__ import print_function
import sys,os,string,copy

from bcfind.volume import *
from bcfind.utils import mkdir_p
import banyan as bn
from progressbar import *

def atoi(s):
    return int(float(s))

def load_markers(filename):
    istream = open(filename)
    C = []
    for line in istream:
        if line[0] != '#':
            items = line.strip().split(',')
            # -1 to repair coordinates
            x, y, z = atoi(items[0])-1, atoi(items[1])-1,  atoi(items[2])-1
            c = Center(x, y, z)
            c.name, comment = items[5], items[6]
            items = comment.split('\t')
            c.mass = float(items[0].split('=')[1])
            c.volume = float(items[1].split('=')[1])
            c.radius = float(items[2].split('=')[1])
            c.hue = float(items[3].split('=')[1])
            C.append(c)
    return C

def build_trees(substacks):
    tx = bn.SortedSet(key_type = (int, int), updator = bn.OverlappingIntervalsUpdator)
    ty = bn.SortedSet(key_type = (int, int), updator = bn.OverlappingIntervalsUpdator)
    tz = bn.SortedSet(key_type = (int, int), updator = bn.OverlappingIntervalsUpdator)
    idx = {}
    for s in substacks.values():
        xx = (s.info['X0'], s.info['X0']+s.info['Width']-1)
        tx.add(xx)
        yy = (s.info['Y0'], s.info['Y0']+s.info['Height']-1)
        ty.add(yy)
        zz = (s.info['Z0'], s.info['Z0']+s.info['Depth']-1)
        tz.add(zz)
        idx[(xx,yy,zz)] = s.substack_id
    return tx, ty, tz, idx

def get_locals(C, tx, ty, tz, idx, substacks):
    C_locals = {}
    pbar = ProgressBar(widgets=['Splitting %d points: ' % len(C), Percentage()],
                       maxval=len(C)).start()
    for pbi,c in enumerate(C):
        ranges_x = tx.overlap_point(c.x)
        ranges_y = ty.overlap_point(c.y)
        ranges_z = tz.overlap_point(c.z)
        sids = {idx[(rx,ry,rz)] for rx in ranges_x for ry in ranges_y for rz in ranges_z}
        # print(len(sids), c.x,c.y,c.z, sids)
        for substack_id in sids:
            if substack_id not in C_locals:
                C_locals[substack_id] = []
            c1 = copy.deepcopy(c)
            c1.x -= substacks[substack_id].info['X0']
            c1.y -= substacks[substack_id].info['Y0']
            c1.z -= substacks[substack_id].info['Z0']
            C_locals[substack_id].append(c1)
        pbar.update(pbi+1)
    pbar.finish()
    return C_locals

def usage():
    print('Usage:       %s indir infile outdir, where:' % sys.argv[0])
    print('indir:       needs indir/info.json and substack dirs, e.g. indir/000')
    print('infile:      file containing a set of markers spanning several substacks')
    print('outdir:      where to save the split .marker files, e.g. outdir/000/ms.marker')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        sys.exit(2)
    indir = sys.argv[1]
    infile = sys.argv[2]
    outdir = sys.argv[3]

    substack_ids = map(string.strip,os.popen('ls '+indir).readlines())
    substack_ids = [f for f in substack_ids if len(f)==6]
    C = load_markers(infile)

    plist = None
    substacks = {}
    for substack_id in substack_ids:
        substack=SubStack(indir,substack_id, plist)
        plist = substack.plist
        substacks[substack.substack_id] = substack
    tx, ty, tz, idx = build_trees(substacks)
    C_locals = get_locals(C, tx, ty, tz, idx, substacks)

    pbar = ProgressBar(widgets=['Saving %d files: ' % len(C_locals), Percentage()],
                       maxval=len(C_locals)).start()
    for pbi, (substack_id, C_local) in enumerate(C_locals.iteritems()):
        mkdir_p(outdir + '/' + substack_id)
        substack.save_markers(outdir + '/' + substack_id + '/ms.marker',C_local)
        pbar.update(pbi+1)
    pbar.finish()


