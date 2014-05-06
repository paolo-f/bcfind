from __future__ import print_function
import os,sys,math
import itertools


from volume import *

def distance((x1,y1,z1),(x2,y2,z2)):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def distances(C_true):
    d=[]
    for i,c1 in enumerate(C_true):
        for j,c2 in enumerate(C_true):
            if i<j:
                d.append(distance((c1.x,c1.y,c1.z),(c2.x,c2.y,c2.z)))
    # d.sort()
    # for i in xrange(len(C_true)):
    #     print('%3d %3.2f'%(i,d[i]))
    import pylab
    pylab.figure()
    # pylab.hist(d, 50, normed=1, histtype='stepfilled')

    pylab.hist(d,[10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22])
    pylab.show()

def usage():
    print('Usage:       %s indir substack_id, where:' % sys.argv[0])
    print('indir:       needs indir/info.json, substacks, e.g. indir/000')
    print('             and ground truth as indir/000-GT.marker')
    print('substack_id: substack identifier, e.g. 000')
if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        sys.exit(2)
    substack=SubStack(sys.argv[1],sys.argv[2])
    gt_markers=sys.argv[1]+'/'+sys.argv[2]+'-GT.marker'
    print('Loading ground truth markers from',gt_markers)
    C_true=substack.load_markers(gt_markers,from_vaa3d=True)
    distances(C_true)
