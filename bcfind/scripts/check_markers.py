from __future__ import print_function
import sys

from volume import *

def check_markers(substack,marker_file):
    C=substack.load_markers(marker_file,from_vaa3d=True)
    d = substack.distance_matrix(C)
    print('\n'.join(map(str,[t for t in d])))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise Exception('Usage: %s indir substack_id marker_file' % sys.argv[0])
    substack=SubStack(sys.argv[1],sys.argv[2])
    check_markers(substack,sys.argv[3])


