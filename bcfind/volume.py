"""
Module defining various objects for annotated 3D images
"""
from __future__ import print_function
import operator
import os
import random
import math
import plistlib
import ujson
import tables
import pandas as pd
import numpy as np
from PIL import Image
import ImageDraw
import cPickle as pickle
from scipy.spatial import cKDTree

from bcfind import timer
from bcfind.log import tee
from bcfind.utils import mkdir_p, which
from bcfind import log


SHARE_DIR = os.path.dirname(log.__file__)+'/share'
hi2rgb = pickle.load(open(SHARE_DIR+'/hi2rgb.pickle', 'rb'))

save_vaa3d_timer = timer.Timer('Save Vaa3D')
save_markers_timer = timer.Timer('Save markers')

def m_load_markers(filename, from_vaa3d=False):
    data = pd.read_csv(filename, skipinitialspace=True, na_filter=False)
    if '#x' in data.keys():  # fix some Vaa3d garbage
        data.rename(columns={'#x': 'x'}, inplace=True)
    if '##x' in data.keys():  # fix some Vaa3d garbage
        data.rename(columns={'##x': 'x'}, inplace=True)
    C = []
    for i in data.index:
        row = data.ix[i]
        c = Center(0, 0, 0)
        for k in row.keys():
            setattr(c,k,row[k])
        if from_vaa3d:
            if c.name == '':
                c.name = 'landmark %d' % (i+1)
        else:  # from predictor..
            try:
                c.volume = float(c.comment.split('v=')[1].split()[0])
                c.mass = float(c.comment.split('m=')[1].split()[0])
                c.hue = float(c.comment.split('hue=')[1].split()[0])
                citems = c.comment.split('\t')
                if len(citems) > 5:
                    c.EVR = [float(x) for x in citems[5].split(':')]
                    if len(c.EVR) > 2:
                        c.last_variance = c.EVR[2]
                if len(citems) > 6:
                    try:
                        c.radius = float(citems[6])
                    except ValueError:
                        pass
            except IndexError:
                print('Warning: comment string unformatted (%s), is this really a predicted marker file?' % c.comment)
        C.append(c)
    return C


def a_load_markers(filename):
    data = pd.read_csv(filename, skipinitialspace=True, na_filter=False)
    C = []
    for i in data.index:
        row = data.ix[i]
        c = Center(0, 0, 0)
        for k in row.keys():
            setattr(c,k,row[k])
        C.append(c)
    return C


class Center(object):
    """Similar to voxel but coordinates are real- instead of integer-valued

    Attributes
    ----------
    hue : int
        used to colorize the intermediate debugging images

    mass : float
        sum of voxel intensities for all voxels that are assigned to this
        center and are close enough to the center. Used in various places.

    volume : float
        number of voxels assigned to this center
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.hue = 0
        self.reset()

    def reset(self):
        self.sum_x = 0        # accumulator for barycenter
        self.sum_y = 0        # accumulator for barycenter
        self.sum_z = 0        # accumulator for barycenter
        self.mass = 0.01      # sum of intensities in the cluster
        self.volume = 0       # number of voxels in the cluster

    def short_desc(self):
        s = 'm=%.2f v=%.2f' % (self.mass, self.volume)
        return s

    def __str__(self):
        if self.volume != 0:
            s = 'm=%.1f\tv=%.2f\tr=%.2f\thue=%.2f' % (self.mass,
                                                      self.volume,
                                                      float(self.mass)/float(self.volume),
                                                      self.hue)
        else:
            s = 'm=%.1f\tv=%.2f\tr=%.2f\thue=%.2f' % (self.mass,
                                                      self.volume,
                                                      0.0,
                                                      self.hue)
        if hasattr(self, 'distances'):
            s = s + '\t' + ':'.join(['%.1f' % d for d in self.distances])
        if hasattr(self, 'EVR'):
            s = s + '\t' + ':'.join(['%.2f' % d for d in self.EVR])
        if hasattr(self, 'radius'):
            s = s + '\t' + str(self.radius)
        if hasattr(self, 'curvature'):
            s = s + '\t' + self.curvature
        if hasattr(self, 'distance'):
            s = s + '\t%.2f' % self.distance
        return s


class ImageSaver(object):
    """Save debugging images

    Save a pictorial representation of center and voxel assignment as
    pseudo-color images. Constructor adds a field 'hue' to each
    center. Can also save the thresholding results.

    Parameters
    ----------
    outdir : str
        Directory where the data will be saved
    substack : obj
        :py:class:`bcfind.volume.SubStack` Substack object to be saved
    C : list
        List of :py:class:`bcfind.volume.Center` objects
    shuffle_hues : bool, optional
        randomize default hues for colorizing voxels in the output images

    """
    def __init__(self, outdir, substack, C, shuffle_hues=True):
        outdir = os.path.abspath(outdir)
        self.substack = substack
        self.savedir = outdir + '/' + substack.substack_id
        hues = [float(i)/len(C) for i in range(len(C))]
        if shuffle_hues:
            random.shuffle(hues)
        for i, c in enumerate(C):
            c.hue = hues[i]

    def compute_hues(self, C, n_neighbors=6):
        # colorize randomly
        for i, c in enumerate(C):
            c.index = i
        colors = {c.index: int(random.uniform(0, 255)) for c in C}
        if which('minizinc') is None:
            for c in C:
                c.hue = colors[c.index]/255.0
            return
        # If we have minizinc then create a knn graph and run a constraint program to colorize nicely
        edges = []
        X = np.array([[c.x, c.y, c.z] for c in C])
        kdtree = cKDTree(X)
        purkinje_radius = 16
        for c in C:
            distances, neighbors = kdtree.query([c.x, c.y, c.z], n_neighbors)
            for d,n in zip(distances,neighbors):
                if d < 3*purkinje_radius and c.index < C[n].index:
                    edges.append([c.index, C[n].index])

        if len(edges) < 1000:  # otherwise solving the constraint program might be too costly
            with open(self.savedir+'/'+'/edges.dzn', 'w') as ostream:
                print('n=%d;' % len(C), file=ostream)
                print('num_edges=%d;' % len(edges), file=ostream)
                print('E = array2d(1..num_edges, 1..2, [', file=ostream)
                print(','.join(map(str, [e[i] for e in edges for i in range(2)])), file=ostream)
                print(']);', file=ostream)
            tee.log('Running minizinc on', len(edges), 'edges')
            curdir = os.getcwd()
            os.chdir(SHARE_DIR)
            mzn_cmd = ('minizinc color.mzn -d %s/edges.dzn -o %s/mnz_sol.txt' % (self.savedir, self.savedir))
            tee.log(mzn_cmd)
            os.system(mzn_cmd)
            os.chdir(curdir)
            istream = open(self.savedir+'/mnz_sol.txt')
            colors = {}
            for line in istream.readlines():
                items = line.strip().split()
                if items[0] == 'c':
                    colors[int(items[1])] = int(items[2])
        for c in C:
            c.hue = colors[c.index]/255.0

    @save_vaa3d_timer.timed
    def save_vaa3d(self, C, Lx, Ly, Lz, Lcluster,
                   draw_centers=True,
                   colorize_voxels=True,
                   trajectories=None,
                   floating_point=False):
        """
        save_vaa3d(C, Lx, Ly, Lz, Lcluster,
                   draw_centers=True, colorize_voxels=True, trajectories=None)

        Dump a colored substack so that Vaa3D can be used to inspect clusters

        Parameters
        ----------

        C : list
            List of centers
        Lx, Ly, Lz : array-like
            Coordinates of foreground voxels
        Lcluster : array-like
            Cluster index of every foreground voxel
        draw_centers : bool, optional
            Should centers be drawn as small circles?
        colorize_voxels : bool, optional
            Should we paint voxels in colors?
        trajectories : list or None, optional
            If not None, draw mean shift trajectories for every seed
        floating_point: bool, optional
            If true, save coordinates in floating point, else round to int
        """
        out_imgs = []
        opixels = []
        pixels = []
        self.compute_hues(C)
        for z, img in enumerate(self.substack.imgs):
            out_img = img.convert('RGB')
            out_imgs.append(out_img)
            opixels.append(out_img.load())
            pixels.append(img.load())

        if colorize_voxels:
            for i in xrange(len(Lx)):
                if Lcluster[i] is not None:
                    # draw = ImageDraw.Draw(out_imgs[Lz[i]])
                    opixels[Lz[i]][Lx[i], Ly[i]] = hi2rgb[int(255*Lcluster[i].hue)][pixels[Lz[i]][Lx[i], Ly[i]]]
        if draw_centers:
            for c in C:
                draw = ImageDraw.Draw(out_imgs[int(round(c.z))])
                rgb = hi2rgb[int(255*c.hue)][156]
                draw.ellipse((c.x-1, c.y-1, c.x+1, c.y+1), fill=rgb)
        if trajectories:
            for seed_no in trajectories:
                rgb = hi2rgb[random.randint(0, 255)][156]
                for mean in trajectories[seed_no]:
                    draw = ImageDraw.Draw(out_imgs[int(round(mean[2]))])
                    draw.point((mean[0], mean[1]), fill=rgb)
        save_name = 'last'
        iter_dir = '%s/%s' % (self.savedir, save_name)
        mkdir_p(iter_dir)
        for z, out_img in enumerate(out_imgs):
            out_img.save(iter_dir+'/'+self.substack.info['Files'][z].split('/')[-1])
            w, h = out_img.size
        # ---------- marker file used by vaa3d
        self.substack.save_markers('%s/%s.marker' % (self.savedir, save_name), C, floating_point)

    def save_above_threshold(self, Lx, Ly, Lz, thresholds=None):
        out_imgs = []
        opixels = []
        pixels = []
        for z, img in enumerate(self.substack.imgs):
            out_img = img.convert('RGB')
            out_imgs.append(out_img)
            opix = out_img.load()
            w, h = out_img.size
            for i in xrange(w):
                for j in xrange(h):
                    if thresholds:
                        opix[i, j] = (self.substack.pixels[z][i,j], 0, 0)  # Red
                    else:
                        opix[i, j] = 0
            opixels.append(opix)
            pixels.append(img.load())
        if thresholds:
            for i in xrange(len(Lx)):
                val = pixels[Lz[i]][Lx[i], Ly[i]]
                if val < thresholds[1]:  # Note: since it's on the lists Lx,Ly,Lz, it's already above thresholds[0]!
                    opixels[Lz[i]][Lx[i], Ly[i]] = (0,val,0)  # Green
                else:
                    opixels[Lz[i]][Lx[i], Ly[i]] = (val,val,0)  # Yellow
        else:
            for i in xrange(len(Lx)):
                opixels[Lz[i]][Lx[i], Ly[i]] = (pixels[Lz[i]][Lx[i], Ly[i]],
                                                pixels[Lz[i]][Lx[i], Ly[i]],
                                                pixels[Lz[i]][Lx[i], Ly[i]])

        save_name = 'above_threshold'
        iter_dir = '%s/%s' % (self.savedir, save_name)
        mkdir_p(iter_dir)
        for z, out_img in enumerate(out_imgs):
            out_img.save(iter_dir+'/'+self.substack.info['Files'][z].split('/')[-1])
            w, h = out_img.size


def valid_suffix(f):
    suffixes = ['.tif', '.tiff', '.jpg']
    for suffix in suffixes:
        if f.endswith(suffix):
            return True
    return False


class SubStack(object):
    """
    A SubStack object contains pixels and metadata associated with a
    portion of a larger volume.

    Large volumes are assumed to be split into several substacks. They
    are organized in a directory structure as follows:

    $INDIR/info.json (or $INDIR/info.plist, but json is faster)
    $INDIR/substacks/xyz

    where xyz is a substack identifier. For example 020013 is the third
    cube along X, first along Y, and fourteenth along Z after splitting a
    bigger volume into (partially overlapping) substacks. The
    info.json file contains all the details of the splitting. To
    create a valid directory structure use :py:mod:`scripts.make_substacks.py`

    Parameters
    ----------
    indir : str
        Directory where the substacks were saved
    substack_id : str
        Identifier of the substack, e.g. '020013'
    plist : object, optional
        Use passed object instead of the info.json (or info.plist) file in indir
    """
    def __init__(self, indir, substack_id, plist=None):
        if not os.path.isdir(indir+'/'+substack_id):
            raise Exception('Substack', substack_id, 'not found in', indir)
        if plist is None:
            if os.path.exists(indir+'/info.json'):
                self.plist = ujson.loads(open(indir+'/info.json').read())
            elif os.path.exists(indir+'/info.plist.pkl'):
                with open(indir+'/info.plist.pkl', 'rb') as fp:
                    self.plist = pickle.load(fp)
            elif os.path.exists(indir+'/info.plist'):
                self.plist = plistlib.readPlist(indir+'/info.plist')
            else:
                raise Exception('Input directory', indir, 'does not have a valid substack structure')
        else:
            self.plist = plist
        self.indir = indir
        self.substack_id = substack_id
        self.info = self.plist['SubStacks'][substack_id]
        self.parent = {'Height':self.plist['Height'], 'Width':self.plist['Width'], 'Depth':self.plist['Depth']}

    def load_volume_from_h5(self,h5filename):
        """
        Load the volume from an HDF5 file
        """
        hf5 = tables.openFile(h5filename, 'r')
        X0,Y0,Z0 = self.info['X0'], self.info['Y0'], self.info['Z0']
        H,W,D = self.info['Height'], self.info['Width'], self.info['Depth']
        np_tensor_3d = hf5.root.full_image[Z0:Z0+D, Y0:Y0+H, X0:X0+W]
        hf5.close()
        self.imgs = []
        self.pixels = []
        for z in range(D):
            img_z = Image.fromarray(np_tensor_3d[z,:,:])
            self.imgs.append(img_z)
            self.pixels.append(img_z.load())
        tee.log(z+1, 'images read into stack (from h5 file)')

    def load_volume(self, convert_to_gray=True, flip=False, ignore_info_files=False, h5filename=None):
        """Loads a sequence of images into a stack

        Parameters
        ----------
        convert_to_gray : bool
            Should be set to true if reading from RGB tiff files
        flip : bool
            Flip along vertical (Y) axis
        ignore_info_files : bool
            If true, don't trust filenames in the info.json file
        h5filename : str
            If not none, read from this HDF5 file rather than TIFF files
        """
        if h5filename is not None:
            self.load_volume_from_h5(h5filename)
            return
        self.imgs = []
        self.pixels = []
        if ignore_info_files:
            idir = self.indir+'/'+self.substack_id
            files = sorted([idir+'/'+f for f in os.listdir(idir) if f[0] != '.' and valid_suffix(f)])
            # print('******* These are the files in',idir)
            # print(files)
        else:
            files = [self.indir+'/'+fname for fname in self.info['Files']]
        if len(files) == 0:
            raise Exception('No valid files found')
        for z, image_file in enumerate(files):
            img_z = Image.open(image_file)
            if flip:  # when reading a stack saved in vaa3d format Y coordinates are reversed (used by save_substack only)
                img_z = img_z.transpose(Image.FLIP_TOP_BOTTOM)
            if convert_to_gray:
                img_z = img_z.convert('L')
            self.imgs.append(img_z)
            self.pixels.append(img_z.load())
            if z % 25 == 0:
                tee.log(image_file, end='')
            else:
                tee.log('.', end='')
        tee.log(z, 'images read into stack')

    def get_volume(self, convert_to_gray=True, flip=False, ignore_info_files=False, h5filename=None):
	if not hasattr(self, 'imgs'):
	    self.load_volume(convert_to_gray, flip, ignore_info_files, h5filename)
	Depth = self.info['Depth']
	Width = self.info['Width']
	Height = self.info['Height']
	patch = np.zeros((Width,Height,Depth))
	for z in range(Depth):
	    patch[:, :, z] = np.array(self.imgs[z]).T
	return patch

    def neighbors_graph(self, C):
        X = np.array([[c.x, c.y, c.z] for c in C])
        kdtree = cKDTree(X)
        for c in C:
            distances, neighbors = kdtree.query([c.x, c.y, c.z], 6)
            c.distances = sorted(distances)[1:]

    @save_markers_timer.timed
    def save_markers(self, filename, C, floating_point=False):
        """save_markers(filename, C)

        Save markers to a Vaa3D readable file.

        Parameters
        ----------
        filename : str
            Name of the file where markers are saved
        C : list
            List of :class:`Center` objects
        floating_point: bool
            If true, save coordinates in floating point, else round to int
        """
        if len(C) == 0:  # might happen when logging deletions as marker files
            return
        self.neighbors_graph(C)
        ostream = open(filename, 'w')
        print('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b', file=ostream)
        # for i,c in enumerate(C):
        for c in C:
            r, g, b = hi2rgb[int(255*c.hue)][156]
            radius = 0
            shape = 1
            if floating_point:
                cx, cy, cz = c.x, c.y, c.z
            else:
                cx, cy, cz = int(round(c.x)), int(round(c.y)), int(round(c.z))
            comment = str(c)
            print(','.join(map(str, [1+cx, 1+cy, 1+cz, radius, shape, c.name, comment, r, g, b])), file=ostream)
        ostream.close()

    def load_markers(self, filename, from_vaa3d=False, check_coords=True):
        suffix = filename.split('.')[-1]
        if suffix == 'marker':
            C = m_load_markers(filename, from_vaa3d)
        elif suffix == 'apo':
            C = a_load_markers(filename)
        else:
            raise ValueError("Don't understand suffix", suffix)
        if check_coords:
            for c in C:
                if (c.x > self.info['Width'] or c.y > self.info['Height'] or c.z > self.info['Depth']):
                    raise Exception('Coordinates in marker file are out of range', (c.x, c.y, c.z), self.substack_id)
        return C

    def distance_matrix(self, C):
        d = {}
        for i1, c1 in enumerate(C):
            for i2, c2 in enumerate(C):
                if i1 < i2:
                    d[c1.name+' '+c2.name] = math.sqrt((c1.x-c2.x)**2+(c1.y-c2.y)**2+(c1.z-c2.z)**2)
        d = sorted(d.iteritems(), key=operator.itemgetter(1))
        return d

    def histogram(self):
        histogram = self.imgs[0].histogram()
        for z in xrange(1, self.info['Depth']):
            histogram = map(sum, zip(histogram, self.imgs[z].histogram()))
        return histogram
