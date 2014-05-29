"""
Mean shift cell detector.
"""
from __future__ import print_function
import numpy as np
import progressbar as pb
import mahotas as mh
import sklearn.decomposition
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import extmath
from scipy.spatial import cKDTree
from collections import namedtuple

from . import volume
from . import threshold
from . import timer
from .log import tee
from .utils import mkdir_p

pca_analysis_timer = timer.Timer('PCA analysis')
mean_shift_timer = timer.Timer('Mean shift')
ms_timer = timer.Timer('Overall method')
patch_ms_timer = timer.Timer('Overall method (patch)')

def is_local_max(x, y, z, px, ww, hh, dd, min_mass):
    """
    Decide if px[x,y] is a local maximum. w and h are image width and
    height.  Criteria: (1) the intensity of x,y should be higher than
    all other voxels in a cube of edge length 3 centered around
    x,y. (2) the sum of intensities in the cube bigger than a
    threshold.
    """
    if x-1 < 0 or x+1 >= ww or y-1 < 0 or y+1 >= hh or z-1 < 0 or z+1 >= dd:
        return False
    mass = (px[z][x, y]+px[z][x-1, y-1]+px[z][x, y-1] +
            px[z][x+1, y-1]+px[z][x-1, y]+px[z][x+1, y] +
            px[z][x-1, y+1]+px[z][x, y+1]+px[z][x+1, y+1])
    mass += (px[z+1][x, y]+px[z+1][x-1, y-1]+px[z+1][x, y-1] +
             px[z+1][x+1, y-1]+px[z+1][x-1, y]+px[z+1][x+1, y] +
             px[z+1][x-1, y+1]+px[z+1][x, y+1]+px[z+1][x+1, y+1])
    mass += (px[z-1][x, y]+px[z-1][x-1, y-1]+px[z-1][x, y-1] +
             px[z-1][x+1, y-1]+px[z-1][x-1, y]+px[z-1][x+1, y] +
             px[z-1][x-1, y+1]+px[z-1][x, y+1]+px[z-1][x+1, y+1])
    r = mass > min_mass
    r = r and (px[z][x, y] >= px[z][x-1, y-1] and px[z][x, y] >= px[z][x, y-1] and
               px[z][x, y] >= px[z][x+1, y-1] and px[z][x, y] >= px[z][x-1, y] and
               px[z][x, y] >= px[z][x+1, y] and px[z][x, y] >= px[z][x-1, y+1] and
               px[z][x, y] >= px[z][x, y+1] and px[z][x, y] >= px[z][x+1, y+1])
    r = r and (px[z][x, y] >= px[z+1][x-1, y-1] and px[z][x, y] >= px[z+1][x, y-1] and
               px[z][x, y] >= px[z+1][x+1, y-1] and px[z][x, y] >= px[z+1][x-1, y] and
               px[z][x, y] >= px[z+1][x+1, y] and px[z][x, y] >= px[z+1][x-1, y+1] and
               px[z][x, y] >= px[z+1][x, y+1] and px[z][x, y] >= px[z+1][x+1, y+1])
    r = r and (px[z][x, y] >= px[z-1][x-1, y-1] and px[z][x, y] >= px[z-1][x, y-1] and
               px[z][x, y] >= px[z-1][x+1, y-1] and px[z][x, y] >= px[z-1][x-1, y] and
               px[z][x, y] >= px[z-1][x+1, y] and px[z][x, y] >= px[z-1][x-1, y+1] and
               px[z][x, y] >= px[z-1][x, y+1] and px[z][x, y] >= px[z-1][x+1, y+1])
    return r


@mean_shift_timer.timed
def mean_shift(X, intensities=None, bandwidth=None, seeds=None,
               cluster_all=True, max_iterations=300, verbose=False, use_scipy=True):
    """mean_shift(X, intensities=None, bandwidth=None, seeds=None,
                  cluster_all=True, max_iterations=300, verbose=False, use_scipy=True)

    Mean shift algorithm

    Implementation taken from scikit-learn with two minor variants:

        - Use (by default) scipy KD-trees, which are faster in our case
        - weigthed version of mean-shift using `intensities` as 
          weights (i.e., we compute centers of mass rather than means)

    Parameters
    ----------

    X : array-like, shape=[n_samples, n_features]
        Input data.

    intensities : array-like, shape=[n_samples]
        Voxel intensities, used to weight the mean

    bandwidth : float
        Kernel bandwidth.

    seeds : array-like, shape=[n_seeds, n_features]
        Point used as initial kernel locations.

    use_scipy : bool
        If true use cKDTree from scipy.spatial, otherwise 
        use NearestNeighbors from sklearn.neighbors

    Returns
    -------

    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.

    labels : array, shape=[n_samples]
        Cluster labels for each point.

    volumes : array, shape=[n_clusters]
        Volume of each cluster (# of points in the cluster)

    masses : array, shape=[n_clusters]
        Mass of each cluster (sum of intensities of points in the cluster).

    trajectories : list
        MS trajectories for debugging purposes.
    """
    if seeds is None:
        seeds = X
    n_points, n_features = X.shape
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    center_volume_dict = {}
    center_mass_dict = {}
    # tee.log('Fitting NearestNeighbors on', n_points, 'points')
    if use_scipy:
        kdtree=cKDTree(X)
    else:
        nbrs = NearestNeighbors(radius=bandwidth).fit(X)

    # For each seed, climb gradient until convergence or max_iterations
    trajectories = {}  # for each seed, a list of points
    tee.log('Moving kernels for', len(seeds), 'seeds')
    pbar = pb.ProgressBar(widgets=['Moving %d seeds: ' % len(seeds), pb.Percentage()],
                          maxval=len(seeds)).start()
    for seed_no, my_mean in enumerate(seeds):
        completed_iterations = 0
        seed = my_mean
        trajectories[seed_no] = []
        while True:
            # Find mean of points within bandwidth
            if use_scipy:
                i_nbrs = kdtree.query_ball_point(my_mean, r=bandwidth)
            else:
                i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth,
                                               return_distance=False)[0]
            points_within = X[i_nbrs]
            if len(points_within) == 0:
                break  # Depending on seeding strategy this condition may occur
            my_old_mean = my_mean  # save the old mean
            if intensities is None:
                my_mean = np.mean(points_within, axis=0)
            else:
                my_mean = np.average(points_within, axis=0, weights=intensities[i_nbrs])
            # If converged or at max_iterations, addS the cluster
            if extmath.norm(my_mean - my_old_mean) < stop_thresh or completed_iterations == max_iterations:
                center_volume_dict[tuple(my_mean)] = len(points_within)
                center_mass_dict[tuple(my_mean)] = sum(intensities[i_nbrs])
                break
            completed_iterations += 1
            trajectories[seed_no].append(my_mean)
        if verbose:
            print('seed', seed, '-->', my_mean,
                  center_volume_dict[tuple(my_mean)], center_mass_dict[tuple(my_mean)], completed_iterations)

        pbar.update(seed_no+1)
    pbar.finish()
    # POST PROCESSING: remove near duplicate points
    # If the distance between two kernels is less than the bandwidth,
    # then we have to remove one because it is a duplicate. Remove the
    # one with fewer points.
    sorted_by_intensity = sorted(center_mass_dict.items(),
                                 key=lambda tup: tup[1], reverse=True)
    sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
    unique = np.ones(len(sorted_centers), dtype=np.bool)
    print('started from', len(seeds), 'seeds, now |unique|=', len(unique))
    # print('|center_mass_dict|=', len(center_mass_dict))
    if len(center_mass_dict) == 0:
        tee.log('No valid seeds. Giving up')
        return None, None, None, None, None

    nbrs = NearestNeighbors(radius=bandwidth).fit(sorted_centers)
    for i, center in enumerate(sorted_centers):
        if unique[i]:
            neighbor_idxs = nbrs.radius_neighbors([center],
                                                  return_distance=False)[0]
            unique[neighbor_idxs] = 0
            unique[i] = 1  # leave the current point as unique
    cluster_centers = sorted_centers[unique]
    print('|cluster_centers|=', len(cluster_centers))
    volumes = [0]*len(cluster_centers)
    masses = [0]*len(cluster_centers)
    for i, c in enumerate(cluster_centers):
        volumes[i] = center_volume_dict[tuple(c)]
        masses[i] = center_mass_dict[tuple(c)]
    # ASSIGN LABELS: a point belongs to the cluster that it is closest to
    nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
    labels = np.zeros(n_points, dtype=np.int)
    distances, idxs = nbrs.kneighbors(X)
    if cluster_all:
        labels = idxs.flatten()
    else:
        labels[:] = -1
        bool_selector = distances.flatten() <= bandwidth
        labels[bool_selector] = idxs.flatten()[bool_selector]
    return cluster_centers, labels, volumes, masses, trajectories


@ms_timer.timed
def ms(substack, args):
    """ms(substack, args)

    Find cells using mean shift.

    In this version, the substack is processed as a whole.

    Parameters
    ----------
    substack : object
        :class:`bcfind.volume.SubStack` object representing the substack to be analyzed.
    args : object
        :py:class:`argparse.Namespace` object containing the
        arguments passed to the find_cells script, in particular
        - args.outdir: directory where results are saved
        - args.hi_local_max_radius: radius of the sphere used to decide whether a local maximum should be a seed
        - args.mean_shift_bandwidth: bandwidth for the mean shift algorithm
    """
    L = []
    intensities = []
    Depth = substack.info['Depth']
    Width = substack.info['Width']
    Height = substack.info['Height']

    histogram = substack.histogram()
    thresholds = threshold.multi_kapur(histogram, 2)
    tee.log('Maximum entropy discretization (Kapur et al. 1985, 3 bins):', thresholds)
    minint = thresholds[0]
    min_mass = minint*27
    if minint <= 1:
        tee.log('minint threshold too low (%d) - I believe there are no cells in this substack' % minint)
        substack.save_markers(args.outdir+'/'+substack.substack_id+'/ms.marker', [])
        return
    if thresholds[1] <= 15:
        tee.log('thresholds[1] threshold too low (%d) - I believe there are no cells in this substack' % thresholds[1])
        substack.save_markers(args.outdir+'/'+substack.substack_id+'/ms.marker', [])
        return

    for z in xrange(Depth):
        ss = substack.pixels[z]
        for x in xrange(Width):
            for y in xrange(Height):
                if ss[x, y] > minint:
                    L.append([x, y, z])
                    intensities.append(ss[x, y])

    tee.log('Found', len(L), 'voxels above the threshold', minint)
    if len(L) < 10:
        tee.log('Too few points (%d) - I believe there are no cells in this substack' % len(L))
        substack.save_markers(args.outdir+'/'+substack.substack_id+'/ms.marker', [])
        return

    L = np.array(L)
    intensities = np.array(intensities)
    bandwidth = args.mean_shift_bandwidth

    C = [volume.Center(L[i][0], L[i][1], L[i][2]) for i in xrange(len(L)) if
         is_local_max(int(round(L[i][0])), int(round(L[i][1])), int(round(L[i][2])),
                      substack.pixels, Width, Height, Depth, min_mass)]
    if len(C) > 80000:
        tee.log('Too many candidates,', len(C), 'I believe this substack is messy and give up')
        return
    seeds = np.array([[c.x, c.y, c.z] for c in C])
    for c in C:
        c.name = 'seed'
    substack.save_markers(args.outdir+'/'+substack.substack_id+'/seeds.marker', C)

    Lx = [int(round(l[0])) for l in L]
    Ly = [int(round(l[1])) for l in L]
    Lz = [int(round(l[2])) for l in L]

    cluster_centers, labels, volumes, masses, trajectories = mean_shift(L, intensities=intensities,
                                                                        bandwidth=bandwidth, seeds=seeds)
    if cluster_centers is None:
        return
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    tee.log("number of estimated clusters : %d" % n_clusters_)
    C = []
    for i, cc in enumerate(cluster_centers):
        tee.log(i, cc)
        c = volume.Center(cc[0], cc[1], cc[2])
        c.name = 'MS_center %d' % i
        c.volume = float(volumes[i])
        c.mass = float(masses[i])
        C.append(c)

    Lcluster = [C[labels[i]] for i in xrange(len(L))]

    _pca_analysis(Lx, Ly, Lz, Lcluster, C)
    # _wpca_analysis(L, C, intensities)
    _finalize_masses(L, C, intensities)
    # _finalize_radii(Lx, Ly, Lz, Lcluster, C)
    filename = args.outdir+'/'+substack.substack_id+'/ms.marker'
    substack.save_markers(filename, C)
    tee.log('Markers saved to', filename)
    if args.save_image:
        image_saver = volume.ImageSaver(args.outdir, substack, C)
        image_saver.save_above_threshold(Lx, Ly, Lz, thresholds)
        image_saver.save_vaa3d(C, Lx, Ly, Lz, Lcluster,
                               draw_centers=True, colorize_voxels=True,
                               trajectories=trajectories)
        tee.log('Debugging images saved in', args.outdir)
    else:
        tee.log('Debugging images not saved')


@patch_ms_timer.timed
def _patch_ms(patch, args):
    histogram = np.histogram(patch,bins=256,range=(0,256))[0]
    # np.set_printoptions(precision=2,threshold=256)
    # print(histogram)
    thresholds = threshold.multi_kapur(histogram, 2)
    tee.log('Maximum entropy discretization (Kapur et al. 1985, 3 bins):', thresholds)
    minint = thresholds[0]
    if minint != thresholds[0]:
        tee.log('Warning: minint was lowered')
    if minint <= 1:
        tee.log('minint threshold too low (%d) - I believe there are no cells in this substack' % minint)
        return None
    if thresholds[1] <= args.min_second_threshold:
        tee.log('thresholds[1] threshold too low (%d) - I believe there are no cells in this substack' % thresholds[1])
        return None

    (Lx,Ly,Lz) = np.where(patch > minint)
    
    intensities = patch[(Lx,Ly,Lz)]
    L=np.array(zip(Lx,Ly,Lz), dtype=np.uint16)
    tee.log('Found', len(L), 'voxels above the threshold', minint)
    if len(L) < 10:
        tee.log('Too few points (%d) - I believe there are no cells in this substack' % len(L))
        return None

    bandwidth = args.mean_shift_bandwidth

    radius=args.hi_local_max_radius
    reg_maxima = mh.regmax(patch.astype(np.int))
    xx, yy, zz = np.mgrid[:2*radius+1, :2*radius+1, :2*radius+1]
    sphere = (xx - radius) ** 2 + (yy - radius) ** 2 + (zz - radius) ** 2
    se = sphere<=radius*radius
    f = se.astype(np.float64)
    min_mass = np.sum(f)*minint
    local_mass = mh.convolve(patch, weights=f, mode='constant', cval=0.0)
    above = local_mass > min_mass
    himaxima = np.logical_and(reg_maxima,above)
    if np.sum(himaxima) > 10000:
        tee.log('Too many candidates,', np.sum(himaxima), 'I believe this substack is messy and give up')
        return None
    C = [volume.Center(x,y,z) for (x,y,z) in zip(*np.where(himaxima))]
    
    if len(C) == 0:
        tee.log('No maxima above. #himaxima=', np.sum(himaxima), '#above=', np.sum(above), '. Giving up')
        return None
    seeds = np.array([[c.x, c.y, c.z] for c in C])
    # Save seeds with some info for debugging purposes
    for c in C:
        c.name = 'seed'
        c.mass = patch[c.x,c.y,c.z]
        c.volume = thresholds[0]
    cluster_centers, labels, volumes, masses, trajectories = mean_shift(L, intensities=intensities,
                                                                        bandwidth=bandwidth, seeds=seeds)
    if cluster_centers is None:
        return None
    masses = np.zeros(len(cluster_centers))
    for i,c in enumerate(cluster_centers):
        masses[i] = local_mass[int(c[0]+0.5),int(c[1]+0.5),int(c[2]+0.5)]
    PatchMSRet = namedtuple('PatchMSRet',
                            ['cluster_centers','labels','masses','L','seeds'])
    r = PatchMSRet(cluster_centers=cluster_centers,
                   labels=labels, masses=masses, L=L, seeds=C)
    return r

    
def pms(substack, args):
    """Find cells using mean shift.

    In this version, the substack is split into eight patches.

    Parameters
    ----------
    substack : object
        :class:`bcfind.volume.SubStack` object representing the substack to be analyzed.
    args : object
        :py:class:`argparse.Namespace` object containing the
        arguments passed to the find_cells script, in particular
        - args.outdir: directory where results are saved
        - args.hi_local_max_radius: radius of the sphere used to decide whether a local maximum should be a seed
        - args.mean_shift_bandwidth: bandwidth for the mean shift algorithm
    """
    D = substack.info['Depth']
    W = substack.info['Width']
    H = substack.info['Height']
    M=20
    patch = np.zeros((W,H,D))
    for z in range(D):
        patch[:,:,z] = np.array(substack.imgs[z]).T
    slicesx = [slice(0, W/2+M), slice(W/2-M,W)]
    slicesy = [slice(0, H/2+M), slice(H/2-M,H)]
    slicesz = [slice(0, D/2+M), slice(D/2-M,D)]
    cluster_centers = np.zeros((0,3))
    cluster_masses = np.zeros(0)
    L = np.zeros((0,3))
    labels = np.zeros(0)
    seeds = []
    counter=0
    for sx in slicesx:
        for sy in slicesy:
            for sz in slicesz:
                counter += 1
                tee.log('%d/8:'%counter, 'Analyzing minisubstack',sx,sy,sz)
                rval = _patch_ms(patch[sx,sy,sz], args)
                origin = [sx.start,sy.start,sz.start]
                if rval is not None:
                    cluster_centers = np.concatenate((cluster_centers, rval.cluster_centers + origin))
                    cluster_masses = np.concatenate((cluster_masses, rval.masses))
                    labels = np.concatenate((labels, rval.labels+len(rval.cluster_centers)))
                    L = np.concatenate((L,rval.L+origin))
                    for c in rval.seeds:
                        c.x += origin[0]
                        c.y += origin[1]
                        c.z += origin[2]
                    seeds.extend(rval.seeds)
    if len(cluster_centers) > 0:
        # remove near duplicate points (because of overlapping margins)
        indices = np.argsort(cluster_masses)
        sorted_centers = cluster_centers[indices]
        sorted_masses = cluster_masses[indices]
        # sorted_volumes = volumes[indices]
        unique = np.ones(len(sorted_centers), dtype=np.bool)
        nbrs = NearestNeighbors(radius=5.5).fit(sorted_centers)
        for i, center in enumerate(sorted_centers):
            if unique[i]:
                neighbor_idxs = nbrs.radius_neighbors([center],
                                                      return_distance=False)[0]
                unique[neighbor_idxs] = 0
                unique[i] = 1  # leave the current point as unique
        cluster_centers = sorted_centers[unique]
        masses = sorted_masses[unique]
        masses_mean = np.mean(masses)
        masses_std = np.std(masses)
        # volumes = sorted_volumes[unique]
    C = []
    for i, cc in enumerate(cluster_centers):
        c = volume.Center(cc[0], cc[1], cc[2])
        c.name = 'MS_center %d' % i
        c.volume = (masses[i]-masses_mean)/masses_std #volumes[i]
        c.mass = masses[i]
        tee.log(i, cc, c)
        C.append(c)
    
    filename = args.outdir+'/'+substack.substack_id+'/ms.marker'
    substack.save_markers(filename, C)
    tee.log('Markers saved to', filename)
    filename = args.outdir+'/'+substack.substack_id+'/seeds.marker'
    substack.save_markers(args.outdir+'/'+substack.substack_id+'/seeds.marker', seeds)
    tee.log(len(seeds), 'seeds saved to', filename)

    if args.save_image:
        image_saver = volume.ImageSaver(args.outdir, substack, C)
        Lx = [int(x) for x in L[:,0]]
        Ly = [int(y) for y in L[:,1]]
        Lz = [int(z) for z in L[:,2]]
        image_saver.save_above_threshold(Lx,Ly,Lz)
        Lcluster = [C[int(labels[i])] for i in xrange(len(L))]
        # Note: no trajectories in this case
        image_saver.save_vaa3d(C, Lx, Ly, Lz, Lcluster,
                               draw_centers=True,
                               colorize_voxels=True)
        tee.log('Debugging images saved in', args.outdir)
    else:
        tee.log('Debugging images not saved')

def _finalize_masses(X, C, intensities):
    """
    Regardless of the parameters of the algorithm, place a ball of
    radius 10 around each center and compute the mass in the ball.
    Rationale: thresholding for discriminating between centers and non
    centers should not depend on parameters used to seek the centers.
    This mass will be later used for the recall-precision curve.
    Hopefully wild variations of performance across different
    substacks will be reduced this way.
    """
    n_points, n_features = X.shape
    tee.log('Finalizing masses - Fitting NearestNeighbors on', n_points, 'points')
    nbrs = NearestNeighbors(radius=10.0).fit(X)

    for c in C:
        array_c = np.array([c.x, c.y, c.z])
        i_nbrs = nbrs.radius_neighbors([array_c], 10.0, return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break
        c.mass = sum(intensities[i_nbrs])


def _finalize_radii(Lx, Ly, Lz, Lcluster, C):
    tee.log('Finalizing radii')
    from sklearn import mixture
    for c in C:
        X = np.array([[Lx[i], Ly[i], Lz[i]] for i in xrange(len(Lx)) if Lcluster[i] is c])
        clf = mixture.GMM(n_components=1, covariance_type='spherical')
        clf.fit(X)
        c.radius = np.sqrt(clf.covars_[0][0])


@pca_analysis_timer.timed
def _pca_analysis(Lx, Ly, Lz, Lcluster, C):
    """
    Determine the eccentricity of each cluster using PCA. The smallest
    normalized explained variance is small for flat of filiform
    objects.
    """
    groups = {c: [] for c in C}
    # nogroup_hist=[0]*256
    for i in xrange(len(Lx)):
        if Lcluster[i] is not None:
            groups[Lcluster[i]].append([Lx[i], Ly[i], Lz[i]])
    for i, c in enumerate(C):
        if c.volume < 10:
            c.EVR = [0.333, 0.333, 0.333]
            c.last_variance = 0.333
            continue
        X = np.array(groups[c])
        pca = sklearn.decomposition.PCA(n_components=3)
        X_r = pca.fit(X).transform(X)
        c.EVR = pca.explained_variance_ratio_
        c.last_variance = c.EVR[2]


def _wpca_analysis(L, C, intensities):
    """
    Determine the eccentricity of each cluster using weighted PCA (See
    Jolliffe 2002, 14.2.1). The smallest normalized explained variance
    is small for flat of filiform objects.

    - L is a numpy matrix (one point on each row)
    - intensities are gray levels of each point

    No cluster assignment is used here: a ball of radius 10 around each
    center is used to find the cloud of points.
    """
    np.set_printoptions(threshold=50000)
    n_points, n_features = L.shape
    tee.log('WPCA - Fitting NearestNeighbors on', n_points, 'points')
    nbrs = NearestNeighbors(radius=10.0).fit(L)
    for i, c in enumerate(C):
        array_c = np.array([c.x, c.y, c.z])
        i_nbrs = nbrs.radius_neighbors([array_c], 10.0, return_distance=False)[0]
        points_within = L[i_nbrs]
        if len(points_within) < 64:  # too small set, there is no point in running PCA
            c.EVR = [0.499, 0.499, 0.002]
            c.last_variance = c.EVR[2]
        else:
            w = np.sqrt(intensities[i_nbrs]/255.0)
            wX = np.dot(np.diag(w), points_within)
            pca = sklearn.decomposition.PCA(n_components=3)
            X_r = pca.fit(wX).transform(wX)
            c.EVR = pca.explained_variance_ratio_
            c.last_variance = c.EVR[2]
        print('WPCA done on', i, '/', len(C), 'name=', c.name, 'EVR=', c.EVR)
        # print('w=',w)
        # print('within=',points_within)
        # print('wX=',wX)
