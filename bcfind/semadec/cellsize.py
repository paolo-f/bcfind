"""

Estimate the size of somata given markers in a substack
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import pylab
import Image
from bcfind.semadec import imtensor

from bcfind.utils import mkdir_p

from bcfind.fast_threshold import two_kapur

# np.set_printoptions(linewidth=160, precision=4, suppress=True, threshold=5000)

def plot_correlations(radii, correlations, centers, radiimax, radiiassigned, args):
    N = 5
    plot_idx = 1000
    for c in centers:
        if c in correlations[radii[0]].keys():
            vals = np.array([correlations[r][c] for r in radii])
            if plot_idx > N*(N+1):
                plot_idx = 1
                pylab.figure(figsize=(22,15))
            pylab.subplot(N+1,N,plot_idx)
            pylab.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3)
            pylab.suptitle(str(args))
            plot_idx += 1
            pylab.plot(radii,vals,label=c.name)
            pylab.plot((radiimax[c],radiimax[c]), (0, correlations[radiimax[c]][c]), 'm-')
            pylab.scatter((radiimax[c],), (correlations[radiimax[c]][c],), color='m', s=7)
            pylab.plot((radiiassigned[c],radiiassigned[c]), (0, correlations[radiiassigned[c]][c]), 'r-')
            pylab.scatter((radiiassigned[c],), (correlations[radiiassigned[c]][c],), facecolors='none', edgecolors='r', s=25)
            pylab.title(c.name)
            pylab.xlabel('r')
            pylab.ylabel('correlation')
            pylab.grid()
            # pylab.legend()
    pylab.show()


def inside_margin(c,substack):
    m = substack.plist['Margin']/2
    if not inside_margin.warned:
        print('\n\n**** Warning!! constant in inside_margin to be fixed\n')
        inside_margin.warned = True
    return min(c.x-m,c.y-m,c.z-m,substack.info['Width']-m-c.x,substack.info['Height']-m-c.y,substack.info['Depth']-m-c.z)
inside_margin.warned = False

def max_correlation(patch, target, c, box):

    def l_correlation(A, B):
        cA = A-np.mean(A)
        cB = B-np.mean(B)
        return np.sum(cA*cB)/np.sqrt((np.sum(cA*cA)*np.sum(cB*cB)))

    correlations = []
    crange = range(-box.shift, box.shift+1)
    for x in crange:
        for y in crange:
            for z in crange:
                cr = l_correlation(
                    patch[c.z-box.size*box.an_z:c.z+box.size*box.an_z,
                          c.y-box.size*box.an_y:c.y+box.size*box.an_y,
                          c.x-box.size*box.an_x:c.x+box.size*box.an_x],
                    target[c.z+z-box.size*box.an_z:c.z+z+box.size*box.an_z,
                           c.y+y-box.size*box.an_y:c.y+y+box.size*box.an_y,
                           c.x+x-box.size*box.an_x:c.x+x+box.size*box.an_x]
                )
                correlations.append(cr)
    return max(correlations)


def save_debug_images(debugdirname, tensor):
    mkdir_p(debugdirname)
    for z in range(tensor.shape[2]):
        out_img = Image.fromarray(np.array(tensor[:,:,z], dtype=np.uint8))
        out_img.save(debugdirname+'/im-%06d.tif' % z)
    print('saved debug image in', debugdirname)


def tensor_from_substack(substack):
    D = substack.info['Depth']
    W = substack.info['Width']
    H = substack.info['Height']
    tensor = np.zeros((D, H, W))
    for z in range(D):
        tensor[z,:,:] = np.array(substack.imgs[z])
    return tensor


def estimate_sizes(radii, centers, tensor, substack, box, debug_images):
    for c in centers:
        c.xr, c.yr, c.zr = np.round([c.x, c.y, c.z])

    correlations = {}
    for radius in radii:
        correlations[radius] = {}
    import progressbar as pb
    incenters = [c for c in centers if inside_margin(c,substack) > 0]
    pbar = pb.ProgressBar(widgets=['Processing %d markers: ' % len(incenters),
                                   pb.Percentage(), ' ', pb.AdaptiveETA()])
    id_c=0
    for c in pbar(incenters):
        print(c.xr,c.yr,c.zr)
        local_slice = [slice(c.zr-box.an_z*(box.size+1),c.zr+box.an_z*(box.size+1)),
                       slice(c.yr-box.an_y*(box.size+1),c.yr+box.an_y*(box.size+1)),
                       slice(c.xr-box.an_x*(box.size+1),c.xr+box.an_x*(box.size+1))]
        local = tensor[local_slice]
        histogram = np.histogram(local,bins=256,range=(0,256))[0]
        thresholds = two_kapur(histogram)
        local[local < thresholds[0]] = 0
        localized_tensor = np.zeros_like(tensor)
        localized_tensor[local_slice] = local

        if debug_images:
            save_debug_images('debugme/'+substack.substack_id+'-'+c.name.replace(' ','-'), localized_tensor)

        for radius in radii:
            sigmas = [radius*box.an_z, radius*box.an_y, radius*box.an_x]
            target_tensor_3d = np.zeros_like(tensor)
            target_tensor_3d[c.zr, c.yr, c.xr] = 1
            target_tensor_3d = filters.gaussian_filter(target_tensor_3d, sigma=sigmas,
                                                       mode='constant', cval=0.0,
                                                       truncate=max(2,radius*0.5))
            # undo scipy normalization of the gaussian filter
            target_tensor_3d = (target_tensor_3d / np.max(target_tensor_3d))
            target_tensor_3d *= 255
            correlations[radius][c] = max_correlation(localized_tensor, target_tensor_3d, c, box)

        id_c+=1

    return correlations


def make_dataframe(radii, correlations, centers, gain, mincorr, defaultr, an_x, an_y, an_z):
    """Create a data frame for the results, formatted according to the Vaa3D .apo file format.
    In particular:
       - the 'res_1' column is filled with the estimated radius of a soma
       - the 'comment' column contains the same information
       - the 'rgn size (#voxels)' column is the volume corresponding to the radius
       - the 'mass' column contains the maximum correlation when varying the radius

    Parameters
    ----------
    radii : array
        radii to try
    correlations : dict of dicts
        for each radius, the correlations at that radius for all centers
    centers : list
        set of centers
    args.gain : float
        When maximizing the correlation, we accept a new maximum only if
        the relative increase is above gain
    """
    columns = ['#no', 'order_info',  # 0,1
               'name', 'comment',  # 2,3
               'z', 'x', 'y',  # 4,5,6
               'max_i', 'mean_i', 'sdev_i',  # 7,8,9
               'rgn size (#voxels)', 'mass',  # 10,11
               'res_1', 'res_2', 'res_3',  # 12,13,14
               'red', 'green', 'blue']  # 15,16,17

    clist = {}
    for column in columns:
        clist[column] = []
    jet = pylab.get_cmap('jet')
    radiimax, radiiassigned = {}, {}
    for i, c in enumerate(centers):
        if c in correlations[radii[0]].keys():
            corrmax = 0
            rmax = radii[0]
            for r in radii:
                # if correlations[r][c] > corrmax:
                if (correlations[r][c] - corrmax)/corrmax > gain:
                    rmax = r
                    corrmax = correlations[r][c]
            radiimax[c] = rmax
            if corrmax < mincorr:
                rmax = defaultr
            radiiassigned[c] = rmax
            # Calculate the volume for Vaa3D visualization as surface.
            # The apo file seems to accept only spherical objects. Because of anisotropy
            # we actually have a prolate spheroid so we compute the volume of such
            # ellipsoid below (but of course Vaa3D will show a sphere)
            volume = 4.0/3.0*np.pi*(an_x*rmax)*(an_y*rmax)*(an_z*rmax)
            colors = jet(int(corrmax*255))

            clist['#no'].append(i)
            clist['order_info'].append(i)
            clist['name'].append(c.name)
            clist['comment'].append('r=%.2f (%.2f)' % (radiimax[c], radiiassigned[c]))
            clist['x'].append(c.x)#add 1 pax
            clist['y'].append(c.y)
            clist['z'].append(c.z)
            clist['max_i'].append(0.0)
            clist['mean_i'].append(0.0)
            clist['sdev_i'].append(0.0)
            clist['rgn size (#voxels)'].append(volume)
            clist['mass'].append(corrmax)
            clist['res_1'].append(radiiassigned[c])
            clist['res_2'].append(radiimax[c])
            clist['res_3'].append('')
            clist['red'].append(int(255*colors[0]))
            clist['green'].append(int(255*colors[1]))
            clist['blue'].append(int(255*colors[2]))

    df = pd.DataFrame(clist, columns=columns)
    return df, radiimax, radiiassigned


def save_apo_file(df, outputfile):
    """This file can be dragged/dropped into Vaa3D 3D view and shows
    spheres of different size around the soma centers. Use Cmd/Ctrl P to
    make spheres transparent in Vaa3D
    """
    df.to_csv(outputfile, index=False)
    print(df[['#no', 'name', 'comment', 'x', 'y', 'z', 'mass']])
    print('Saved', outputfile)
