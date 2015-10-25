import numpy as np
import timeit
import tables
import uuid
from progressbar import *
from PIL import Image
import cv2
from pyparsing import Word, nums, restOfLine, Suppress, Group, Combine, Optional,alphanums
import glob

from bcfind.volume import SubStack
from bcfind.utils import mkdir_p
from bcfind.semadec import imtensor

def parse_transformation_file(transformation_file):
    """
    Method that parses the file reporting the rigid transformation, estimated after the fine registration procedure .

    Parameters
    ----------
    
    transformation_file: str
	file reporting the estimated rigid transfrormation

    Returns
    -------

    R : numpy array of shape (3, 3)
	rotational component of the estimated rigid transformation
    t : numpy array of shape (3)
	translational component of the estimated rigid transformation
    """
    scientific_notation = "e-" + Word(nums)
    float_number = Combine(Optional("-") + Word(nums) + Optional("." + Word(nums) + Optional(scientific_notation) ))
    row_matrix =  Group(float_number + float_number + float_number)
    comma = Suppress(",")
    grammar = ('#' + restOfLine).suppress() | Word(alphanums + "_" + alphanums) + comma + row_matrix + comma + row_matrix + comma + row_matrix + comma + row_matrix + restOfLine.suppress()

    try:
        f = open(transformation_file, 'r')
    except IOError:
        print('file not existing')
        sys.exit(1)

    line = f.readline()
    parsed_line = grammar.parseString(line)
    R = np.zeros((3,3))
    R[0, :]=np.array(parsed_line[1][:]).astype(np.float)
    R[1, :]=np.array(parsed_line[2][:]).astype(np.float)
    R[2, :]=np.array(parsed_line[3][:]).astype(np.float)
    t=np.array(parsed_line[4][:]).astype(np.float)

    return R, t

def transform_substack(indir, tensorimage, substack_id, R, t, extramargin, outdir=None, invert=False, save_tiff=False, save_hdf5=False):
    """
    Method that applies a previously estimated rigid transformation to a specific substack.

    Parameters
    ----------
    
    indir : str
	substack dir of the input view
    tensorimage : str
	the whole tensor in hdf5 format
    substack_id : str
	id of the substack that will be transformed
    R : numpy array of shape (3, 3)
	rotational component of the estimated rigid transformation
    t : numpy array of shape (3)
	translational component of the estimated rigid transformation
    extramargin : int
	extramargin used to extract the transformed substack from tensorimage
    outdir : str 
	output directory where the transformed substack will be saved (Default: None)
    invert : boolean 
	if True the tranformation is inverted (Default: False)
    save_tiff : boolean 
	save the transformed substack in a stack of tiff slices (Default: False)
    save_hdf5 : boolean 
	save the transformed substack as a hdf5 tensor (Default: False)

    Returns
    -------

    pixels_transformed_input: numpy tensor
	tensor of the transformed substack

    """
    ss = SubStack(indir, substack_id)
    input_stack_file = tensorimage
    hf5 = tables.openFile(input_stack_file, 'r')
    full_D, full_H, full_W = hf5.root.full_image.shape
    X0,Y0,Z0 = ss.info['X0'], ss.info['Y0'], ss.info['Z0']
    origin = (Z0, Y0, X0)
    H,W,D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    or_ss_shape = (D,H,W)
    offset_W=int((3**(1/2.0)*W + W/2.0 - W)/2.0)
    offset_H=int((3**(1/2.0)*H + H/2.0 - H)/2.0)
    offset_D=int((3**(1/2.0)*D + D/2.0 - D)/2.0)


    if offset_W < extramargin:
        offset_W = extramargin
    if offset_H < extramargin:
        offset_H = extramargin
    if offset_D < extramargin:
        offset_D = extramargin

    offset_D_left = offset_D if int(origin[0] - offset_D) > 0 else origin[0]
    offset_H_left = offset_H if int(origin[1] - offset_H) > 0 else origin[1]
    offset_W_left = offset_W if int(origin[2] - offset_W) > 0 else origin[2]
    offset_D_right = offset_D if int(origin[0] + or_ss_shape[0] + offset_D) <= full_D else full_D - (origin[0] + or_ss_shape[0])
    offset_H_right = offset_H if int(origin[1] + or_ss_shape[1] + offset_H) <= full_H else full_H - (origin[1] + or_ss_shape[1])
    offset_W_right = offset_W if int(origin[2] + or_ss_shape[2] + offset_W) <= full_W else full_W - (origin[2] + or_ss_shape[2])



    pixels_input = hf5.root.full_image[origin[0] - offset_D_left:origin[0] + or_ss_shape[0] + offset_D_right,
                                       origin[1] - offset_H_left:origin[1] + or_ss_shape[1] + offset_H_right,
                                       origin[2] - offset_W_left:origin[2] + or_ss_shape[2] + offset_W_right]


    exmar_D_left = 0 if offset_D_left == origin[0] else extramargin
    exmar_H_left  = 0 if offset_H_left == origin[1] else extramargin
    exmar_W_left  = 0 if offset_W_left == origin[2] else extramargin

    depth_target, height_target, width_target = or_ss_shape[0] + 2 * extramargin, or_ss_shape[1] + 2 * extramargin, or_ss_shape[2] + 2 * extramargin #new
    depth_input, height_input, width_input = pixels_input.shape[0], pixels_input.shape[1],  pixels_input.shape[2]
    pixels_transformed_input = np.zeros((depth_target,height_target,width_target), dtype=np.uint8)


    total_start = timeit.default_timer()

    coords_2d_target = np.vstack(np.indices((width_target,height_target)).swapaxes(0,2).swapaxes(0,1))
    invR = R.T

    if invert:
        t = -np.dot(invR, t)
        invR = R

    invR_2d_transpose = np.transpose(np.dot(invR[:, 0:2], np.transpose(coords_2d_target - t[0:2])))
    offset_coords = np.array([[offset_W_left - exmar_W_left, offset_H_left - exmar_H_left, offset_D_left - exmar_D_left]]*invR_2d_transpose.shape[0])#new

    for z in xrange(0, depth_target, 1):
        R_t_3d = np.transpose(invR_2d_transpose + invR[:, 2] * (z - t[2]) + offset_coords)
        good_indices = np.array(range(R_t_3d.shape[1]))
        good_indices = good_indices[(R_t_3d[0, :] > 0) * (R_t_3d[1, :] > 0) * (R_t_3d[2, :] > 0) * (R_t_3d[0, :] < (width_input - 1)) * (R_t_3d[1, :] < (height_input - 1)) * (R_t_3d[2, :] < (depth_input - 1))]
        R_t_3d = R_t_3d.take(good_indices,axis=1)
        R_t_3d = np.round(R_t_3d).astype(int)
        coords_2d_target_tmp = coords_2d_target.take(good_indices, axis=0)
        coords_3d_target_tmp = np.hstack((coords_2d_target_tmp, np.ones((coords_2d_target_tmp.shape[0], 1)).astype(int)*z))
        pixels_transformed_input[coords_3d_target_tmp[:, 2], coords_3d_target_tmp[:, 1], coords_3d_target_tmp[:, 0]] = pixels_input[R_t_3d[2, :], R_t_3d[1, :], R_t_3d[0, :]]

    total_stop = timeit.default_timer()
    print ("total time transformation stack:%s "%(str(total_stop - total_start)))

    pixels_transformed_input = np.array(pixels_transformed_input, dtype=np.uint8)
    if save_tiff or save_hdf5:
	mkdir_p(outdir)
	if save_tiff:
	    minz = int(ss.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
	    _prefix = '_'.join(ss.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'
	    substack_outdir = outdir + '/' + substack_id
	    imtensor.save_tensor_as_tif(pixels_transformed_input, substack_outdir, minz, prefix=_prefix)
	if save_hdf5: 
	    target_shape = (depth_target, height_target, width_target)
	    atom = tables.UInt8Atom()
	    h5f = tables.openFile(outdir + '/' + ss.substack_id + '.h5', 'w')
	    ca = h5f.createCArray(h5f.root, 'full_image', atom, target_shape)
	    for z in xrange(0, depth_target, 1):
		ca[z, :, :] = pixels_transformed_input[z,:,:]
	    h5f.close()

    return pixels_transformed_input


def transform_tensor(input_view, target_view, transformed_view, R, t, suffix='.tif', convert_to_gray=True):
    """
    Method that applies a previously estimated rigid transformation to an input tensor
    
    Parameters
    ----------

    input_view : str
	directory of hdf5 file of the input view
    target_view : str
	directory of hdf5 file of the target view
    transformed_view : str
	directory of hdf5 file of the transformed input view
    R : numpy array of shape (dim_points, 3)
	rotational component of the estimated rigid transformation
    t : numpy array of shape (3)
	translational component of the estimated rigid transformation
    suffix : str
	suffix of the slice images (Default: '.tif')
    convert_to_gray: boolean
	if True the slices are read and saved in greyscale (Default: True)

    """

    if (os.path.isdir(target_view)):
        filesTarget = sorted([target_view + '/' + f for f in os.listdir(target_view) if f[0] != '.' and f.endswith(suffix)])
        img_z = np.asarray(Image.open(filesTarget[0]))
        height_target, width_target = img_z.shape
        depth_target = len(filesTarget)
    elif (tables.isHDF5File(target_view)):
        hf5_target_view = tables.openFile(target_view, 'r')
        depth_target, height_target, width_target = hf5_target_view.root.full_image.shape
        hf5_target_view.close()
    else:
        raise Exception('%s is neither a hdf5 file nor a valid directory'%input_view)

    if (os.path.isdir(input_view)):
        filesInput = sorted([input_view + '/' + f for f in os.listdir(input_view) if f[0] != '.' and f.endswith(suffix)])
        img_z = np.asarray(Image.open(filesInput[0]))
        height_input, width_input = img_z.shape
        depth_input = len(filesInput)
        pixels_input = np.empty(shape=(depth_input, height_input, width_input), dtype=np.uint8)
        for z, image_file in enumerate(filesInput):
            img_z = Image.open(image_file)
            if convert_to_gray:
                img_z = img_z.convert('L')
            pixels_input[z, :, :] = np.asarray(img_z)
    elif (tables.isHDF5File(input_view)):
        hf5_input_view = tables.openFile(input_view, 'r')
        depth_input, height_input, width_input = hf5_input_view.root.full_image.shape
        pixels_input = hf5_input_view.root.full_image[0:depth_input,0:height_input,0:width_input]
        hf5_input_view.close()
    else:
        raise Exception('%s is neither a hdf5 file nor a valid directory'%input_view)


    pixels_transformed_input = np.empty((depth_target, height_target, width_target), dtype=np.uint8)

    h5_output=False
    if (transformed_view.endswith('.h5')):
        target_shape = (depth_target, height_target, width_target)
        atom = tables.UInt8Atom()
        h5f = tables.openFile(transformed_view, 'w')
        ca = h5f.createCArray(h5f.root, 'full_image', atom, target_shape)
        h5_output=True
    else:
        if not os.path.exists(transformed_view):
            os.makedirs(transformed_view)
        else:
            files = glob.glob(transformed_view + '/*')
            for f in files:
                os.remove(f)
    
    coords_2d_target = np.vstack(np.indices((width_target,height_target)).swapaxes(0,2).swapaxes(0,1))
    invR = np.linalg.inv(R)
    invR_2d_transpose = np.transpose(np.dot(invR[:, 0:2], np.transpose(coords_2d_target - t[0:2])))
    pbar = ProgressBar(maxval=depth_target,widgets=['Rotating %d slices: ' % (depth_target),
                                Percentage(), ' ', ETA()])
    rangez=range(0,depth_target,1)
    for z in pbar(rangez):
        R_t_3d = np.transpose(invR_2d_transpose + invR[:, 2] * (z - t[2]))
        good_indices = np.arange(R_t_3d.shape[1])
        good_indices = good_indices[(R_t_3d[0, :] > 0) * (R_t_3d[1, :] > 0) * (R_t_3d[2, :] > 0) * (R_t_3d[0, :] < (width_input - 1)) * (R_t_3d[1, :] < (height_input - 1)) * (R_t_3d[2, :] < (depth_input - 1))]
        R_t_3d = R_t_3d.take(good_indices,axis=1)
        R_t_3d = np.round(R_t_3d).astype(int)
        coords_2d_target_tmp = coords_2d_target.take(good_indices, axis=0)
        coords_3d_target_tmp = np.hstack((coords_2d_target_tmp, np.ones((coords_2d_target_tmp.shape[0], 1)).astype(int) * z))
        pixels_transformed_input[coords_3d_target_tmp[:, 2], coords_3d_target_tmp[:, 1], coords_3d_target_tmp[:, 0]] = pixels_input[R_t_3d[2, :], R_t_3d[1, :], R_t_3d[0, :]]
        if h5_output:
            ca[z, :, :] = pixels_transformed_input[z,:,:]
        else:
            im = Image.fromarray(np.uint8(pixels_transformed_input[z]))
            im.save(transformed_view + '/slice_' + str(z).zfill(4) + ".tif", 'TIFF')
    if h5_output:
        h5f.close()


def fuse_tensors(outdir,pixels_redChannel,pixels_greenChannel,pixels_blueChannel):
    """
    Method that fuses multiple tensors in a RGB tensor

    Parameters
    ----------
    
    pixels_redChannel: numpy tensor
	numpy tensor of the first view (R channel)
    pixels_greeChannel: numpy tensor
	numpy tensor of the second view (G channel)
    pixels_blueChannel: numpy tensor
	numpy tensor of the third view (B channel)
    outdir: str
	directory where the output tensor will be saved
    """
    mkdir_p(outdir)
    num_slices=pixels_redChannel.shape[0]
    for z in xrange(0, num_slices, 1):
        pixels_merged = cv2.merge((pixels_redChannel[z], pixels_greenChannel[z], pixels_blueChannel[z]))
        im = Image.fromarray(pixels_merged)
        tempname = '/tmp/'+str(uuid.uuid4())+'.tif'
        im.save(tempname)
        destname = outdir+'/slice_'+str(z).zfill(4)+'.tif'
        os.system('tiffcp -clzw:2 ' + tempname + ' ' + destname)
        os.remove(tempname)
    print('...fusion computed (%s slices) ' %z)
    print('fused stack saved in ', outdir)

