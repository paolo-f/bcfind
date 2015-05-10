__author__ = 'paciscopi'

import numpy as np
import cPickle as pickle
import math
import os
import tables
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from pylearn2.models.autoencoder import *
from pylearn2.models.mlp import MLP
from pylearn2.models.rbm import *

import argparse
from pylearn2.utils import serial
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse
from pylearn2.datasets import control
import sys


def main(args):


    model = pickle.load(open(args.model_file))
    params = model.get_params()


    if isinstance(model, RBM) or isinstance(model,GaussianBinaryRBM):
        W = params[0].get_value()
    elif isinstance(model, MLP):
        W = params[0].get_value()
    else:
        raise TypeError('Model has type', type(model))

    W=W.T
    h = W.shape[0]
    v = W.shape[1]

    num_rows_W = v
    v = v/2

    #    hbias = params[1].get_value()
    #elif isinstance(model, DenoisingAutoencoder):
    #    W = params[2].get_value()
    patch_side = int(np.ceil(math.pow(v, 1/3.0)))
    patch_shape = (patch_side, patch_side)

    is_color = False
    idx = range(patch_side)
    patch_rescale = True

    num_filters_to_be_displayed = min(args.numfilters, h)

    filename_model = os.path.basename(args.model_file)


    hr = num_filters_to_be_displayed
    hc = patch_side
    grid_shape = (hr, hc)

    pv_1view = patch_viewer.PatchViewer(grid_shape, patch_shape, is_color)
    pv_2view = patch_viewer.PatchViewer(grid_shape, patch_shape, is_color)

    for i in range(0, num_filters_to_be_displayed):
        patch_2d_vector_list_1view = np.hsplit(W[i, 0:v], patch_side)
        patch_2d_vector_list_2view = np.hsplit(W[i, v:num_rows_W], patch_side)

        for j in range(0, patch_side):
            patch = np.array(np.hsplit(patch_2d_vector_list_1view[j], patch_side))
            pv_1view.add_patch(patch, patch_rescale)

        for j in range(0, patch_side):
            patch = np.array(np.hsplit(patch_2d_vector_list_2view[j], patch_side))
            pv_2view.add_patch(patch, patch_rescale)


    if not os.path.exists(args.weights_out_path):
        os.makedirs(args.weights_out_path)


    pv_1view.save(args.weights_out_path+"/"+filename_model+"_hidden_first_"+str(hr)+"_1view_filters.png")
    pv_2view.save(args.weights_out_path+"/"+filename_model+"_hidden_first_"+str(hr)+"_2view_filters.png")


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Print the weights of a RBM pretrained layer
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model_file', metavar='indir', type=str,
                        help='pkl file of the pretrained RBM layer')
    parser.add_argument('weights_out_path', metavar='weights_out_path', type=str,
                        help='folder in which the weights will be printed')
    parser.add_argument('--numfilters', metavar='numfilters', dest='numfilters',
                        action='store', type=int, default=50,
                        help='Number of filters that will be displayed')


    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    main(args)
