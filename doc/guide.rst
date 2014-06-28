.. _guide:

User guide
**********

Throughout this section we assume that bcfind has been setup in your system as described in :ref:`installation`.

==============
Make substacks
==============


..
   sidebar:: Why splitting into substacks?

   There are several reasons:
   
   - Parallelization on computer clusters
   - Finding a good foreground threshold is impossible on the whole
     image due to large contrast variabilities associated with brain
     clearing and CLSM. Good thresholds can be found on substacks
     without resorting to costly local thresholding algorithms.
   - Ground truth is necessary for supervised semantic deconvolution
     and for performance evaluation. Annotating large images is
     impractical/impossible.
   - Substacks can be easily visualized for debugging purposes

In this step you will split your volume into smaller substacks. For this purpose
use the program ``make_substacks.py`` as follows:

.. code-block:: console

    $ export DATA_DIR=/my/data/
    $ make_substacks.py ${DATA_DIR}/mouse1/cerebellum NX NY NZ

where ``NX``, ``NY``, and ``NZ`` are the number of splits along the three
dimensions.  The input directory ``${WHOLE_STACK_DIR}/mouse1/cerebellum`` should
contain TIFF files forming an image sequence.  File names should
contain a prefix, and underscore, and consecutive integers (the
initial value can be any) representing Z coordinates. For example:

.. code-block:: console

    $ ls -l ${DATA_DIR}/mouse1/cerebellum

    total 29290796
    -rw-r--r-- 1 paolo paolo  4641180 Dec 31 23:52 full_013244.tif
    -rw-r--r-- 1 paolo paolo  4685040 Jan  1 00:48 full_013245.tif
    -rw-r--r-- 1 paolo paolo  4688068 Dec 31 23:33 full_013246.tif
    -rw-r--r-- 1 paolo paolo  4716752 Dec 31 23:43 full_013247.tif
    ....

The program ``make_substacks.py`` will create substacks in your data
dir: for example ``${DATA_DIR}/substacks/mouse1/cerebellum/020305/``
will contain the 3rd (along x), 4th (along y) and 6th (along z)
substack as a sequence of TIFF files.

.. _run-cell-finder:

===============
Run cell finder
===============

.. code-block:: console

    $ find_cells.py ${DATA_DIR}/substacks/mouse1/cerebellum/ 020305 ${RESULTS_DIR}

The program will seek cell soma on substack ``020305`` and save centers as a `Vaa3D
<http://www.vaa3d.org/>`_ marker file in
``${RESULTS_DIR}/020305/ms.marker``
Logs and debug information are stored in the same directory.

.. hint:: If you have a cluster you should process several substacks in parallel
          
===================
Measure performance
===================

First, create ground truth files for some of your substacks. You
should use `Vaa3D <http://www.vaa3d.org/>`_ for this purpose and save
files in `.marker` format in the substacks directory using this naming
convention: ``${DATA_DIR}/substacks/mouse1/cerebellum/020305-GT.marker``

At this point you can measure performance by running:

.. code-block:: console

    $ eval_perf.py ${DATA_DIR}/substacks/mouse1/cerebellum/ 020305 ${RESULTS_DIR}

The script will show precision, recall, and F1-measure on the
substack. In the file ``${RESULTS_DIR}/020305/errors.marker`` you will
also get a Vaa3d marker file with both predicted and ground truth
markers. This file can be inspected in the 3D view of Vaa3D. Colors are defined as follows:


===========  ================================================
Color        Meaning
===========  ================================================
Cyan         True positives
Red          False positives
Magenta      False negatives
Orange       False positives removed by the manifold filter
Green        Ground truth
===========  ================================================




=================================
Supervised semantic deconvolution
=================================

The goal of semantic deconvution is to enhance and standardize the
visibility of specific entities of interest in the image (somata in
our case). The semantic deconvution modules read the raw volume as
subtensors from HDF5 files rather than using TIFF files. To convert a
TIFF stack into a corresponding HDF5 file run the script

.. code-block:: console

    $ tif2HDF5.py ${DATA_DIR}/mouse1/cerebellum ${DATA_DIR}/mouse1/cerebellum.h5

In order to train the neural network you must first create a suitable
data set. First, you need a set of labeled substacks.

.. warning:: It is important that *every* soma is marked in the
             training substacks (unmarked somata add supervision noise
             to the neural network training).

After having labeled enough data (for example substacks ``040604``
``091205`` ``110207``), run the script ``make_sup_dataset.py`` as
follows:

.. code-block:: console

    $ make_sup_dataset.py --negatives ${DATA_DIR}/substacks/mouse1/cerebellum/\
    ${DATA_DIR}/mouse1/cerebellum.h5 training-set.h5 040604 091205 110207
      
The training set will be saved in the HDF5 file
``training-set.h5``. Now you may use it for training a neural network
using `pylearn2 <http://deeplearning.net/software/pylearn2/>`_. For
this purpose, the YAML files in the folder ``demo-yaml`` may be a
helpful starting point. We found that pretraining two layers as RBMs
is effective but you may want to try other alternatives such as
denoising or contractive autoencoders or even avoid pretraining. See
the pylearn2 documentation for this purpose. In our setting, training
takes several hours on GPU.

.. note:: The neural network must take as input a 3D patch and respond with an output 3D patch of the same size.

Once the neural network is trained, you may apply it to your volume.
Pylearn2 will typically save trained models as Python pickle files.
Suppose the network was saved as ``trained-network.pkl``. Then we can obtain
the deconvolved image for substack ``020305`` as follows:

.. code-block:: console

    $ run_semantic_deconvolution.py ${DATA_DIR}/substacks/mouse1/cerebellum/ \
    020305 ${DATA_DIR}/mouse1/cerebellum.h5 trained-network.pkl \
    train-set.h5 ${DECONVOLVED_DIR}

Again, this step can be parallelized on a cluster. The forward pass of
the neural network can be run on GPU if available.

Once semantic deconvolution is completed, you may run ``cell_find.py``
on the preprocessed images. Its predictive performance should be
improved.

.. _merging-markers:

==================================
Merging markers from all substacks
==================================

When the ``find_cells.py`` script has run in each substack, as shown in section :ref:`_run-cell-finder` for a single substack, 
the user has to merge all the produced markers file in order to obtain a single file containing, for example, the point cloud of 
Purkinje somata of the mouse cerebellum.
The ``merge_markers.py`` script serves this purpose. Its usage is

.. code-block:: console
    
    $ export MERGED_DATA_DIR=/my/merged/data
    $ merge_markers.py ${DATA_DIR}/substacks/mouse1/cerebellum/ ${RESULTS_DIR} ${MERGED_DATA_DIR}/your_merged_filename.marker


The ``merge_markers.py`` script has a ``--verbose`` option for debug purposes.
Merging markers is a mandatory step for the application of the Manifold Filter, explained in section :ref:`_manifold-filter`.

.. _manifold-filter:

===============
Manifold filter
===============

The goal of the Manifold Filter is to exploit the manifold structure of some type of brain cells in order to remove false positives produced by the cell finder.
The method has been tested using a whole mouse cerebellum dataset, which shows a strong manifold structure.
It has been found very effective on removing false positives of such Purkinje somata.

This section explains how to use the manifold filter included in our bcfind software.
It assumes that a merged markers file has been produced, i.e. a ``${MERGED_DATA_DIR}/your_merged_filename.marker`` which contains the whole dataset, 
as shown in section :ref:`_merging-markers`.

In ``${BCFIND_INSTALL_DIR}/manifold`` you can find the ``parameters.py`` file, which contains the more or less stable parameters of the manifold filter, along with their meaning. Such parameters are tuned for our experiments.

The first script that needs to be called is ``fast_main_patching.py``:

.. code-block:: console
    $ export OUTPUT_FOLDER=/where/to/save/results
    $ fast_main_patching.py ${MERGED_DATA_DIR}/your_merged_filename.marker ${OUTPUT_FOLDER}

Such script is a sort of preprocessing that analyzes the markers in ``${MERGED_DATA_DIR}/your_merged_filename.marker`` and output relevant information for the subsequent step of the filter.
In particular, it outputs a file, whose filename is the number of nearest neighbors needed to build the data graph, in ``${OUTPUT_FOLDER}/nn``.
Moreover, in ``${OUTPUT_FOLDER}/seeds`` you will find a list of folders whose names are ``{0, 1, ..., (jobs-1)}``, where ``jobs`` is a parameter defined in the aforementioned ``parameters.py`` file.

The next script to be called is ``single_patch.py`` and it needs to be called multiple times, in particular once for each folder contained in ``${OUTPUT_FOLDER}/seeds``.
It is easy to understand that this step can be parallelized: depending on how many cores you have at your disposal, you can set the ``jobs`` parameter in ``parameters.py`` to create the corrisponding number of folders in ``${OUTPUT_FOLDER}/seeds``.
The ``single_patch.py`` script is the one which actually computes the distances to do the filtering.
You can call it in this way:

.. code-block:: console
    $ export CHARTS_FOLDER=/where/to/save/processed/charts
    $ # MAX_DISTANCE: maximum geodesic radius for creating charts with Uniform Cost Search
    $ export MAX_DISTANCE=500
    $ # SIGMA: the parameter of the Gaussian Kernel used in Lowess Regression
    $ export SIGMA=30
    $ single_patch.py ${MERGED_DATA_DIR}/your_merged_filename.marker ${CHARTS_FOLDER} ${MAX_DISTANCE} `cd ${OUTPUT_FOLDER}/nn && ls` ${OUTPUT_FOLDER}/seeds/?

where ``?`` in the last parameter can be changed with the aforementioned folders names ``{0, 1, ..., (jobs    -1)}``.
The ``single_patch.py`` script has a ``--debug`` option for debug purposes.
Using it will create CSV debug files for each chart.
Such files will be saved in special folders inside ``${CHARTS_FOLDER}``.

Once you have finished this step, ``${CHARTS_FOLDER}`` will contain all the processed charts.
``main_produce_cleaned_marker.py`` will merge all of such charts into a single global file.
Its usage is

.. code-block:: console
    $ export FINAL_OUTPUT_FOLDER=/where/to/save/final/results
    $ main_produce_cleaned_marker.py ${CHARTS_FOLDER} ${FINAL_OUTPUT_FOLDER}

In ``${FINAL_OUTPUT_FOLDER}`` you will find the final markers file, named ``cleaned.marker``.
Again, this script has a ``--debug`` option which will save a CSV file in the same folder.

These final files have a ``distance`` column: simply delete rows that have such value above a ``${THRESHOLD}`` of your choice to discard false positives.
You can do this using the ``delete_fp.py`` script, which is very straightforward to use:

.. code-block:: console
    $ export THRESHOLD=20
    $ delete_fp.py ${FINAL_OUTPUT_FOLDER}/cleaned.marker ${FINAL_OUTPUT_FOLDER} ${THRESHOLD}

In ``${FINAL_OUTPUT_FOLDER}`` you will find the filtered file.
