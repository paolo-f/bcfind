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
      
The training set will be saved in the HDF5 file ``training-set.h5``. Now you may use it for
training a neural network using
`pylearn2 <http://deeplearning.net/software/pylearn2/>`_. For this
purpose, the YAML files in the folder ``demo-yaml`` may be a helpful
starting point. We found that pretraining two layers as RBMs is
effective but you may want to try other alternatives. See the pylearn2
documentation for this purpose.

.. note:: The neural network must take as input a 3D patch and respond with an output 3D patch of the same size.

Once the neural network is trained, you may apply it to your volume.
Pylearn2 will typically save trained models as Python pickle files.
Suppose the network was saved as ``trained-network.pkl``. Then we can obtain
the deconvolved image for substack ``020305`` as follows:

.. code-block:: console

    $ run_semantic_deconvolution.py ${DATA_DIR}/substacks/mouse1/cerebellum/ \
    020305 ${DATA_DIR}/mouse1/cerebellum.h5 trained-network.pkl \
    train-set.h5 ${DECONVOLVED_DIR}

Again, this step can be parallelized on a cluster.
One possibility is to use `IPython <http://ipython.org/>`_.
First, create a script ``parallel_sem.py`` as follows:

.. code-block:: python

   from IPython.parallel import Client

   def submit(substack):
       import argparse
       import scripts.run_semantic_deconvolution
       parser = scripts.run_semantic_deconvolution.get_parser()
       args = parser.parse_args(['/my/data/substacks/mouse1/cerebellum',
                                 substack, '/my/data/mouse1/cerebellum.h5', 'trained-network.pkl',
                                 'train-set.h5', '/my/data/mouse1/cerebellum/deconvolved/',
                                 '--speedup', '4', '--extramargin', '6'])
       scripts.run_semantic_deconvolution.main(args)

   c = Client()
   NX, NY, NZ = 15, 40, 15 # be consistent with your choice in make_substacks.py
   substacks = ['%02d%02d%02d'%(x,y,z) for x in range(NX) for y in range(NY) for z in range(NZ)]
   view = c.load_balanced_view()
   r = view.map(submit, substacks)


Then start ipython cluster, e.g.

.. code-block:: console

   $ ipcluster start -n 8


Finally run from ipython and monitor its execution using ``view.queue_status()``:

.. ipython::
   :verbatim:

   In [1]: run parallel_sem.py
   
   

Once semantic deconvolution is completed, you may run ``cell_find.py``
on the preprocessed images. It's predictive performance should be
improved.

===============
Manifold filter
===============

:WRITEME:
