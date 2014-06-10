.. _installation:

Installation
************

Get sources
===========

Sources are hosted on `bitbucket
<http://bitbucket.org/paolo-f/bcfind>`_. To obtain the code, clone the
repository in some place in your system:

.. code-block:: console

   $ git clone https://paolo-f@bitbucket.org/paolo-f/bcfind.git
   $ export BCFIND_INSTALL_DIR=`pwd`/bcfind

and set the following environment variables in your shell startup script:

.. code-block:: bash

    export PYTHONPATH=${BCFIND_INSTALL_DIR}:${PYTHONPATH}
    export PATH=${BCFIND_INSTALL_DIR}/bcfind/scripts:${PATH}


Dependencies
============
* Recent versions of the following Python packages are required for
  various operations: `PIL`, `tables`, `pandas`, `scikit-learn`,
  `progressbar-latest`, `numpy`, `scipy`, `mahotas`, `ujson`. They are
  all available on the `Python Package Index <https://pypi.python.org/pypi/pip>`_.
  
* `pylearn2 <http://deeplearning.net/software/pylearn2//>`_ and its
  dependencies (in particular `Theano
  <http://deeplearning.net/software/theano/>`_) are required for
  semantic deconvolution. Install them from the git repositories as
  earlier versions (e.g. those obtainable via `pip` or `conda`) may
  not work with our code.

* We recommend working with a Python distribution with optimized
  libraries for `numpy` and `scipy`.  `Enthough Canopy
  <https://www.enthought.com/products/canopy/>`_ and `Continuum
  Analytics Anaconda <http://continuum.io/downloads>`_ both include MKL
  optimization and are freely available for academic use.


Optional dependencies
=====================

* If installed, `minizinc <http://www.minizinc.org/>`_ is used for
  colorizing detected soma in the debug stacks saved by
  ``find_cells.py`` (with the option ``--save_image``): soma that are
  close in space will get distinct hues. If minizinc is not available,
  detected soma are colored using random hues.

License
=======
The code is released under the `GPLv3 <http://gplv3.fsf.org//>`_
If you use this software in a scientific context, please cite
the following paper_.

.. _paper: additional.html
