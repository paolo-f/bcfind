.. _scripts:

Main scripts
************

Creating substacks
------------------

make_substacks.py
^^^^^^^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.make_substacks
   :func: get_parser
   :prog: make_substacks.py
          
Finding somata
--------------

find_cells.py
^^^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.find_cells
   :func: get_parser
   :prog: find_cells.py

Performance evaluation and marker files
---------------------------------------

eval_perf.py
^^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.eval_perf
   :func: get_parser
   :prog: eval_perf.py

results_table.py
^^^^^^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.results_table
   :func: get_parser
   :prog: results_table.py

merge_markers.py
^^^^^^^^^^^^^^^^

.. argparse::
    :module: bcfind.scripts.merge_markers
    :func: get_parser
    :prog: merge_markers.py

Semantic deconvolution
----------------------


tif2HDF5.py
^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.tif2HDF5
   :func: get_parser
   :prog: tif2HDF5.py



make_sup_dataset.py
^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.make_sup_dataset
   :func: get_parser
   :prog: make_sup_dataset.py


run_semantic_deconvolution.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
   :module: bcfind.scripts.run_semantic_deconvolution
   :func: get_parser
   :prog: run_semantic_deconvolution.py


Manifold modeling
-----------------

fast_main_patching.py
^^^^^^^^^^^^^^^^^^^^^

.. argparse::
    :module: bcfind.scripts.fast_main_patching
    :func: get_parser
    :prog: fast_main_patching.py

single_patch.py
^^^^^^^^^^^^^^^

.. argparse::
    :module: bcfind.scripts.single_patch
    :func: get_parser
    :prog: single_patch.py

main_produce_cleaned_marker.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. argparse::
    :module: bcfind.scripts.main_produce_cleaned_marker
    :func: get_parser
    :prog: main_produce_cleaned_marker.py

delete_fp.py
^^^^^^^^^^^^

.. argparse::
    :module: bcfind.scripts.delete_fp
    :func: get_parser
    :prog: delete_fp.py
