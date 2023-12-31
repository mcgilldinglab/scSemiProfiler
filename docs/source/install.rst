Installation
============
This page includes instructions for installing scSemiProfiler.

Prerequisites
-------------

First, install `Anaconda <https://www.anaconda.com/>`_ for your operating system if you have not. You can find specific instructions for different operating systems `here <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

Second, create a new conda environment and activate it::

    conda create -n semiprofiler python=3.9
    conda activate semiprofiler

Finally, install the version of PyTorch compatible with your devices by following the `instructions on the official website <https://pytorch.org/get-started/locally/>`_.

Installing scSemiProfiler
-------------------------

There are 2 options to install scSemiProfiler.

Option 1: Install from download directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download scSemiProfiler from the `GitHub repository <https://github.com/mcgilldinglab/scSemiProfiler>`_, go to the downloaded scSemiProfiler root directory and use pip tool to install::

    pip install .

Option 2: Install from Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    pip install --upgrade https://github.com/mcgilldinglab/scSemiProfiler/zipball/main

The installation should take less than 2 minutes.
The `environment.txt <environment.txt>`_ file includes information about the environment that we used to test scSemiProfiler.



