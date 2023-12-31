Tutorials
=========

We provide an example going through how to use ``scSemiProfiler`` to preprocess and semi-profile a small dataset with 12 COVID-19 samples from patients of 6 different severity levels (stored in the `example_data <https://github.com/mcgilldinglab/scSemiProfiler/tree/main/example_data>`_ folder in our GitHub repository). To semi-profile a cohort, the following steps will be executed: (1) initial setup, which includes preprocessing and clustering bulk data, and selecting initial representatives; (1.5) obtaining single-cell data for representatives; (2) processing single-cell data and performing feature augmentations; (3) single-cell inference using deep generative models.

Then, once the inference is complete, the semi-profiled cohort can be utilized for various single-cell-level downstream analyses and compared with the results of the real-profiled cohort. The high similarity between the real and semi-profiled versions demonstrates the reliable performance of scSemiProfiler. If the budget allows, you have the option to employ an active learning algorithm to select additional representatives and proceed to the next round of semi-profiling. As more representatives are selected, the semi-profiling performance typically improves, but the costs also increase. We illustrate this trade-off relationship with an overall error versus cost curve.

You can also download our `GitHub repository <https://github.com/mcgilldinglab/scSemiProfiler>`_ and run the `example <https://github.com/mcgilldinglab/scSemiProfiler/blob/main/example.ipynb>`_ locally. Before running the notebook, your need to install scSemiProfiler and then install the conda environment as a Jupyter Notebook kernel::

    conda install ipykernel
    python -m ipykernel install --user --name=semiprofiler --display-name="scSemiProfiler"

Then open the notebook. You can now select the kernel "scSemiProfiler" in Jupyter Notebook and run our example notebook.


.. toctree::
    example.ipynb
