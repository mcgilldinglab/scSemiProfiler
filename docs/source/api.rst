API documentation
=================

This section provides detailed API documentation for all public functions
and classes in ``seSemiProfiler``.



Initial Setup
~~~~~


.. module:: scSemiProfiler.initial_setup
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.initial_setup.initsetup



Get Representatives Single-cell (used in example)
~~~~~


.. module:: scSemiProfiler.get_eg_representatives
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.get_eg_representatives.get_eg_representatives


Single-cell Processing & Feature Augmentation
~~~~~


.. module:: scSemiProfiler.singlecell_process
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.singlecell_process.scprocess


Single-cell Inference
~~~~~


.. module:: scSemiProfiler.inference
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.inference.tgtinfer
   scSemiProfiler.inference.scinfer


Representatives Selection
~~~~~


.. module:: scSemiProfiler.representative_selection
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.representative_selection.activeselection


Global Mode
~~~~~


.. module:: scSemiProfiler.utils
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.utils.inspect_data
   scSemiProfiler.utils.global_stop_checking



Utils - Downstream Analysis
~~~~~


.. module:: scSemiProfiler.utils
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.utils.estimate_cost
   scSemiProfiler.utils.visualize_recon
   scSemiProfiler.utils.visualize_inferred
   scSemiProfiler.utils.loss_curve
   scSemiProfiler.utils.assemble_cohort
   scSemiProfiler.utils.assemble_representatives
   scSemiProfiler.utils.compare_umaps
   scSemiProfiler.utils.compare_adata_umaps
   scSemiProfiler.utils.celltype_proportion
   scSemiProfiler.utils.composition_by_group
   scSemiProfiler.utils.geneset_pattern
   scSemiProfiler.utils.celltype_signature_comparison
   scSemiProfiler.utils.rrho
   scSemiProfiler.utils.enrichment_comparison
   scSemiProfiler.utils.get_error
   scSemiProfiler.utils.errorcurve

   

Utils - Statistics
~~~~~


.. module:: scSemiProfiler.utils
.. currentmodule:: scSemiProfiler

.. autosummary::
   :toctree: api

   scSemiProfiler.utils.comb
   scSemiProfiler.utils.hyperp
   scSemiProfiler.utils.hypert
   scSemiProfiler.utils.faiss_knn
