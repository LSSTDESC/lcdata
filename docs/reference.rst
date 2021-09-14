***************
Reference / API
***************

.. currentmodule:: lcdata


Datasets
========

.. autosummary::
   :toctree: api

   Dataset
   LightCurveMetadata
   HDF5Dataset
   HDF5LightCurves

*Loading a dataset*

.. autosummary::
   :toctree: api

   from_light_curves
   from_observations
   from_avocado
   read_hdf5

*Manipulating light curves*


.. autosummary::
   :toctree: api

   parse_light_curve
   to_sncosmo
   lightcurve.generate_object_id


Schemas
=======

.. autosummary::
   :toctree: api

   schema.verify_schema
   schema.get_default_value
   schema.find_alias
   schema.format_table


Utilities
=========

.. autosummary::
   :toctree: api

   utils.download_file
   utils.download_zenodo
