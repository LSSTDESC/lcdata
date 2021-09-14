*****
Usage
*****

Overview
========

lcdata is designed to handle large datasets of light curves. Light curves are
represented as tables in ``~astropy.table.Table`` format, and are very similar to the
ones used in sncosmo. A dataset can be created in several ways. For example, we can
create a dataset from a list of sncosmo-like light curves:

    >>> import lcdata
    >>> import sncosmo
    >>> light_curves = [sncosmo.load_example_data() for i in range(5)]
    >>> dataset = lcdata.from_light_curves(light_curves)

The individual light curves in this dataset can be accessed as ``dataset.light_curves``.

Metadata
========

The metadata associated with all of the light curves can be accessed from a common
``~astropy.table.Table`` as ``dataset.meta``:

    >>> dataset.meta
          object_id        ra dec   type  redshift  x1  c          x0           t0  
    --------------------- --- --- ------- -------- --- --- ----------------- -------
    lcdata_xbdwhv_0000000 nan nan Unknown      0.5 0.5 0.2 1.20482820761e-05 55100.0
    lcdata_xbdwhv_0000001 nan nan Unknown      0.5 0.5 0.2 1.20482820761e-05 55100.0
    lcdata_xbdwhv_0000002 nan nan Unknown      0.5 0.5 0.2 1.20482820761e-05 55100.0
    lcdata_xbdwhv_0000003 nan nan Unknown      0.5 0.5 0.2 1.20482820761e-05 55100.0
    lcdata_xbdwhv_0000004 nan nan Unknown      0.5 0.5 0.2 1.20482820761e-05 55100.0

lcdata enforces a consistent metadata format. All light curves are guaranteed to have
the following keys in their metadata.

- object_id: A unique identifer. Default: randomly assigned string
- ra: The right ascension. Default: nan
- dec: The declination. Default: nan
- type: A string representing the type of the light curve. Default: Unknown
- redshift: The redshift. Default: nan

Astronomical data comes in many different formats, and keyword usage is not
standardized. lcdata will try to find all of these keys in the metadata using a list of
known aliases.

    >>> light_curve = sncosmo.load_example_data()
    >>> light_curve.meta = {
    ...     'id': 'example_id',
    ...     'right_ascension': 1.,
    ...     'decl': 2.,
    ...     'class': 'Type Ia',
    ...     'other_var': 5.
    ... }

    >>> dataset = lcdata.from_light_curves([light_curve])
    >>> print(dataset.meta)
    object_id   ra dec   type  redshift other_var
    ---------- --- --- ------- -------- ---------
    example_id 1.0 2.0 Type Ia      nan       5.0


Light Curves
============

lcdata will standardize the format of light curves, similarly to how the metadata is
standardized. Each light curve is guaranteed to have the following keys:

- time: times at which the light curve was sampled. Converted to a 64-bit float.
- flux: The flux at each point on the light curve. Converted to a 32-bit float.
- fluxerr: The uncertainty on the flux. Converted to a 32-bit float.
- band: A string representing bandpass that the light curve was observed in. We
  recommend using the sncosmo bandpass names here. Converted to a binary string.

Additional columns are left as is. If the light curve columns have different labels,
lcdata will try to infer which ones are which using a set of aliases.

    >>> light_curve = astropy.table.Table({
    ...     'bandpass': ['lsstu', 'lsstb', 'lsstr'],
    ...     'flux': [1., 2., 10.],
    ...     'mjd': [59000., 59010., 59020.],
    ...     'fluxerr': [1., 0.5, 3.],
    ...     'myvar': [1., 2., 5.],
    ... })
    >>> print(dataset.light_curves[0])
      time  flux fluxerr  band myvar
    ------- ---- ------- ----- -----
    59000.0  1.0     1.0 lsstu   1.0
    59010.0  2.0     0.5 lsstb   2.0
    59020.0 10.0     3.0 lsstr   5.0


Dataset Manipulation
====================

Datasets can be manipulated in various ways.

Addition:

    >>> dataset = dataset1 + dataset2

Selecting a subset:

    >>> dataset = dataset[5:10]


Saving a Dataset in HDF5 format
===============================

lcdata has an optimized HDF5 reader/writer that can be used to rapidly load very large
light curve datasets.

Datasets can be read from and written out to disk in HDF5 format.

    >>> dataset.write_hdf5('./dataset.h5')

    >>> dataset = lcdata.read_hdf5('./dataset.h5')

A dataset on disk can be appended to:

    >>> dataset_2.write_hdf5('./dataset.h5', append=True)

Some datasets are too large to fit in memory all at once. lcdata can load only the
metadata of a dataset into memory, and then access the light curves themselves on
demand.

    >>> # Read only the metadata
    >>> dataset = lcdata.read_hdf5('./dataset.h5', in_memory=False)

    >>> # Read a specific light curve
    >>> light_curve = dataset.light_curves[10]

    >>> # Select a subset of the dataset and load all of its light curves into memory.
    >>> subset = dataset[1000:2000].load()

A common use case for this functionality is to process all of the light curves in the
dataset in chunks. lcdata provides a helper to do this:

    >>> for chunk in dataset.iterate_chunks(chunk_size=1000):
    ...     # At each iteration, chunk is an lcdata Dataset with the next 1000
    ...     # light curves.
