*****************
Built-in Datasets
*****************

lcdata contains scripts that can be used to download commonly-used light curve datasets
and save them in an efficient and fast-to-read HDF5 format. The currently-supported
datasets are:

=========== ============================ =============================================================================
  Dataset          Script                     Reference
=========== ============================ =============================================================================
PLAsTiCC    ``lcdata_download_plasticc`` `PLAsTiCC <https://plasticc.org>`_
PanSTARRS-1 ``lcdata_download_ps1``      `Villar et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...905...94V>`_
=========== ============================ =============================================================================

After installing lcdata, these scripts can be run from any directory on the command
line to download the corresponding dataset(s). By default, these will be placed in
``./data/``, although the location can be changed with the ``--directory`` flag.

Using downloaded datasets
=========================

A full description of how datasets are used in lcdata can be found on the :doc:`usage`
page.

To open a dataset that can fit in memory and read metadata/light curves from it::

    >>> dataset = lcdata.read_hdf5('./data/ps1.h5')

    >>> print(dataset.meta)
    <Table length=5243>
    object_id    ra      dec      type  redshift ...
       str9   float64  float64   str13  float64  ...
    --------- -------- -------- ------- -------- ...
    PS0909006 333.9503   1.1848    SNIa    0.284 ...
    PS0909010  37.1182  -4.0789    SNIa     0.27 ...
    PS0910012  52.4718 -28.0867   SNIax    0.079 ...
    PS0910016  35.3073    -3.91    SNIa     0.23 ...
          ...      ...      ...     ...      ... ...

    >>> print(dataset.light_curves[0])
    <Table length=376>
      time    flux  fluxerr  band 
    float64 float32 float32 bytes6
    ------- ------- ------- ------
    55029.6  -9.919  11.995 ps1::g
    55074.4 211.372   33.03 ps1::g
    55086.4 112.477  10.463 ps1::g
    55089.4 102.993   10.58 ps1::g
        ...     ...     ...    ...


For large datasets that can't fit into memory, a common workflow is to process the
dataset in smaller chunks that can fit in memory. To load a large dataset in chunks of
1000 light curves::

    >>> dataset = lcdata.read_hdf5('./data/plasticc_test.h5', in_memory=False)

    >>> for chunk in dataset.iterate_chunks(chunk_size=1000):
    ...    # At each iteration, chunk is an lcdata Dataset with the next 1000
    ...    # light curves.
