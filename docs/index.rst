.. lcdata documentation master file, created by
   sphinx-quickstart on Fri Aug 13 13:30:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lcdata
======

About
-----

lcdata is a package for manipulating large datasets of astronomical time series. lcdata
is designed to handle very large datasets: it uses a compact internal representation to
be able to keep many light curves in memory at the same time. For datasets that are too
large to fit in memory, it offers the option of reading them from disks in chunks.
lcdata also contains tools to download different publicly available releases of
astronomical time series.


.. toctree::
   :maxdepth: 1
   :titlesonly:

   installation
   usage
   reference

Source code: https://github.com/kboone/lcdata