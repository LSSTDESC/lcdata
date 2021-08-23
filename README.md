# lcdata

Tools for manipulating large datasets of astronomical light curves

[![Build Status](https://github.com/kboone/lcdata/actions/workflows/run_tests.yml/badge.svg)](https://github.com/kboone/lcdata/actions/workflows/run_tests.yml)
[![Coverage Status](https://codecov.io/gh/kboone/lcdata/branch/main/graph/badge.svg?token=BDXNHT0P7G)](https://codecov.io/gh/kboone/lcdata)
[![Documentation Status](https://readthedocs.org/projects/lcdata/badge/?version=latest)](https://lcdata.readthedocs.io/en/latest/?badge=latest)

## About

`lcdata` is a package for manipulating large datasets of astronomical time series. `lcdata` is designed to handle very large datasets: it uses a compact internal representation to be able to keep many light curves in memory at the same time. For datasets that are too large to fit in memory, it offers the option of reading them from disks in chunks. `lcdata` also contains tools to download different publicly available releases of astronomical time series.

## Installation and Usage

Instructions on how to install and use `lcdata` can be found on the [lcdata
readthedocs page](https://lcdata.readthedocs.io/en/latest/).
