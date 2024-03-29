#!/usr/bin/env python

"""Download and preprocess the PLAsTiCC dataset. We convert the CSV files that the
PLAsTiCC dataset comes in to HDF5 files that we can work with much more easily and that
are stored compressed on disk.
"""

import argparse
import astropy
import astropy.table
import gzip
import os
import time
import numpy as np

import lcdata


def read_file(rawdir, name):
    path = os.path.join(rawdir, name)
    with gzip.open(path) as gzip_file:
        table = astropy.table.Table.read(gzip_file, format='csv')

    return table


def update_object_id(table):
    format_func = np.vectorize(lambda i: f'PLAsTiCC {i:09d}')
    table['object_id'] = format_func(table['object_id'])


def update_bands(table):
    band_map = {
        0: 'lsstu',
        1: 'lsstg',
        2: 'lsstr',
        3: 'lssti',
        4: 'lsstz',
        5: 'lssty',
    }
    update_func = np.vectorize(band_map.get)
    table['passband'] = update_func(table['passband'])


def update_classes(table):
    class_map = {
        6: 'muLens-Single',
        15: 'TDE',
        16: 'EB',
        42: 'SNII',
        52: 'SNIax',
        53: 'Mira',
        62: 'SNIbc',
        64: 'KN',
        65: 'M-dwarf',
        67: 'SNIa-91bg',
        88: 'AGN',
        90: 'SNIa',
        92: 'RRL',
        95: 'SLSN-I',
        99: 'Unknown',
        991: 'muLens-Binary',
        992: 'ILOT',
        993: 'CaRT',
        994: 'PISN',
        995: 'muLens-String',
    }
    update_func = np.vectorize(class_map.get)
    table['true_target'] = update_func(table['true_target'])


def download_plasticc(directory, train_only=False):
    rawdir = os.path.join(directory, 'plasticc_raw')

    print("Downloading the PLAsTiCC dataset from zenodo...")
    lcdata.utils.download_zenodo("2539456", rawdir)

    print("Processing training dataset...")
    train_path = os.path.join(directory, 'plasticc_train.h5')
    train_meta = read_file(rawdir, 'plasticc_train_metadata.csv.gz')
    train_observations = read_file(rawdir, 'plasticc_train_lightcurves.csv.gz')
    update_classes(train_meta)
    update_object_id(train_meta)
    update_object_id(train_observations)
    update_bands(train_observations)
    dataset = lcdata.from_observations(train_meta, train_observations)
    dataset.write_hdf5(train_path, overwrite=True)

    if train_only:
        print("Skipping test dataset processing.")
        print("\nDone!")
        return

    print("Processing test dataset...")
    test_path = os.path.join(directory, 'plasticc_test.h5')
    test_meta = read_file(rawdir, 'plasticc_test_metadata.csv.gz')
    update_classes(test_meta)
    update_object_id(test_meta)

    for test_idx in range(1, 12):
        print(f"Processing chunk {test_idx:2d}/11...", end=' ', flush=True)
        start_time = time.time()

        test_observations = read_file(
            rawdir, f'plasticc_test_lightcurves_{test_idx:02d}.csv.gz'
        )
        update_object_id(test_observations)
        update_bands(test_observations)
        dataset = lcdata.from_observations(test_meta, test_observations)
        if test_idx == 1:
            # Create a new file
            dataset.write_hdf5(test_path, overwrite=True)
        else:
            # Append to the current file
            dataset.write_hdf5(test_path, append=True)

        elapsed_time = time.time() - start_time
        print(f'done in {elapsed_time:.1f}s')

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Download the PLAsTiCC dataset from Zenodo."
    )
    parser.add_argument('--directory', default='./data/',
                        help="Directory to download the dataset to.")
    parser.add_argument('--train-only', action='store_true',
                        help="Download both datasets, but only process the training set. Mostly for testing.")

    args = parser.parse_args()
    download_plasticc(args.directory, train_only=args.train_only)


if __name__ == "__main__":
    main()
