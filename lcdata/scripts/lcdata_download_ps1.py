#!/usr/bin/env python

"""Download and preprocess the PS1 dataset from Villar et al. 2020."""

import argparse
import astropy
import astropy.table
import bz2
import io
import os
import re
import numpy as np
import sncosmo
import zipfile

try:
    import importlib_resources as resources
except ImportError:
    from importlib import resources

import lcdata
import lcdata.datasets


def update_bands(table):
    band_map = {
        'g': 'ps1::g',
        'r': 'ps1::r',
        'i': 'ps1::i',
        'z': 'ps1::z',
    }
    update_func = np.vectorize(band_map.get)
    table['FLT'] = update_func(table['FLT'])


def parse_light_curve(file, metadata):
    # Read the light curve
    lc = sncosmo.read_snana_ascii(file, default_tablename='OBS')
    lc_meta, lc_obs = lc

    lc_meta = lc[0]
    lc_obs = lc[1]['OBS']

    update_bands(lc_obs)

    # Find the matching metadata
    meta_match = metadata[metadata['object_id'] == lc_meta['SNID']]
    assert len(meta_match) == 1
    lc_obs.meta = dict(meta_match[0])

    return lc_obs


def download_ps1(directory):
    rawdir = os.path.join(directory, 'ps1_raw')

    print("Downloading the PS1 dataset from zenodo...")
    lcdata.utils.download_zenodo("3974950", rawdir)

    print("Parsing metadata...")
    with resources.files(lcdata.datasets).joinpath('vav2020_table1_patched.tex.bz2').open('rb') as compressed:
        with bz2.open(compressed, 'rt') as file:
            raw_metadata = np.genfromtxt(file, delimiter=' & ', skip_header=7,
                                         skip_footer=8, dtype=None, encoding='ascii',
                                         missing_values='-', usemask=True)

    metadata = astropy.table.Table(
        raw_metadata,
        names=[
            'object_name', 'object_id', 'iau_name', 'cbet', 'ra', 'dec', 'mwebv',
            'type', 'redshift', 'host_ra', 'host_dec', 'num_points', 'telescope',
            'unsupervised', 'supervised'
        ]
    )
    metadata['supervised'] = [i[0] == 'Y' for i in metadata['supervised']]
    metadata['unsupervised'] = [i[0] == 'Y' for i in metadata['unsupervised']]

    print("Parsing light curves...")
    light_curves = []
    with zipfile.ZipFile(os.path.join(rawdir, "ps1_sne_zenodo.zip"), "r") as zfile:
        filenames = [filename for filename in zfile.namelist() if
                     re.match('ps1_sne_zenodo/PS1_.*\\.snana.dat', filename)]
        for filename in lcdata.utils.tqdm(filenames):
            with zfile.open(filename) as file:
                lc = parse_light_curve(io.TextIOWrapper(file), metadata)
                light_curves.append(lc)

    print("Writing dataset...")
    dataset_path = os.path.join(directory, 'ps1.h5')
    dataset = lcdata.from_light_curves(light_curves)
    dataset.write_hdf5(dataset_path, overwrite=True)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Download the PS1 dataset from Zenodo."
    )
    parser.add_argument('--directory', default='./data/')

    args = parser.parse_args()
    download_ps1(args.directory)


if __name__ == "__main__":
    main()
