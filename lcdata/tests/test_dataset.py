import pytest

from astropy.table import Table
import numpy as np
import lcdata
import os


def make_light_curve(object_id='test'):
    lc = Table({
        'time': [0, 1, 2, 5, 8.3],
        'flux': [1., 2., 3., 4., 2.3],
        'fluxerr': [1., 2., 3., 4., 2.3],
        'band': ['lsstu', 'lsstg', 'lsstr', 'lsstr', 'lsstg'],
    })

    lc.meta = {
        'object_id': object_id,
        'ra': 10.,
        'dec': 5.,
        'myvar': 1.,
        'strvar': 'test',
    }

    return lc


def make_dataset(start_idx=0, end_idx=10):
    lcs = [make_light_curve(f'test_{i}') for i in range(start_idx, end_idx)]
    dataset = lcdata.from_light_curves(lcs)

    dataset.meta['maskedvar'] = np.ma.masked_array(
        np.arange(start_idx, end_idx),
        mask=np.arange(start_idx, end_idx) % 2 == 0
    )

    return dataset


@pytest.fixture
def dataset():
    return make_dataset()


@pytest.fixture
def dataset_2():
    return make_dataset(20, 25)


@pytest.fixture
def dataset_hdf5(dataset, tmp_path):
    """Make an HDF5 dataset and write it to disk."""
    path = str(tmp_path / 'test.h5')
    dataset.write_hdf5(path, object_id_itemsize=10)

    return path


def test_from_light_curves():
    lc = make_light_curve()
    dataset = lcdata.from_light_curves([lc])

    assert len(dataset) == 1


def test_from_light_curves_order():
    # We sort the light curves in alphabetical order. Make sure that this works
    # correctly.
    lc1 = make_light_curve(object_id='1')
    lc2 = make_light_curve(object_id='2')
    lc3 = make_light_curve(object_id='3')

    # Add a flag to the fluxes so that we can track them.
    lc1['flux'][0] = 1.
    lc2['flux'][0] = 2.
    lc3['flux'][0] = 3.

    dataset = lcdata.from_light_curves([lc2, lc3, lc1])

    assert np.all(dataset.meta['object_id'] == ['1', '2', '3'])
    assert dataset.light_curves[0]['flux'][0] == 1.
    assert dataset.light_curves[1]['flux'][0] == 2.
    assert dataset.light_curves[2]['flux'][0] == 3.


def test_dataset_length(dataset):
    assert len(dataset) == 10


def test_dataset_light_curves(dataset):
    assert isinstance(dataset.light_curves[0], Table)


def test_dataset_meta(dataset):
    assert len(dataset.meta['object_id']) == 10


def test_dataset_meta_update(dataset):
    dataset.meta['object_id'][0] = 'new_id'
    assert dataset.light_curves[0].meta['object_id'] == 'new_id'


def test_dataset_lc_meta_update(dataset):
    dataset.light_curves[0].meta['object_id'] = 'new_id'
    assert dataset.meta['object_id'][0] == 'new_id'


def test_dataset_lc_meta_delete_required(dataset):
    del dataset.light_curves[0].meta['ra']
    assert np.isnan(dataset.light_curves[0].meta['ra'])


def test_dataset_lc_meta_delete(dataset):
    del dataset.light_curves[0].meta['myvar']
    assert 'myvar' not in dataset.light_curves[0].meta


def test_dataset_add_meta(dataset):
    new_meta = dataset.meta[['object_id']]
    new_meta['test'] = 2.
    dataset.add_meta(new_meta)
    assert 'test' in dataset.meta.colnames


def test_dataset_add_meta_duplicate(dataset):
    num_cols = len(dataset.meta.columns)
    new_meta = dataset.meta[['object_id', 'myvar']]
    dataset.add_meta(new_meta)
    assert len(dataset.meta.columns) == num_cols


def test_dataset_add_meta_conflict(dataset):
    new_meta = dataset.meta[['object_id', 'myvar']]
    new_meta['myvar'] = 2.
    dataset.add_meta(new_meta)
    assert 'myvar_2' in dataset.meta.colnames


def test_dataset_add_meta_bytes(dataset):
    # Sometimes we have bytes vs string columns with the same data. These should be
    # considered to be the same column if each individual string matches.
    num_cols = len(dataset.meta.columns)
    new_meta = dataset.meta[['object_id', 'strvar']]
    new_meta['strvar'] = new_meta['strvar'].astype(bytes)
    dataset.add_meta(new_meta)
    assert len(dataset.meta.columns) == num_cols


def test_dataset_add_meta_masked(dataset):
    num_cols = len(dataset.meta.columns)
    new_meta = dataset.meta[['object_id', 'maskedvar']]
    new_meta['maskedvar'].mask = True
    new_meta['maskedvar'].mask[0] = False
    dataset.add_meta(new_meta)
    assert len(dataset.meta.columns) == num_cols


def test_dataset_add_meta_original_masked(dataset):
    num_cols = len(dataset.meta.columns)
    new_meta = dataset.meta[['object_id', 'maskedvar']]
    dataset.meta['maskedvar'] = np.arange(10)
    dataset.add_meta(new_meta)
    assert len(dataset.meta.columns) == num_cols
    assert all(dataset.meta['maskedvar'] == np.arange(10))


def test_dataset_add_meta_new_masked(dataset):
    num_cols = len(dataset.meta.columns)
    new_meta = dataset.meta[['object_id', 'maskedvar']]
    new_meta['maskedvar'] = np.arange(10)
    dataset.add_meta(new_meta)
    assert len(dataset.meta.columns) == num_cols
    assert all(dataset.meta['maskedvar'] == np.arange(10))


def test_dataset_process_light_curve(dataset):
    # Make sure that if we reprocess a light curve from a dataset nothing happens.
    lc = dataset.light_curves[0]
    parsed_lc = lcdata.parse_light_curve(lc)
    assert lc is parsed_lc
    assert lc.meta is parsed_lc.meta


def test_dataset_get_lc_index(dataset):
    lc = dataset.get_lc(3)
    assert lc.meta['object_id'] == 'test_3'


def test_dataset_get_lc_object_id(dataset):
    lc = dataset.get_lc('test_2')
    assert lc.meta['object_id'] == 'test_2'


def test_dataset_get_lc_kwargs(dataset):
    lc = dataset.get_lc(object_id='test_4')
    assert lc.meta['object_id'] == 'test_4'


def test_dataset_get_sncosmo_lc(dataset):
    lc = dataset.get_sncosmo_lc(3)

    assert 'zp' in lc.colnames
    assert 'zpsys' in lc.colnames


def test_from_light_curves_duplicate_object_ids():
    lcs = [make_light_curve('same_object_id') for i in range(2)]
    with pytest.raises(ValueError):
        lcdata.from_light_curves(lcs)


def test_from_light_curves_no_meta():
    lcs = []
    for idx in range(5):
        lc = make_light_curve(f'test_{idx}')
        lc.meta = {}
        lcs.append(lc)

    dataset = lcdata.from_light_curves(lcs)
    assert len(dataset.meta) == 5


def test_dataset_addition(dataset, dataset_2):
    merge_dataset = dataset + dataset_2

    assert len(merge_dataset) == 15
    assert len(merge_dataset.meta) == 15
    assert dataset.light_curves[0].meta['object_id'] in merge_dataset.meta['object_id']
    assert (dataset_2.light_curves[0].meta['object_id'] in
            merge_dataset.meta['object_id'])


def test_dataset_duplicate_object_ids(dataset):
    with pytest.raises(ValueError):
        dataset + dataset


def test_dataset_get_subset(dataset):
    sub_dataset = dataset[2:4]

    assert len(sub_dataset) == 2
    assert (dataset.light_curves[2].meta['object_id']
            == sub_dataset.light_curves[0].meta['object_id'])


def test_dataset_get_single_subset(dataset):
    sub_dataset = dataset[2]

    assert len(sub_dataset) == 1


def test_dataset_hdf5_write(dataset_hdf5):
    assert os.path.exists(dataset_hdf5)


def test_dataset_hdf5_read(dataset_hdf5):
    dataset = lcdata.read_hdf5(dataset_hdf5)
    assert len(dataset) == 10


def test_dataset_hdf5_append(dataset_hdf5, dataset_2):
    dataset_2.write_hdf5(dataset_hdf5, append=True)

    dataset = lcdata.read_hdf5(dataset_hdf5)
    assert len(dataset) == 15


def test_dataset_hdf5_overwrite(dataset_hdf5, dataset_2):
    dataset_2.write_hdf5(dataset_hdf5, overwrite=True)

    dataset = lcdata.read_hdf5(dataset_hdf5)
    assert len(dataset) == 5


def test_dataset_hdf5_no_overwrite(dataset_hdf5, dataset_2):
    with pytest.raises(OSError):
        dataset_2.write_hdf5(dataset_hdf5)


def test_dataset_hdf5_disk_read(dataset_hdf5):
    dataset = lcdata.read_hdf5(dataset_hdf5, in_memory=False)
    assert len(dataset) == 10


def test_dataset_hdf5_disk_load(dataset_hdf5):
    dataset = lcdata.read_hdf5(dataset_hdf5, in_memory=False).load()
    assert len(dataset.light_curves[0]) > 0


def test_dataset_hdf5_disk_slice(dataset_hdf5):
    dataset = lcdata.read_hdf5(dataset_hdf5, in_memory=False)[3:6]
    assert len(dataset) == 3
    assert len(dataset.light_curves[0]) > 0


def test_dataset_hdf5_disk_lc_slice(dataset_hdf5):
    dataset = lcdata.read_hdf5(dataset_hdf5, in_memory=False)
    light_curves = dataset.light_curves[3:6]
    assert len(light_curves) == 3
    assert len(light_curves[0]) > 0


def test_dataset_read_hdf5_disk_chunk(dataset_hdf5):
    dataset = lcdata.read_hdf5(dataset_hdf5, in_memory=False)
    total_count = 0
    for chunk in dataset.iterate_chunks(2):
        assert len(chunk) == 2
        total_count += len(chunk)

    assert total_count == len(dataset)
