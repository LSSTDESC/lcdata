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
    }

    return lc


def make_dataset(start_idx=0, end_idx=10):
    lcs = [make_light_curve(f'test_{i}') for i in range(start_idx, end_idx)]
    dataset = lcdata.from_light_curves(lcs)

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


def test_dataset_get_sncosmo_lc(dataset):
    lc = dataset.get_sncosmo_lc(3)

    assert 'zp' in lc.colnames
    assert 'zpsys' in lc.colnames


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
