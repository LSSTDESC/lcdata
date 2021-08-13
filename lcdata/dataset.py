import astropy
import astropy.table
import numpy as np
import os
from collections import abc

from . import schema
from .utils import warn_first_time, get_str_dtype_length, verify_unique

__all__ = ["Dataset", "LightCurveMetadata", "HDF5LightCurves", "HDF5Dataset",
           "read_hdf5", "from_observations", "from_light_curves", "from_avocado",
           "to_sncosmo"]


class LightCurveMetadata(abc.MutableMapping):
    """Class to handle the metadata for a light curve.

    This is a view into the metadata table that behaves like a dict.
    """
    def __init__(self, meta_row):
        self.meta_row = meta_row
        self._cache_dict = None

    def __getitem__(self, key):
        value = self.meta_row[key]
        if isinstance(value, np.ma.core.MaskedConstant):
            raise KeyError
        return value

    def __setitem__(self, key, value):
        self.meta_row[key] = value

    def __delitem__(self, key):
        if key in schema.metadata_schema:
            # Key is in the schema, set it to the default value.
            self.meta_row[key] = schema.get_default_value(schema.metadata_schema, key)
        else:
            # Mask out the entry to make it disappear. First, convert the column to a
            # masked one if it isn't already.
            table = self.meta_row.table
            if not isinstance(table[key], astropy.table.MaskedColumn):
                # Convert to a masked column
                table[key] = astropy.table.MaskedColumn(table[key])

            # Mask out the value
            self.meta_row[key] = np.ma.masked

    def __iter__(self):
        # Return all of the keys for values that aren't masked.
        for key, value in zip(self.meta_row.keys(), self.meta_row.values()):
            if not isinstance(value, np.ma.core.MaskedConstant):
                yield key

    def __len__(self):
        # Return the number of keys with values that aren't masked.
        count = 0
        for value in self.meta_row.values():
            if not isinstance(value, np.ma.core.MaskedConstant):
                count += 1
        return count

    def __str__(self):
        return str(dict(self))

    def __repr__(self):
        return f"{type(self).__name__}({dict(self)})"

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(dict(self), memo)

    def copy(self, use_cache=False, update_cache=False):
        if use_cache and self._cache_dict is not None and not update_cache:
            return self._cache_dict.copy()

        result = dict(self)

        if use_cache or update_cache:
            self._cache_dict = result.copy()

        return result


def parse_observations_table(observations):
    """Parse a table of observations and retrieve the individual light curves.

    Parameters
    ----------
    observations : `astropy.table.Table`
        Table of observations.

    Returns
    -------
    object_ids : astropy.table.column.Column[str]
        `object_id` values for each light curve.
    light_curves : list[astropy.table.Table]
        List of light curves.
    """
    light_curves = []
    lc_data = observations.group_by('object_id')
    for lc in lc_data.groups:
        lc.remove_column('object_id')
        light_curves.append(lc)

    return lc_data.groups.keys['object_id'], light_curves


class Dataset:
    """A dataset of light curves."""
    def __init__(self, meta, light_curves=None):
        # Parse the metadata to get it in a standardized format.
        unordered_meta = schema.format_table(meta, schema.metadata_schema)

        # Reorder the metadata
        order = np.argsort(unordered_meta['object_id'])
        self.meta = unordered_meta[order]

        # Make sure that the object_id keys are unique.
        unique_object_ids, object_id_counts = np.unique(self.meta['object_id'],
                                                        return_counts=True)
        if len(self.meta) != len(unique_object_ids):
            duplicate_object_ids = unique_object_ids[object_id_counts > 1]
            raise ValueError(f"{len(duplicate_object_ids)} duplicate object_ids found "
                             f"({duplicate_object_ids.data}).")

        if light_curves is not None:
            # Make sure that the metadata and light curves arrays are the same length.
            if len(self.meta) != len(light_curves):
                raise ValueError(f"Mismatch between metadata (length {len(self.meta)}) "
                                 f"and light curves (length {len(light_curves)}).")

            # Parse all of the light curves to get them in a standardized format.
            light_curves = [schema.format_table(i, schema.light_curve_schema) for i in
                            light_curves]

            # Load the light curves into a numpy array. Doing this directly with
            # np.array() calls .as_array() on every Table which is not what we want.
            # Manually loading the array works and is much faster.
            self.light_curves = np.empty(len(light_curves), dtype=object)
            for i in range(len(light_curves)):
                self.light_curves[i] = light_curves[i]

            # Sort the light curves
            self.light_curves = self.light_curves[order]

            # Set up the meta data for each light curve to point to our table.
            for lc, meta_row in zip(self.light_curves, self.meta):
                lc.meta = LightCurveMetadata(meta_row)

    def __add__(self, other):
        verify_unique(self.meta['object_id'], other.meta['object_id'])
        combined_meta = astropy.table.vstack([self.meta, other.meta])
        combined_light_curves = np.hstack([self.light_curves, other.light_curves])
        return Dataset(combined_meta, combined_light_curves)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, key):
        """Get a subset of the Dataset

        Parameters
        ----------
        key : Any
            key to use for selection, can be any index to a numpy array.

        Returns
        -------
        Dataset
            subset of the Dataset.
        """
        if isinstance(key, int):
            key = slice(key, key+1)

        meta = self.meta[key]
        light_curves = self.light_curves[key]

        return Dataset(meta, light_curves)

    def get_sncosmo_lc(self, idx):
        lc = self.light_curves[idx]
        lc = to_sncosmo(lc)
        return lc

    def write_hdf5(self, path, append=False, overwrite=False, object_id_itemsize=0,
                   band_itemsize=0):
        """Write the dataset to an HDF5 file

        Parameters
        ----------
        path : str
            Output path to write to
        append : bool, optional
            Whether to append if there is an existing file, by default False
        overwrite : bool, optional
            Whether to overwrite if there is an existing file, by default False
        """
        from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
        import tables

        meta = self.meta

        # Figure out what we are doing.
        if os.path.exists(path):
            if not append and not overwrite:
                raise OSError(f"File exists: {path}")
            elif append:
                # Append to an existing file. We merge the metadata and overwrite what
                # was previously there since there can often be differences in the
                # columns/formats. The observations are in a consistent format, so we
                # can just append them.
                old_meta = read_table_hdf5(path, '/metadata')
                current_meta = self.meta

                # Check that there is no overlap.
                verify_unique(old_meta['object_id'], current_meta['object_id'])

                # Stack the metadata. We rewrite it and overwrite whatever was there
                # before.
                meta = astropy.table.vstack([old_meta, self.meta])

                # Sort the metadata by the object_id.
                meta = meta[np.argsort(meta['object_id'])]

                overwrite = True
            elif overwrite:
                # If both append and overwrite are set, we append.
                os.remove(path)
        else:
            # No file there, so appending is the same as writing to a new file.
            append = False

        # Write out the LC data
        with tables.open_file(path, 'a') as f:
            # Figure out the dtype of our data.  We need to use fixed length ASCII
            # strings in HDF5. Find the longest strings in each column to not waste
            # unnecessary space.
            for lc in self.light_curves:
                object_id_itemsize = max(object_id_itemsize, len(lc.meta['object_id']))
                band_itemsize = max(band_itemsize,
                                    get_str_dtype_length(lc['band'].dtype))

            if append:
                # Make sure that the column sizes used in the file are at least as long
                # as what we want to append.
                obs_node = f.get_node('/observations')

                for key, itemsize in (('object_id', object_id_itemsize),
                                      ('band', band_itemsize)):
                    file_itemsize = obs_node.col(key).itemsize
                    if file_itemsize < itemsize:
                        # TODO: handle resizing the table automatically.
                        raise ValueError(
                            f"File column size too small for key '{key}' "
                            f"(file={file_itemsize}, new={itemsize}). Can't append. "
                            f"Specify a larger value for '{key}_itemsize' when "
                            f"initially creating the file."
                        )

                dtype = obs_node.dtype
            else:
                # TODO: make this format configurable.
                dtype = [
                    ('object_id', f'S{object_id_itemsize}'),
                    ('time', 'f8'),
                    ('flux', 'f4'),
                    ('fluxerr', 'f4'),
                    ('band', f'S{band_itemsize}'),
                ]

            # Setup an empty record array
            length = np.sum([len(i) for i in self.light_curves])
            data = np.recarray((length,), dtype=dtype)

            start = 0

            for lc in self.light_curves:
                end = start + len(lc)

                data['object_id'][start:end] = lc.meta['object_id']
                data['time'][start:end] = lc['time']
                data['flux'][start:end] = lc['flux']
                data['fluxerr'][start:end] = lc['fluxerr']
                data['band'][start:end] = lc['band']

                start = end

            # Write out the observations.
            if append:
                f.get_node('/observations').append(data)
            else:
                filters = tables.Filters(complevel=5, complib='blosc', fletcher32=True)
                table = f.create_table('/', 'observations', data, filters=filters)
                table.cols.object_id.create_index()

        # Write out the metadata
        write_table_hdf5(meta, path, '/metadata', overwrite=True, append=True,
                         serialize_meta=True)


class HDF5LightCurves(abc.Sequence):
    def __init__(self, path, meta):
        self.path = path
        self.meta = meta

    def load(self, start_idx=None, end_idx=None):
        import tables

        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx > len(self):
            end_idx = len(self)

        start_object_id = self.meta[start_idx]['object_id']
        end_object_id = self.meta[end_idx - 1]['object_id']

        # Read the light curves.
        with tables.open_file(self.path, 'r') as f:
            obs_node = f.get_node('/observations')
            observations = astropy.table.Table(obs_node.read_where(
                f"(object_id >= b'{start_object_id}') & "
                f"(object_id <= b'{end_object_id}')"
            ))

        lc_object_ids, unordered_light_curves = parse_observations_table(observations)

        # Match the light curves to the metadata and preprocess them.
        lc_map = {k: i for i, k in enumerate(lc_object_ids)}
        light_curves = []
        for idx in range(start_idx, end_idx):
            meta_row = self.meta[idx]
            lc = unordered_light_curves[lc_map[meta_row['object_id']]]
            lc = schema.format_table(lc, schema.light_curve_schema)
            lc.meta = LightCurveMetadata(meta_row)
            light_curves.append(lc)

        return light_curves

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                warn_first_time('slicing_hdf5',
                                'Stepped slicing loads all data. Avoid it.')
            lcs = self.load(key.start, key.stop)
            if key.step is not None:
                lcs = lcs[::key.step]
            return lcs
        elif isinstance(key, int):
            return self.load(key, key+1)[0]
        else:
            raise TypeError('indices must be integers or slices.')

    def __len__(self):
        return len(self.meta)


class HDF5Dataset(Dataset):
    """A dataset corresponding to an HDF5 file on disk.

    This class only loads the metadata by default. Accessing a light curve in the
    dataset will read it from disk.
    """
    def __init__(self, path, meta):
        self.path = path
        super().__init__(meta)

        self.light_curves = HDF5LightCurves(self.path, self.meta)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key+1)

        meta = self.meta[key]

        return HDF5Dataset(self.path, meta)

    def load(self):
        """Load the current dataset into memory."""
        return Dataset(self.meta, self.light_curves.load())

    def count_chunks(self, chunk_size):
        """Count the number of chunks that are in the dataset for a given chunk_size"""
        return (len(self) - 1) // chunk_size + 1

    def get_chunk(self, chunk_idx, chunk_size):
        """Get a chunk from the dataset

        Parameters
        ----------
        chunk_idx : int
            Index of the chunk to retrieve.
        chunk_size : int
            Number of light curves per chunk.

        Returns
        -------
        dataset : `Dataset`
            A `Dataset` object for the given chunk with the light curves loaded.
        """
        return self[chunk_size * chunk_idx:chunk_size * (chunk_idx + 1)].load()

    def iterate_chunks(self, chunk_size):
        for chunk_idx in range(self.count_chunks(chunk_size)):
            yield self.get_chunk(chunk_idx, chunk_size)


def read_hdf5(path, in_memory=True):
    """Read a dataset from an HDF5 file

    Parameters
    ----------
    path : str
        Path of the dataset
    """
    from astropy.io.misc.hdf5 import read_table_hdf5
    import tables

    # Read the metadata
    meta = read_table_hdf5(path, '/metadata')

    if in_memory:
        # Read all of the light curve data
        with tables.open_file(path, 'r') as f:
            obs_node = f.get_node('/observations')
            observations = astropy.table.Table(obs_node.read())

        return from_observations(meta, observations)
    else:
        # Don't read all of the light curve data. It can be loaded later if needed.
        return HDF5Dataset(path, meta)


def from_observations(meta, observations):
    """Load a dataset from a table of all of the observations.

    Parameters
    ----------
    meta : `astropy.table.Table`
        Table containing the metadata with one row for each light curve.
    observations : `astropy.table.Table`
        Table containing all of the observations.

    Returns
    -------
    `Dataset`
        A Dataset of light curves built from these tables.
    """
    lc_object_ids, light_curves = parse_observations_table(observations)

    # Match the metadata to the light curves.
    meta_map = {k: i for i, k in enumerate(meta['object_id'])}
    meta_indices = [meta_map[i] for i in lc_object_ids]
    meta = meta[meta_indices]

    return Dataset(meta, light_curves)


def from_light_curves(light_curves):
    try:
        meta = astropy.table.Table([i.meta for i in light_curves])
    except TypeError:
        # Metadata is empty for everything. Create an empty table.
        meta = astropy.table.Table({
            'object_id': astropy.table.MaskedColumn(np.zeros(len(light_curves)),
                                                    mask=np.ones(len(light_curves)))
        })
    return Dataset(meta, light_curves)


def from_avocado(name, **kwargs):
    """Load an avocado dataset"""
    import avocado

    dataset = avocado.load(name, **kwargs)

    # Convert to astropy tables.
    light_curves = []
    for i in dataset.objects:
        lc = astropy.table.Table.from_pandas(i.observations)
        light_curves.append(lc)

    meta = astropy.table.Table.from_pandas(dataset.metadata, index=True)

    return Dataset(meta, light_curves)


def to_sncosmo(light_curve):
    light_curve = light_curve.copy()

    # Convert the band names from bytes to strings
    light_curve['band'] = light_curve['band'].astype(str)

    # Add in zeropoint information.
    light_curve['zp'] = 25.
    light_curve['zpsys'] = 'ab'

    return light_curve
