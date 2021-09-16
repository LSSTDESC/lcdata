import astropy
import astropy.table
import numpy as np
import os
import sys
from collections import abc

from . import schema
from .lightcurve import light_curve_meta_schema, to_sncosmo, parse_light_curve, \
    LightCurveMetadata
from .utils import warn_first_time, get_str_dtype_length, verify_unique

__all__ = ["Dataset", "HDF5LightCurves", "HDF5Dataset", "read_hdf5",
           "from_observations", "from_light_curves", "from_avocado"]


def _parse_observations_table(observations):
    """Parse a table of observations and retrieve the individual light curves.

    Parameters
    ----------
    observations : `~astropy.table.Table`
        Table of observations.

    Returns
    -------
    object_ids : `~astropy.table.Column`[str]
        `object_id` values for each light curve.
    light_curves : list[`~astropy.table.Table`]
        List of light curves.
    """
    light_curves = []
    lc_data = observations.group_by('object_id')
    for lc in lc_data.groups:
        lc.remove_column('object_id')
        light_curves.append(lc)

    return lc_data.groups.keys['object_id'], light_curves


class Dataset:
    """A dataset of light curves.

    Parameters
    ----------
    meta : `~astropy.table.Table`
        Metadata table.
    light_curves : List[`~astropy.table.Table`], optional
        List of light curves where each light curve is represented by an astropy Table.
    """
    def __init__(self, meta, light_curves=None):
        # Parse the metadata to get it in a standardized format.
        unordered_meta = schema.format_table(meta, light_curve_meta_schema)

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
            light_curves = [parse_light_curve(lc, parse_meta=False) for lc in
                            light_curves]

            # Load the light curves into a numpy array. Doing this directly with
            # np.array() calls .as_array() on every Table which is not what we want.
            # Manually loading the array works and is much faster.
            self.light_curves = np.empty(len(light_curves), dtype=object)
            for i in range(len(light_curves)):
                self.light_curves[i] = light_curves[order[i]]

            self._update_lc_meta()

    def _update_lc_meta(self):
        """Set up the meta data for each light curve to point to our metadata table.
        """
        for lc, meta_row in zip(self.light_curves, self.meta):
            lc.meta = LightCurveMetadata(meta_row)

    def __add__(self, other):
        """Combine two datasets.

        This concatenates the two datasets. There cannot be any overlap in the datasets
        or objects with the same `object_id`.

        Parameters
        ----------
        other : `~lcdata.Dataset`
            Dataset to combine this one with

        Returns
        -------
        combined_dataset: `~lcdata.Dataset`
            Combined dataset
        """
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
        if np.issubdtype(type(key), np.integer):
            key = slice(key, key+1)

        meta = self.meta[key]
        light_curves = self.light_curves[key]

        return Dataset(meta, light_curves)

    def get_lc(self, key=None, **kwargs):
        if np.issubdtype(type(key), np.integer):
            return self.light_curves[key]
        elif key is not None:
            mask = self.meta['object_id'] == key
        else:
            mask = np.ones_like(self.meta, dtype=bool)

        for kwargs_key, kwargs_value in kwargs.items():
            mask &= self.meta[kwargs_key] == kwargs_value

        loc = np.argwhere(mask)[0]
        if len(loc) > 1:
            raise KeyError("Multiple light curves match query.")
        elif len(loc) == 0:
            raise KeyError("No light curve matches query.")
        return self.light_curves[loc[0]]

    def get_sncosmo_lc(self, key, **kwargs):
        """Get a light curve in sncosmo format.

        Parameters
        ----------
        idx : int
            Index of the light curve to retrieve.

        Returns
        -------
        `~astropy.table.Table`
            Light curve in sncosmo format.
        """
        lc = self.get_lc(key, **kwargs)
        lc = to_sncosmo(lc)
        return lc

    def add_meta(self, meta, suffix='_2', warn_on_disagreement=True):
        """Add additional metadata into the dataset.

        The other metadata table will be merged using a left join on the `object_id`
        key. It is fine if the other metadata table is in a different order, missing
        light curves, or has additional light curves that aren't in the dataset. For any
        missing entries, the resulting metadata table will be masked.

        Parameters
        ----------
        meta : `~astropy.table.Table`
            Metadata table to merge.
        suffix : str
            Suffix to add to the second column name if there are disagreeing columns.
        warn_on_disagreement : bool
            If True, print a warning if two columns disagree.
        """
        dup_flag = '__lcdata_dup__'

        new_meta = astropy.table.join(
            self.meta,
            meta,
            keys='object_id',
            join_type='left',
            uniq_col_name='{col_name}{table_name}',
            table_names=['', dup_flag],
        )

        # Handle duplicate columns.
        for dup_colname in new_meta.colnames:
            if dup_colname[-len(dup_flag):] != dup_flag:
                continue

            # Found a duplicate
            colname = dup_colname[:-len(dup_flag)]
            col = new_meta[colname]
            dup_col = new_meta[dup_colname]

            col_masked = isinstance(col, astropy.table.MaskedColumn)
            dup_masked = isinstance(dup_col, astropy.table.MaskedColumn)
            check_nan = np.issubdtype(col.dtype, np.floating)
            if not col_masked and not dup_masked:
                # Standard columns. Compare them directly.
                comparison = col == dup_col
                if check_nan:
                    comparison |= (np.isnan(col) & np.isnan(dup_col))
                if np.all(comparison):
                    # Same data, drop the duplicate.
                    del new_meta[dup_colname]
                    continue
            else:
                # At least one masked column. Check if they agree in the non-masked
                # parts.
                common_mask = False
                if col_masked:
                    common_mask |= col.mask
                if dup_masked:
                    common_mask |= dup_col.mask

                col_common = col[~common_mask]
                dup_col_common = dup_col[~common_mask]
                comparison = col_common == dup_col_common
                if check_nan:
                    comparison |= (np.isnan(col_common) & np.isnan(dup_col_common))
                if len(comparison) == 0 or np.all(comparison):
                    # Columns agree in the parts where they are both valid, merge them.
                    if not col_masked:
                        # col is already full, nothing to do.
                        pass
                    elif not dup_masked:
                        # dup_col is full, use it directly.
                        new_meta[colname] = dup_col
                    else:
                        # merge two masked arrays.
                        col[~dup_col.mask] = dup_col[~dup_col.mask]

                    del new_meta[dup_colname]
                    continue

            # Disagreement between the two columns.
            if warn_on_disagreement:
                print(f"WARNING: column {colname} has disagreeing values, renaming "
                      f"second one to {colname}{suffix}.", file=sys.stderr)

            new_meta.rename_column(dup_colname, colname + suffix)

        self.meta = new_meta

        self._update_lc_meta()

    def write_hdf5(self, path, append=False, overwrite=False, object_id_itemsize=0,
                   band_itemsize=0):
        """Write the dataset to an HDF5 file

        Parameters
        ----------
        path : str
            Output path to write to
        append : bool, optional
            Whether to append if there is an existing file, default False
        overwrite : bool, optional
            Whether to overwrite if there is an existing file, default False
        object_id_itemsize : int, optional
            Width to use for the object_id string column. Inferred from the longest
            string if not specified.
        band_itemsize : int, optional
            Width to use for the band string column. Inferred from the longest string if
            not specified.
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
    """Class to interface with light curves in an HDF5 file on disk.

    The light curves are kept on disk, and only loaded into memory when explicitly asked
    for.

    Parameters
    ----------
    dataset : `HDF5Dataset`
        Dataset to handle the light curves for.
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def load(self, start_idx=None, end_idx=None):
        """Load a series of light curves into memory

        Parameters
        ----------
        start_idx : int, optional
            Start of the slice to load, default 0 (the first element)
        end_idx : int, optional
            End of the slice to load, default the last element.

        Returns
        -------
        List[`~astropy.table.Table`]
            List of light curves.
        """
        import tables

        if start_idx is None:
            start_idx = 0
        if end_idx is None or end_idx > len(self):
            end_idx = len(self)

        start_object_id = self._dataset.meta[start_idx]['object_id']
        end_object_id = self._dataset.meta[end_idx - 1]['object_id']

        # Read the light curves.
        with tables.open_file(self._dataset.path, 'r') as f:
            obs_node = f.get_node('/observations')
            observations = astropy.table.Table(obs_node.read_where(
                f"(object_id >= b'{start_object_id}') & "
                f"(object_id <= b'{end_object_id}')"
            ))

        lc_object_ids, unordered_light_curves = _parse_observations_table(observations)

        # Match the light curves to the metadata and preprocess them.
        lc_map = {k: i for i, k in enumerate(lc_object_ids)}
        light_curves = []
        for idx in range(start_idx, end_idx):
            meta_row = self._dataset.meta[idx]
            lc = unordered_light_curves[lc_map[meta_row['object_id']]]
            lc = parse_light_curve(lc, parse_meta=False)
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
        elif np.issubdtype(type(key), np.integer):
            return self.load(key, key+1)[0]
        else:
            raise TypeError('indices must be integers or slices.')

    def __len__(self):
        return len(self._dataset.meta)


class HDF5Dataset(Dataset):
    """Dataset corresponding to an HDF5 file on disk.

    This class only loads the metadata by default. Accessing a light curve in the
    dataset will read it from disk.

    Typically you won't want to use this class directly. Instead, call
    `lcdata.read_hdf5` with ``in_memory`` set to False to read a file.

    Parameters
    ----------
    path : str
        Path to the file.
    meta : `~astropy.table.Table`
        Metadata table.
    """
    def __init__(self, path, meta):
        self.path = path
        super().__init__(meta)

        self.light_curves = HDF5LightCurves(self)

    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            key = slice(key, key+1)

        meta = self.meta[key]

        return HDF5Dataset(self.path, meta)

    def load(self):
        """Load the current dataset into memory."""
        return Dataset(self.meta, self.light_curves.load())

    def _update_lc_meta(self):
        """Light curves aren't loaded in memory, so don't need to update them."""
        pass

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
        `Dataset`
            A `Dataset` object for the given chunk with the light curves loaded.
        """
        return self[chunk_size * chunk_idx:chunk_size * (chunk_idx + 1)].load()

    def iterate_chunks(self, chunk_size):
        """Iterate through the dataset in chunks

        Parameters
        ----------
        chunk_size : int
            Number of light curves per chunk

        Yields
        -------
        `Dataset`
            `Dataset` object for each chunk with the light curves loaded.
        """
        for chunk_idx in range(self.count_chunks(chunk_size)):
            yield self.get_chunk(chunk_idx, chunk_size)


def read_hdf5(path, in_memory=True):
    """Read a dataset from an HDF5 file

    Parameters
    ----------
    path : str
        Path of the dataset
    in_memory : bool
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
    meta : `~astropy.table.Table`
        Table containing the metadata with one row for each light curve.
    observations : `~astropy.table.Table`
        Table containing all of the observations.

    Returns
    -------
    `Dataset`
        A Dataset of light curves built from these tables.
    """
    lc_object_ids, light_curves = _parse_observations_table(observations)

    # Match the metadata to the light curves.
    meta_map = {k: i for i, k in enumerate(meta['object_id'])}
    meta_indices = [meta_map[i] for i in lc_object_ids]
    meta = meta[meta_indices]

    return Dataset(meta, light_curves)


def from_light_curves(light_curves):
    """Load a dataset from a list of light curves.

    Parameters
    ----------
    light_curves : List[`~astropy.table.Table`]
        List of light curves

    Returns
    -------
    `Dataset`
        Dataset containing all of these light curves.
    """
    try:
        # Pull out the metadata. astropy checks that the rows are all dicts, so make
        # sure that they actually are. This is a problem otherwise for our
        # LightCurveMetadata objects.
        dict_meta = [dict(i.meta) for i in light_curves]

        meta = astropy.table.Table(dict_meta)
    except TypeError:
        # Metadata is empty for everything. Create an empty table.
        meta = astropy.table.Table({
            'object_id': astropy.table.MaskedColumn(np.zeros(len(light_curves)),
                                                    mask=np.ones(len(light_curves)))
        })
    return Dataset(meta, light_curves)


def from_avocado(name, **kwargs):
    """Load a dataset from avocado.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    `Dataset`
        the loaded dataset in lcdata format.
    """
    import avocado

    dataset = avocado.load(name, **kwargs)

    # Convert to astropy tables.
    light_curves = []
    for i in dataset.objects:
        lc = astropy.table.Table.from_pandas(i.observations)
        light_curves.append(lc)

    meta = astropy.table.Table.from_pandas(dataset.metadata, index=True)

    return Dataset(meta, light_curves)
