import astropy
import numpy as np
import os
import sys


def get_str_dtype_length(dtype):
    """Return the length of a string dtype handling unicode properly."""
    if dtype.type == np.unicode_:
        return dtype.itemsize // 4
    else:
        return dtype.itemsize


observation_aliases = {
    # Adapted from sncosmo
    'time': ('time', 'date', 'jd', 'mjd', 'mjdos', 'mjd_obs'),
    'flux': ('flux', 'f'),
    'fluxerr': ('fluxerr', 'flux_error', 'fe', 'fluxerror', 'flux_err'),
    'band': ('band', 'bandpass', 'filter', 'flt'),
    'zp': ('zp', 'zpt', 'zeropoint', 'zero_point'),
    'zpsys': ('zpsys', 'zpmagsys', 'magsys'),
}

metadata_keys = (
    # Common metadata keys to handle. This array has entries of the format:
    # (key: str, type: Type, required: bool, default: Any, aliases: list[str])
    # All of these keys are guaranteed to be in the metadata.
    ('object_id', str, True, None, ('object_id',)),
    ('ra', float, False, np.nan, ('ra', 'right_ascension', 'host_ra',
                                  'host_right_ascension', 'hostgal_ra',
                                  'hostgal_right_ascension')),
    ('dec', float, False, None, ('dec', 'decl', 'declination', 'host_dec', 'host_decl',
                                 'host_declination', 'hostgal_dec', 'hostgal_decl',
                                 'hostgal_declination')),
    ('type', str, False, 'Unknown', ('type', 'label', 'class', 'classification')),
    ('redshift', float, False, np.nan, ('redshift', 'z', 'true_z', 'host_z',
                                        'host_specz', 'hostgal_z', 'hostgal_specz')),
)


def find_alias(keyword, names, aliases, ignore_failure=False):
    """Find an alias for a given keyword

    Inspired by and very similar to `sncosmo.alias_map`.

    Parameters
    ----------
    keyword : str
        Keyword to search for
    names : list[str]
        List of names that are available
    aliases : list[str]
        List of aliases to search through. The first one that is available will be
        returned.
    ignore_failure : bool
        If True, raise a ValueError on failure. If False, return None

    Returns
    -------
    alias : str
        The matching alias.
    """
    lowered_names = [i.lower() for i in names]
    for alias in aliases:
        if alias in lowered_names:
            return alias

    if ignore_failure:
        return None
    else:
        raise ValueError(f"Couldn't find key {keyword}. Possible aliases {aliases}.")


def verify_unique(list_1, list_2, ignore_failure=False):
    """Verify that two lists have no elements in common.

    Parameters
    ----------
    list_1 : list
        First list
    list_2 : list
        Second list to compare
    ignore_failure : bool
        If True, raise a ValueError on failure. If False, return False

    Returns
    -------
    unique : bool
        Returns True if there are no overlapping elements between the two lists. Returns
        False if there are overlapping elements and ignore_failure is set to False.

    Raises
    ------
    ValueError
        If ignore_failure is False, raises a ValueError if there are overlapping
        elements.
    """
    ids_1 = set(list_1)
    ids_2 = set(list_2)
    common_ids = ids_1.intersection(ids_2)

    if common_ids:
        # Found an overlap.
        if ignore_failure:
            return False
        else:
            raise ValueError(
                f"Found overlap of {len(common_ids)} entries including "
                "'{common_ids.pop()}'. Can't handle."
            )
    else:
        return True


warned_default_zp = False
warned_default_zpsys = False


def parse_meta(meta):
    """Parse a metadata table and get it into a standardized format.

    Parameters
    ----------
    meta : astropy.table.Table
        Metadata table to parse.
    """
    # Make a copy so that we don't mess anything up.
    meta = meta.copy()

    # Parse each key and make sure that it is in the right format.
    for key, meta_type, required, default, aliases in metadata_keys:
        alias = find_alias(key, meta.keys(), aliases, ignore_failure=True)

        if alias is None:
            if required:
                raise ValueError(f"Missing required metadata key {key}.")
            else:
                # Key not available, set it to the default value.
                meta[key] = default
        elif alias != key:
            # The key exists, but under the incorrect name. Rename it.
            meta.rename_column(alias, key)

        # Check that we have the right dtype
        if not np.issubdtype(meta[key].dtype, meta_type):
            # Cast to the right type
            meta[key] = meta[key].astype(meta_type)

        # All of the keys in metadata_keys are expected to be available for all light
        # curves, so fill in missing values if we have a masked column.
        if isinstance(meta[key], astropy.table.MaskedColumn):
            meta[key] = meta[key].filled(default)

    # Fix the column order.
    col_order = [i[0] for i in metadata_keys]
    for col in meta.colnames:
        if col not in col_order:
            col_order.append(col)
    meta = meta[col_order]

    return meta


def parse_light_curve(light_curve):
    """Parse a light curve and get it into a standardized format.

    Parameters
    ----------
    light_curve : astropy.table.Table
        The light curve to parse.

    Returns
    -------
    parsed_light_curve
        The parsed light curve in a standardized format.
    """

    # Standardize the observations.
    # Check if the light curve is in our standardized format and skip all of this if it
    # is.
    standard_colnames = ['time', 'flux', 'fluxerr', 'band', 'zp', 'zpsys']
    if light_curve.colnames != standard_colnames:
        # Nope, need to move some things around.
        required_keys = ['time', 'flux', 'fluxerr', 'band']
        use_keys = [find_alias(i, light_curve.colnames, observation_aliases[i]) for i in
                    required_keys]

        # zp and zpsys are often missing. Default to 25 AB if that is the case which is
        # almost always the format of supernova data.
        zp_key = find_alias('zp', light_curve.colnames, observation_aliases['zp'],
                            ignore_failure=True)
        zpsys_key = find_alias('zpsys', light_curve.colnames,
                               observation_aliases['zpsys'], ignore_failure=True)

        if zp_key is not None and zpsys_key is not None:
            # Have all keys
            use_keys.append(zp_key)
            use_keys.append(zpsys_key)
            new_light_curve = light_curve[use_keys]
        else:
            # Missing either zeropoint or magnitude system information
            new_light_curve = light_curve[use_keys]

            if zp_key is None:
                # No ZP available, default to 25.0 global warned_default_zp
                global warned_default_zp
                if not warned_default_zp:
                    print("WARNING: No zeropoint specified, assuming 25.0",
                          file=sys.stderr)
                    warned_default_zp = True
                new_light_curve['zp'] = 25.
            else:
                new_light_curve['zp'] = light_curve[zp_key]

            if zpsys_key is None:
                # No magnitude system available, default to AB
                global warned_default_zpsys
                if not warned_default_zpsys:
                    print("WARNING: No magnitude system specified, assuming AB",
                          file=sys.stderr)
                    warned_default_zpsys = True
                new_light_curve['zpsys'] = 'ab'
            else:
                new_light_curve['zpsys'] = light_curve[zpsys_key]

        light_curve = new_light_curve

        # Rename non-standard columns
        for target, alias in zip(standard_colnames, use_keys):
            if target != alias:
                light_curve.rename_column(alias, target)

    return light_curve


class Dataset:
    """A dataset of light curves."""
    def __init__(self, meta, light_curves):
        # Make sure that the metadata and light curves arrays are the same length.
        if len(meta) != len(light_curves):
            raise ValueError(f"Mismatch between metadata (length {len(meta)}) and "
                             f"light curves (length {len(light_curves)}).")

        # Parse the metadata to get it in a standardized format.
        self.meta = parse_meta(meta)

        # Parse all of the light curves to get them in a standardized format.
        light_curves = [parse_light_curve(i) for i in light_curves]
        self.light_curves = np.array(light_curves, dtype=object)

        # Set up light curve metadata
        # Turn the metadata into dictionaries that we can index. We ignore any masked
        # columns.
        # meta_dicts = {}
        # for lc_meta in meta:
            # meta_dict = {}
            # for key, value in zip(lc_meta.keys(), lc_meta.values()):
                # if not isinstance(value, np.ma.core.MaskedConstant):
                    # meta_dict[key] = value
            # meta_dicts[lc_meta['object_id']] = meta_dict

    def __add__(self, other):
        verify_unique(self.meta['object_id'], other.meta['object_id'])
        combined_meta = astropy.table.vstack([self.meta, other.meta])
        combined_light_curves = np.hstack([self.light_curves, other.light_curves])
        return Dataset(combined_meta, combined_light_curves)

    @classmethod
    def from_tables(cls, meta, observations):
        """Load a dataset from a metadata and an observations table.

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
        # Load the individual light curves.
        light_curves = []
        lc_data = observations.group_by('object_id')
        for object_id, lc in zip(lc_data.groups.keys['object_id'], lc_data.groups):
            lc.remove_column('object_id')
            light_curves.append(lc)

        # TODO: match metadata to observations.

        return cls(meta, light_curves)

    @classmethod
    def from_avocado(cls, name, **kwargs):
        """Load an avocado dataset"""
        import avocado

        dataset = avocado.load(name, **kwargs)

        # Convert to astropy tables.
        light_curves = []
        for i in dataset.objects:
            lc = astropy.table.Table.from_pandas(i.observations)
            light_curves.append(lc)

        meta = astropy.table.Table.from_pandas(dataset.metadata, index=True)

        return cls(meta, light_curves)

    def write_hdf(self, path, append=False, overwrite=False):
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

        if append:
            if not os.path.exists(path):
                # Nothing there, so just go ahead.
                pass
            else:
                # There is a file there. Merge the metadata.
                old_meta = read_table_hdf5(path, '/metadata')
                current_meta = self.meta

                # Check that there is no overlap.
                verify_unique(old_meta['object_id'], current_meta['object_id'])

                meta = astropy.table.vstack([old_meta, self.meta])

                # TODO

        # Consolidate and write out the metadata.
        write_table_hdf5(self.meta, path, '/metadata', overwrite=overwrite,
                         serialize_meta=True)

        # Figure out the dtype of our data.  We need to use fixed length ASCII strings
        # in HDF5. Find the longest strings in each column to not waste unnecessary
        # space.
        object_id_size = 0
        band_size = 0
        zpsys_size = 0

        for lc in self.light_curves:
            object_id_size = max(object_id_size, len(lc.meta['object_id']))
            band_size = max(band_size, get_str_dtype_length(lc['band'].dtype))
            zpsys_size = max(zpsys_size, get_str_dtype_length(lc['band'].dtype))

        dtype = [
            ('object_id', f'S{object_id_size}'),
            ('time', 'f8'),
            ('flux', 'f8'),
            ('fluxerr', 'f8'),
            ('band', f'S{band_size}'),
            ('zp', 'f4'),
            ('zpsys', f'S{zpsys_size}'),
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
            data['zp'][start:end] = lc['zp']
            data['zpsys'][start:end] = lc['zpsys']

            start = end

        # Write out the LC data
        with tables.open_file(path, 'a') as f:
            filters = tables.Filters(complevel=5, complib='blosc', fletcher32=True)
            f.create_table('/', 'observations', data, filters=filters)

    @classmethod
    def read_hdf(cls, path):
        """Read a dataset from an HDF5 file

        Parameters
        ----------
        path : str
            Path of the dataset
        """
        from astropy.io.misc.hdf5 import read_table_hdf5
        import tables

        # Consolidate and write out the metadata
        metadata = read_table_hdf5(path, '/metadata')

        # Read the light curve data
        with tables.open_file(path, 'r') as f:
            obs_node = f.get_node('/observations')
            observations = astropy.table.Table(obs_node.read())

        return cls.from_tables(metadata, observations)
