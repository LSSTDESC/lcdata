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


aliases = {
    # Photometric observation columns
    # Adapted from sncosmo
    'time': ('time', 'date', 'jd', 'mjd', 'mjdos', 'mjd_obs'),
    'band': ('band', 'bandpass', 'filter', 'flt'),
    'flux': ('flux', 'f'),
    'fluxerr': ('fluxerr', 'flux_error', 'fe', 'fluxerror', 'flux_err'),
    'zp': ('zp', 'zpt', 'zeropoint', 'zero_point'),
    'zpsys': ('zpsys', 'zpmagsys', 'magsys'),

    # Metadata
    'ra': ('ra', 'right_ascension', 'host_ra', 'host_right_ascension', 'hostgal_ra',
           'hostgal_right_ascension'),
    'dec': ('dec', 'decl', 'declination', 'host_dec', 'host_decl', 'host_declination',
            'hostgal_dec', 'hostgal_decl', 'hostgal_declination'),
    'type': ('type', 'label', 'class', 'classification'),
    'redshift': ('redshift', 'z', 'true_z', 'host_z', 'host_specz', 'hostgal_z',
                 'hostgal_specz'),
}


def find_alias(keyword, names, ignore_failure=False):
    """Find an alias for a given keyword

    Inspired by and very similar to `alias_map` in `sncosmo`.

    Parameters
    ----------
    keyword : str
        The keyword to find an alias for
    names : list[str]
        A list of names that are available.

    Returns
    -------
    alias : str
        The alias in names that corresponds to the keyword.
    """
    lowered_names = [i.lower() for i in names]
    for alias in aliases[keyword]:
        if alias in lowered_names:
            return alias

    if ignore_failure:
        return None
    else:
        raise ValueError(f"Couldn't find key {keyword}. Possible aliases {names}.")


warned_default_zp = False
warned_default_zpsys = False


def parse_light_curve(light_curve):
    """Parse a light curve and get it into a standardized format.

    Parameters
    ----------
    light_curve : astropy.Table
        The light curve to parse.

    Returns
    -------
    parsed_light_curve
        The parsed light curve in a standardized format.
    """

    # Check if the light curve is in our standardized format.
    standard_colnames = ['time', 'flux', 'fluxerr', 'band', 'zp', 'zpsys']
    if light_curve.colnames != standard_colnames:
        # Nope, need to move some things around.
        required_keys = ['time', 'flux', 'fluxerr', 'band']
        use_keys = [find_alias(i, light_curve.colnames) for i in required_keys]

        # zp and zpsys are often missing. Default to 25 AB if that is the case which is
        # almost always the format of supernova data.
        zp_key = find_alias('zp', light_curve.colnames, ignore_failure=True)
        zpsys_key = find_alias('zpsys', light_curve.colnames, ignore_failure=True)

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

    # Standardize metadata
    metadata_keys = ['ra', 'dec', 'type', 'redshift']

    for key in metadata_keys:
        if key in light_curve.meta.keys():
            # Already have the key, ignore.
            continue

        # Find the alias and swap it.
        alias = find_alias(key, light_curve.meta.keys(), ignore_failure=True)
        if alias is not None:
            light_curve.meta[key] = light_curve.meta.pop(alias)

    # Add in default metadata and standardize the order of the meta keys.
    new_meta = {
        'ra': np.nan,
        'dec': np.nan,
        'type': 'Unknown',
        'redshift': np.nan,
    }
    new_meta.update(light_curve.meta)
    light_curve.meta = new_meta

    return light_curve


class Dataset:
    """A dataset of light curves."""
    def __init__(self, light_curves):
        # Parse all of the light curves to get them in a standardized format.
        light_curves = [parse_light_curve(i) for i in light_curves]
        self.light_curves = np.array(light_curves, dtype=object)

    @property
    def meta(self):
        return astropy.table.Table([i.meta for i in self.light_curves])

    def __add__(self, other):
        return type(self)(np.hstack([self.light_curves, other.light_curves]))

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
        # Turn the metadata into dictionaries that we can index. We ignore any masked
        # columns.
        meta_dicts = {}
        for lc_meta in meta:
            meta_dict = {}
            for key, value in zip(lc_meta.keys(), lc_meta.values()):
                if not isinstance(value, np.ma.core.MaskedConstant):
                    meta_dict[key] = value
            meta_dicts[lc_meta['object_id']] = meta_dict

        # Load the individual light curves.
        light_curves = []
        lc_data = observations.group_by('object_id')
        for object_id, lc in zip(lc_data.groups.keys['object_id'], lc_data.groups):
            lc.meta = meta_dicts[object_id]
            lc.remove_column('object_id')
            light_curves.append(lc)

        return cls(light_curves)

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

        for i in range(len(light_curves)):
            light_curves[i].meta = dict(meta[i])

        return cls(light_curves)

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
        from astropy.io.misc.hdf5 import write_table_hdf5
        import tables

        # if append:
            # if not os.path.exists(path):
                # Nothing there, so just go ahead.

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
