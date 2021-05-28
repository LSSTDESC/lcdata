import astropy
import astropy.table
import numpy as np

from .utils import warn_first_time

# TODO: describe the schema format

light_curve_schema = {
    'time': {
        'dtype': np.float64,
        'required': True,
        'aliases': ('time', 'date', 'jd', 'mjd', 'mjdobs'),
    },
    'flux': {
        'dtype': np.float32,
        'required': True,
        'aliases': ('flux', 'f', 'fluxcal'),
    },
    'fluxerr': {
        'dtype': np.float32,
        'required': True,
        'aliases': ('fluxerr', 'fluxerror', 'fe', 'fluxcalerr', 'fluxcalerror'),
    },
    'band': {
        'dtype': bytes,
        'required': True,
        'aliases': ('band', 'bandpass', 'passband', 'filter', 'flt'),
    },
    'zp': {
        'dtype': np.float32,
        'required': True,
        'default': 25.,
        'aliases': ('zp', 'zpt', 'zeropoint'),
    },
    'zpsys': {
        'dtype': bytes,
        'required': True,
        'default': 'ab',
        'aliases': ('zpsys', 'zpmagsys', 'magsys'),
    }
}

metadata_schema = {
    'object_id': {
        'dtype': str,
        'required': True,
        'aliases': ('objectid', 'id'),
    },
    'ra': {
        'dtype': float,
        'required': False,
        'default': np.nan,
        'aliases': ('ra', 'rightascension', 'hostra', 'hostrightascension', 'hostgalra',
                    'hostgalrightascension')
    },
    'dec': {
        'dtype': float,
        'required': False,
        'default': np.nan,
        'aliases': ('dec', 'decl', 'declination', 'hostdec', 'hostdecl',
                    'hostdeclination', 'hostgaldec', 'hostgaldecl',
                    'hostgaldeclination')
    },
    'type': {
        'dtype': str,
        'required': False,
        'default': 'Unknown',
        'aliases': ('type', 'label', 'class', 'classification', 'truetarget',
                    'target'),
    },
    'redshift': {
        'dtype': float,
        'required': False,
        'default': np.nan,
        'aliases': ('redshift', 'z', 'truez', 'hostz', 'hostspecz', 'hostgalz',
                    'hostgalspecz')
    },
}

# TODO: verify schema.
# - combinations of required and default
# - format of aliases


def get_default_value(schema, key):
    schema_info = schema[key]
    if schema_info['required']:
        if 'default' in schema_info:
            # Required, but we have a default value. Warn that we are using it.
            warn_first_time(
                f'default_{key}',
                f"Missing values for required key '{key}', assuming "
                f"'{schema_info['default']}'."
            )
        else:
            # Key is required, but not available.
            raise ValueError(f"Key '{key}' is required.")

    return schema_info['default']


def find_alias(names, aliases):
    """Given a list of names, find the one that matches a list of aliases.

    Inspired by and very similar to `sncosmo.alias_map`.

    Parameters
    ----------
    names : list[str]
        List of names that are available
    aliases : list[str]
        List of aliases to search through. The first one that is available will be
        returned.

    Returns
    -------
    alias : str or None
        Matching alias is one was found, or None otherwise.
    """
    name_map = {i.lower().replace('_', '').replace(' ', ''): i for i in names}
    for alias in aliases:
        if alias in name_map:
            return name_map[alias]

    return None


def format_table(table, schema, verbose=False):
    # First, check if the table is in the right format already.
    if len(table.columns) >= len(schema):
        for table_col, (schema_key, schema_info) in zip(
                table.columns.values(), schema.items()):
            if table_col.name != schema_key:
                # Mismatch on column name
                break
            if not np.issubdtype(table_col.dtype, schema_info['dtype']):
                # Mismatch on dtype
                break
        else:
            # Current table follows the schema, return it as is.
            if verbose:
                print("Table is compliant, returning it as is.")
            return table

    # Current table doesn't follow the schema. Reformat it to get it in our standard
    # format.

    # Get a list of all current columns. We'll pop them out one by one and reformat them
    # to get our new order.
    old_columns = table.columns.copy()

    if verbose:
        print("Formatting table...")

    new_table = astropy.table.Table()

    for schema_key, schema_info in schema.items():
        old_table_key = find_alias(old_columns.keys(), schema_info['aliases'])

        if old_table_key is None:
            # Key not available. Use a default value if possible.
            try:
                default_value = get_default_value(schema, schema_key)
            except ValueError:
                raise ValueError(f"Couldn't find required key '{schema_key}'. "
                                 f"Possible aliases {schema_info['aliases']}.")

            if verbose:
                print(f"- {schema_key}: using default value '{default_value}'")

            new_table.add_column(default_value, name=schema_key)
        else:
            if verbose:
                print(f"- {schema_key}: using column '{old_table_key}'")

            # Alias is available.
            column = old_columns.pop(old_table_key)

            # Cast to the right dtype if necessary.
            if not np.issubdtype(column.dtype, schema_info['dtype']):
                if verbose:
                    print(f"    Converting from dtype '{column.dtype}' to dtype "
                          f"'{schema_info['dtype']}'.")
                column = column.astype(schema_info['dtype'])

            # If it is a masked column, fill with the default value. All columns in the
            # schema should be available for any dataset, even if they just have the
            # default value.
            if isinstance(column, astropy.table.MaskedColumn) and column.mask.sum():
                default_value = get_default_value(schema, schema_key)

                if verbose:
                    print(f"    Filling missing values with default value "
                          f"'{default_value}'.")

                column = column.filled(default_value)

            new_table.add_column(column, name=schema_key)

    # Add in remaining columns.
    for column in old_columns.values():
        new_table.add_column(column)

    # Add metadata
    new_table.meta = table.meta

    return new_table
