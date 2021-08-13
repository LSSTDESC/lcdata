import astropy
import astropy.table
import numpy as np

from .utils import generate_object_id

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
}

metadata_schema = {
    'object_id': {
        'dtype': str,
        'default_function': generate_object_id,
        'aliases': ('objectid', 'id'),
    },
    'ra': {
        'dtype': float,
        'default': np.nan,
        'aliases': ('ra', 'rightascension', 'hostra', 'hostrightascension', 'hostgalra',
                    'hostgalrightascension')
    },
    'dec': {
        'dtype': float,
        'default': np.nan,
        'aliases': ('dec', 'decl', 'declination', 'hostdec', 'hostdecl',
                    'hostdeclination', 'hostgaldec', 'hostgaldecl',
                    'hostgaldeclination')
    },
    'type': {
        'dtype': str,
        'default': 'Unknown',
        'aliases': ('type', 'label', 'class', 'classification', 'truetarget',
                    'target'),
    },
    'redshift': {
        'dtype': float,
        'default': np.nan,
        'aliases': ('redshift', 'z', 'truez', 'hostz', 'hostspecz', 'hostgalz',
                    'hostgalspecz')
    },
}

# TODO: verify schema.
# - combinations of required and default
# - format of aliases


def get_default_value(schema, key, count=None):
    schema_info = schema[key]
    if schema_info.get('required', False):
        # Key is required, but not available.
        raise ValueError(f"Key '{key}' is required.")
    if 'default' in schema_info:
        return schema_info['default']
    if 'default_function' in schema_info:
        if count is None:
            return schema_info['default_function']()
        else:
            return [schema_info['default_function']() for i in range(count)]
    else:
        raise ValueError(f"Invalid schema: key '{key}' not required and no "
                         "default value!")


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
                default_value = get_default_value(schema, schema_key, len(table))
            except ValueError:
                raise ValueError(f"Couldn't find required key '{schema_key}'. "
                                 f"Possible aliases {schema_info['aliases']}.")

            if verbose:
                print(f"- {schema_key}: using default value '{default_value}'")

            if len(new_table) == 0 and not isinstance(default_value, list):
                new_table.add_column([default_value] * len(table), name=schema_key)
            else:
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
                count = column.mask.sum()
                default_value = get_default_value(schema, schema_key, count)

                if verbose:
                    print(f"    Filling missing values with default value "
                          f"'{default_value}'.")

                # If we are working with a string column, convert it to an object dtype
                # while we're working to avoid truncation issues.
                if column.dtype.type in (np.unicode_, np.bytes_):
                    old_type = column.dtype.type
                    column = column.astype(object)
                else:
                    old_type = None

                if isinstance(default_value, list):
                    column[column.mask] = default_value
                    column = column.filled()
                else:
                    column = column.filled(default_value)

                if old_type is not None:
                    column = column.astype(old_type)

            new_table.add_column(column, name=schema_key)

    # Add in remaining columns.
    for column in old_columns.values():
        new_table.add_column(column)

    # Add metadata
    new_table.meta = table.meta

    return new_table
