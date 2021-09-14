import astropy.table
import numpy as np

"""This module describes and verifies schemas for astropy Tables.

A schema is a dictionary of keys, each of which represents one column in the table.
Each key in the schema much have the following attributes specified:
- dtype: the numpy dtype to use.
- aliases: a list of aliases for the key when formatting a new table. This should
    include the name of the key itself. The aliases should all be lowercase with no
    whitespace or underscores.
- One of the following entries representing the key type:
    - required: set to True if the key is required.
    - default: the default value for the column if it is missing.
    - default_function: A function to call to get the default value. The function can
        return different things every time that it is called.
"""


def verify_schema(schema):
    """Verify a schema

    Parameters
    ----------
    schema : dict[dict]
        Schema to verify. See `schema.py` for details.

    Raises
    ------
    ValueError
        For any noncomplient schemas. The error message will describe what part of the
        schema is invalid.
    """
    for key, key_schema in schema.items():
        # Make sure that the dtype is specified.
        if 'dtype' not in key_schema:
            raise ValueError(f"Invalid schema: key '{key}' missing dtype.")

        # Make sure that the aliases are in the correct format.
        if 'aliases' not in key_schema:
            raise ValueError(f"Invalid schema: key '{key}' missing aliases.")

        aliases = key_schema['aliases']

        if not find_alias([key], aliases):
            raise ValueError(f"Invalid schema: key '{key}' doesn't match aliases "
                             f"{aliases}")

        for alias in aliases:
            parse_alias = alias.lower().replace('_', '').replace(' ', '')
            if parse_alias != alias:
                raise ValueError(f"Invalid schema: alias '{alias}' for key '{key}' "
                                 f"should be '{parse_alias}'")

        # Make sure that the key type is specified.
        if ('required' not in key_schema
                and 'default' not in key_schema
                and 'default_function' not in key_schema):
            raise ValueError("Invalid schema: must specify one of [required, default, "
                             f"default_function for key '{key}'")

        # Make sure that they are no extra keys.
        if len(key_schema) != 3:
            raise ValueError(f"Invalid schema: extra entries found for key '{key}'.")


def get_default_value(schema, key, count=None):
    """Get the default value for a key in a schema.

    Parameters
    ----------
    schema : dict[dict]
        Schema to compare to
    key : str
        Key to look for in the schema.
    count : int, optional
        For default functions that return different values, the number of values to
        return. By default, only a single value is returned.

    Returns
    -------
    default_value
        The default value parsed from the schema.

    Raises
    ------
    ValueError
        If a default value does not exist for this key in the schema.
    """
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


def format_dict(dictionary, schema, verbose=False):
    """Format a dictionary with a given schema.

    Parameters
    ----------
    dictionary : `dict`
        Dictionary to format
    schema : dict[dict]
        Schema to use for formatting. See schema.py for details.
    verbose : bool, optional
        Whether to print debugging messages, by default False

    Returns
    -------
    `dict`
        Formatted dictionary

    Raises
    ------
    ValueError
        If there are required keys in the schema that are missing in the dictionary.
    """
    # First, check if the dictionary is in the right format already.
    if len(dictionary.keys()) >= len(schema):
        for (dict_key, dict_value), (schema_key, schema_info) in zip(
                dictionary.items(), schema.items()):
            if dict_key != schema_key:
                # Mismatch on column name
                break
            if not isinstance(dict_value, schema_info['dtype']):
                # Mismatch on dtype
                break
        else:
            # Current table follows the schema, return it as is.
            if verbose:
                print("Dictionary is compliant, returning it as is.")
            return dictionary

    # Current dictionary doesn't follow the schema. Reformat it to get it in our
    # standard format. Make a copy, and we'll pop each entry out one-by-one as we
    # convert them.
    dictionary = dictionary.copy()

    if verbose:
        print("Formatting dictionary...")

    new_dict = dict()

    for schema_key, schema_info in schema.items():
        old_key = find_alias(dictionary.keys(), schema_info['aliases'])

        if old_key is None:
            # Key not available. Use a default value if possible.
            try:
                default_value = get_default_value(schema, schema_key)
            except ValueError:
                raise ValueError(f"Couldn't find required key '{schema_key}'. "
                                 f"Possible aliases {schema_info['aliases']}.")

            if verbose:
                print(f"- {schema_key}: using default value '{default_value}'")
            new_dict[schema_key] = default_value
        else:
            # Alias is available.
            if verbose:
                print(f"- {schema_key}: using column '{old_key}'")

            value = dictionary.pop(old_key)

            # Cast to the right dtype if necessary.
            if not isinstance(value, schema_info['dtype']):
                if verbose:
                    print(f"    Converting from dtype '{type(value)}' to dtype "
                          f"'{schema_info['dtype']}'.")
                value = schema_info['dtype'](value)

            new_dict[schema_key] = value

    # Add in remaining keys
    new_dict.update(dictionary)

    return new_dict


def format_table(table, schema, verbose=False):
    """Format a table with a given schema.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table to format
    schema : dict[dict]
        Schema to use for formatting
    verbose : bool, optional
        Whether to print debugging messages, by default False

    Returns
    -------
    `~astropy.table.Table`
        Formatted table

    Raises
    ------
    ValueError
        If there are required keys in the schema that are missing in the table.
    """
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
