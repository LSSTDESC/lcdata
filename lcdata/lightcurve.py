import numpy as np

from . import schema


_session_id = None
_object_id_count = 0


def generate_object_id():
    """Generate a random unique object ID for a light curve.

    We want to make this unique but readable. It is also important that if different
    datasets are generated with different runs of they program they have different IDs.
    To accomplish this, we use the format lcdata_[random session string]_[count]. The
    random_session_string will be consistent for all of the light curves generated in
    the same session. The count will start from zero and increase.
    """
    global _session_id
    global _object_id_count

    if _session_id is None:
        import random
        import string
        _session_id = ''.join(random.choice(string.ascii_lowercase) for i in range(6))

    object_id = f'lcdata_{_session_id}_{_object_id_count:07d}'
    _object_id_count += 1

    return object_id


# Schema describing light curves.
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

# Schema describing light curve metadata.
light_curve_meta_schema = {
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


def parse_light_curve(light_curve, parse_meta=True, verbose=False):
    """Parse a light curve and convert it to lcdata format.

    We currently assume that the light curve has a zeropoint of 25 in the AB magnitude
    system. The zeropoint/zpsys columns that are present in sncosmo-formatted light
    curves are ignored. This should be improved at some point.

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Input light curve in an arbitrary format
    parse_meta : bool, optional
        Whether to parse the metadata, by default True

    Returns
    -------
    `~astropy.table.Table`
        Light curve in lcdata format
    """
    # Format the observations table
    light_curve = schema.format_table(light_curve, light_curve_schema, verbose=verbose)

    # Format the metadata
    if parse_meta:
        light_curve.meta = schema.format_dict(light_curve.meta, light_curve_meta_schema,
                                              verbose=verbose)

    return light_curve


def to_sncosmo(light_curve):
    """Convert an lcdata light curve to sncosmo format.

    This adds the zp and zpsys keys that are required by sncosmo, and converts the band
    name to a string instead of the bytes type used internally.

    Parameters
    ----------
    light_curve : `~astropy.table.Table`
        Light curve in lcdata format

    Returns
    -------
    `~astropy.table.Table`
        Light curve in sncosmo format
    """
    light_curve = light_curve.copy()

    # Convert the band names from bytes to strings
    light_curve['band'] = light_curve['band'].astype(str)

    # Add in zeropoint information.
    light_curve['zp'] = 25.
    light_curve['zpsys'] = 'ab'

    return light_curve
