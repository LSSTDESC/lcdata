from astropy.table import Table
import pytest

import lcdata


@pytest.fixture
def light_curve():
    light_curve = Table({
        'mjd': [1., 2., 3.],
        'flux': [1., 2., 1.],
        'fluxerr': [0.5, 1., 1.],
        'band': ['lsstu', 'lsstb', 'lsstr'],
    })

    light_curve.meta = {
        'z': 0.1,
        'ra': 10.,
        'dec': 20.,
    }

    return light_curve


def test_light_curve_schema():
    lcdata.schema.verify_schema(lcdata.lightcurve.light_curve_schema)


def test_light_curve_meta_schema():
    lcdata.schema.verify_schema(lcdata.lightcurve.light_curve_meta_schema)


def test_generate_object_id():
    object_id_1 = lcdata.lightcurve.generate_object_id()
    object_id_2 = lcdata.lightcurve.generate_object_id()

    assert object_id_1 != object_id_2


def test_parse_light_curve(light_curve):
    parsed_light_curve = lcdata.parse_light_curve(light_curve)
    assert 'time' in parsed_light_curve.columns
    assert 'redshift' in parsed_light_curve.meta.keys()


def test_to_sncosmo(light_curve):
    parsed_light_curve = lcdata.parse_light_curve(light_curve)
    sncosmo_light_curve = lcdata.to_sncosmo(parsed_light_curve)

    assert 'zp' in sncosmo_light_curve.colnames
    assert 'zpsys' in sncosmo_light_curve.colnames
