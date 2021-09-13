import lcdata


def test_light_curve_schema():
    lcdata.schema.verify_schema(lcdata.lightcurve.light_curve_schema)


def test_light_curve_meta_schema():
    lcdata.schema.verify_schema(lcdata.lightcurve.light_curve_meta_schema)
