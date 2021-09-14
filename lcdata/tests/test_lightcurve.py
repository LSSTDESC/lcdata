import lcdata


def test_light_curve_schema():
    lcdata.schema.verify_schema(lcdata.lightcurve.light_curve_schema)


def test_light_curve_meta_schema():
    lcdata.schema.verify_schema(lcdata.lightcurve.light_curve_meta_schema)


def test_generate_object_id():
    object_id_1 = lcdata.lightcurve.generate_object_id()
    object_id_2 = lcdata.lightcurve.generate_object_id()

    assert object_id_1 != object_id_2
