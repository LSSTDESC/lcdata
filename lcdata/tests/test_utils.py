import lcdata


def test_generate_object_id():
    object_id_1 = lcdata.utils.generate_object_id()
    object_id_2 = lcdata.utils.generate_object_id()

    assert object_id_1 != object_id_2
