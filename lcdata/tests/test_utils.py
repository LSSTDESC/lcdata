import lcdata


def test_generate_object_id():
    object_id_1 = lcdata.utils.generate_object_id()
    object_id_2 = lcdata.utils.generate_object_id()

    assert object_id_1 != object_id_2


def test_warn_first_time(capsys):
    lcdata.utils.warn_first_time('key1', 'Warning 1')
    captured = capsys.readouterr()
    assert 'Warning 1' in captured.err
    lcdata.utils.warn_first_time('key1', 'Warning 2')
    captured = capsys.readouterr()
    assert captured.err == ''
    lcdata.utils.warn_first_time('key2', 'Warning 3')
    captured = capsys.readouterr()
    assert 'Warning 3' in captured.err
