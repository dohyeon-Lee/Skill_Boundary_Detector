from abcdl.backends import open_file, local_path_for


def test_open_local_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    with open_file(str(p)) as f:
        assert f.read() == b"hello"


def test_local_path_for_passthrough(tmp_path):
    p = tmp_path / "y.bin"
    p.write_bytes(b"z")
    assert local_path_for(str(p)) == str(p)


def test_open_http(monkeypatch):
    # fsspec is used for remote; assert open_file routes http(s) through fsspec.open
    import abcdl.backends as backends

    class _Ctx:
        def __enter__(self): return __import__("io").BytesIO(b"web")
        def __exit__(self, *a): return False

    called = {}
    monkeypatch.setattr(backends, "_fsspec_open", lambda uri, mode: (called.update({"uri": uri}) or None, _Ctx())[1])
    with open_file("https://example.com/a.bin") as f:
        assert f.read() == b"web"
    assert called["uri"] == "https://example.com/a.bin"
