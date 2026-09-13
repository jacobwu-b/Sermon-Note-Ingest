import pytest

from poller import net


def test_is_fetchable_enclosure_true_for_http_and_https():
    assert net.is_fetchable_enclosure("http://example.org/a.mp3") is True
    assert net.is_fetchable_enclosure("https://example.org/a.mp3") is True


def test_is_fetchable_enclosure_false_for_other_schemes_and_empty():
    assert net.is_fetchable_enclosure("ftp://example.org/a.mp3") is False
    assert net.is_fetchable_enclosure("") is False
    assert net.is_fetchable_enclosure("file:///etc/passwd") is False


def test_http_download_refuses_non_http_url(tmp_path):
    with pytest.raises(net.AudioDownloadError):
        net.http_download("file:///etc/passwd", tmp_path / "out")


def test_http_download_streams_body_to_dest(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, chunks):
            self._chunks = list(chunks) + [b""]

        def read(self, _size):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        return FakeResponse([b"hello ", b"world"])

    monkeypatch.setattr(net.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out"
    net.http_download("https://example.org/a.mp3", dest)
    assert dest.read_bytes() == b"hello world"


def test_http_download_raises_when_body_exceeds_cap(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self):
            self._served = False

        def read(self, _size):
            if self._served:
                return b""
            self._served = True
            return b"x" * (net._MAX_DOWNLOAD_BYTES + 1)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(net.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    with pytest.raises(net.AudioDownloadError, match="exceeds"):
        net.http_download("https://example.org/a.mp3", tmp_path / "out")


def test_download_audio_retries_then_succeeds(tmp_path):
    attempts = []

    def flaky_download(url, dest):
        attempts.append(url)
        if len(attempts) < 2:
            raise net.AudioDownloadError("transient")
        dest.write_bytes(b"ok")

    net.download_audio(
        "https://example.org/a.mp3",
        tmp_path / "out",
        download=flaky_download,
        sleep=lambda _s: None,
        attempts=3,
    )
    assert (tmp_path / "out").read_bytes() == b"ok"
    assert len(attempts) == 2


def test_download_audio_raises_after_exhausting_attempts(tmp_path):
    def always_fails(url, dest):
        raise net.AudioDownloadError("nope")

    with pytest.raises(net.AudioDownloadError, match="after 2 attempts"):
        net.download_audio(
            "https://example.org/a.mp3",
            tmp_path / "out",
            download=always_fails,
            sleep=lambda _s: None,
            attempts=2,
        )
