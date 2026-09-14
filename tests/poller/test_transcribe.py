import hashlib
import sys
import types

import pytest

from poller import net, transcribe


def test_transcribe_audio_returns_text_and_hash(tmp_path):
    def fake_download(url, dest):
        dest.write_bytes(b"fake audio")

    def fake_transcribe(audio_path):
        assert audio_path.read_bytes() == b"fake audio"
        return "  Hello, this is a sermon.  "

    text, digest = transcribe.transcribe_audio(
        "https://example.org/a.mp3",
        transcribe=fake_transcribe,
        download=fake_download,
        sleep=lambda _s: None,
    )
    assert text == "Hello, this is a sermon."
    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_transcribe_audio_propagates_download_failure(tmp_path):
    def always_fails(url, dest):
        raise net.AudioDownloadError("cdn rejected the request")

    with pytest.raises(net.AudioDownloadError):
        transcribe.transcribe_audio(
            "https://example.org/a.mp3",
            transcribe=lambda p: "unused",
            download=always_fails,
            sleep=lambda _s: None,
        )


def test_transcribe_audio_raises_empty_transcript_error_after_retries():
    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    with pytest.raises(transcribe.EmptyTranscriptError):
        transcribe.transcribe_audio(
            "https://example.org/a.mp3",
            transcribe=lambda p: "   ",
            download=fake_download,
            sleep=lambda _s: None,
        )


def test_transcribe_audio_raises_transcription_error_when_model_keeps_raising():
    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    def always_raises(_path):
        raise RuntimeError("ctranslate2 blew up")

    with pytest.raises(transcribe.TranscriptionError, match="ctranslate2 blew up"):
        transcribe.transcribe_audio(
            "https://example.org/a.mp3",
            transcribe=always_raises,
            download=fake_download,
            sleep=lambda _s: None,
        )


def test_transcribe_audio_succeeds_on_a_later_attempt(tmp_path):
    calls = []

    def flaky_transcribe(_path):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("transient decode error")
        return "recovered transcript"

    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    text, _digest = transcribe.transcribe_audio(
        "https://example.org/a.mp3",
        transcribe=flaky_transcribe,
        download=fake_download,
        sleep=lambda _s: None,
    )
    assert text == "recovered transcript"
    assert len(calls) == 2


def test_get_model_passes_cpu_threads_from_config(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("WHISPER_CPU_THREADS", "3")
    monkeypatch.setattr(transcribe, "_model", None)

    calls = []

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, **kwargs):
            calls.append((model_size_or_path, kwargs))

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    transcribe._get_model()

    assert calls == [("small", {"device": "cpu", "compute_type": "int8", "cpu_threads": 3})]
