import hashlib
import sys
import types

import pytest

from poller import net, transcribe


def test_transcribe_audio_returns_text_and_hash(tmp_path):
    def fake_download(url, dest):
        dest.write_bytes(b"fake audio")

    def fake_transcribe(audio_path, hotwords):
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
            transcribe=lambda p, hw: "unused",
            download=always_fails,
            sleep=lambda _s: None,
        )


def test_transcribe_audio_raises_empty_transcript_error_after_retries():
    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    with pytest.raises(transcribe.EmptyTranscriptError):
        transcribe.transcribe_audio(
            "https://example.org/a.mp3",
            transcribe=lambda p, hw: "   ",
            download=fake_download,
            sleep=lambda _s: None,
        )


def test_transcribe_audio_raises_transcription_error_when_model_keeps_raising():
    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    def always_raises(_path, _hotwords):
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

    def flaky_transcribe(_path, _hotwords):
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


def test_transcribe_audio_forwards_hotwords_to_the_model_call():
    seen = []

    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    def fake_transcribe(audio_path, hotwords):
        seen.append(hotwords)
        return "text"

    transcribe.transcribe_audio(
        "https://example.org/a.mp3",
        "Keith Crosby, Luke",
        transcribe=fake_transcribe,
        download=fake_download,
        sleep=lambda _s: None,
    )
    assert seen == ["Keith Crosby, Luke"]


def test_transcribe_audio_defaults_to_no_hotwords():
    seen = []

    def fake_download(url, dest):
        dest.write_bytes(b"audio")

    def fake_transcribe(audio_path, hotwords):
        seen.append(hotwords)
        return "text"

    transcribe.transcribe_audio(
        "https://example.org/a.mp3", transcribe=fake_transcribe, download=fake_download, sleep=lambda _s: None
    )
    assert seen == [None]


class _FakeSegment:
    def __init__(self, text):
        self.text = text


def _install_fake_whisper(monkeypatch, calls):
    class FakeWhisperModel:
        def __init__(self, model_size_or_path, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            calls.append((audio, kwargs))
            return iter([_FakeSegment(" Hello, church. "), _FakeSegment(" Amen. ")]), None

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(transcribe, "_model", None)


def test_default_transcribe_passes_hotwords_beam_size_condition_flag_and_silence_guard_from_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
    monkeypatch.setenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true")
    monkeypatch.setenv("WHISPER_DOMAIN_PROMPT", "true")
    monkeypatch.setenv("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "1.5")
    calls = []
    _install_fake_whisper(monkeypatch, calls)
    audio = tmp_path / "audio"
    audio.write_bytes(b"audio")

    text = transcribe.default_transcribe(audio, "Keith Crosby, Luke")

    assert text == "Hello, church. Amen."
    assert calls == [
        (
            str(audio),
            {
                "vad_filter": True,
                "log_progress": True,
                "beam_size": 3,
                "condition_on_previous_text": True,
                "hotwords": "Keith Crosby, Luke",
                "word_timestamps": True,
                "hallucination_silence_threshold": 1.5,
            },
        )
    ]


def test_default_transcribe_by_default_sends_no_hotwords_or_previous_text_and_guards_silence(
    monkeypatch, tmp_path
):
    # ADR-0019: hotwords leaked into and dropped speech from transcripts, and
    # previous-window text made short repetition loops self-reinforcing.
    for var in (
        "WHISPER_DOMAIN_PROMPT",
        "WHISPER_BEAM_SIZE",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD",
    ):
        monkeypatch.delenv(var, raising=False)
    calls = []
    _install_fake_whisper(monkeypatch, calls)
    audio = tmp_path / "audio"
    audio.write_bytes(b"audio")

    transcribe.default_transcribe(audio, "Keith Crosby, Luke")

    _audio, kwargs = calls[0]
    assert kwargs["hotwords"] is None
    assert "initial_prompt" not in kwargs
    assert kwargs["beam_size"] == 5
    assert kwargs["condition_on_previous_text"] is False
    # faster-whisper ignores the threshold unless word timestamps are on.
    assert kwargs["word_timestamps"] is True
    assert kwargs["hallucination_silence_threshold"] == 2.0


def test_default_transcribe_omits_the_silence_guard_when_its_threshold_is_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "0")
    calls = []
    _install_fake_whisper(monkeypatch, calls)
    audio = tmp_path / "audio"
    audio.write_bytes(b"audio")

    transcribe.default_transcribe(audio, None)

    _audio, kwargs = calls[0]
    assert "word_timestamps" not in kwargs
    assert "hallucination_silence_threshold" not in kwargs
