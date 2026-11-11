from types import SimpleNamespace

from src.config import Config
from src.media import transcription


def test_faster_whisper_is_a_valid_provider(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_PROVIDER", "faster-whisper")
    assert Config().transcription_provider == "faster_whisper"


def test_faster_whisper_result_matches_whisper_shape(monkeypatch, tmp_path):
    words = [
        SimpleNamespace(word=" Hello", start=0.0, end=0.4, probability=0.9),
        SimpleNamespace(word=" world.", start=0.5, end=1.0, probability=0.8),
    ]
    segment = SimpleNamespace(start=0.0, end=1.0, text=" Hello world.", words=words)
    calls = {}

    class FakeModel:
        def transcribe(self, path, word_timestamps, language):
            calls.update(path=path, word_timestamps=word_timestamps, language=language)
            return iter([segment]), SimpleNamespace(language="en", duration=1.0)

    video = tmp_path / "video.mp4"
    audio = tmp_path / "video.transcription.mp3"
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda _: audio)
    monkeypatch.setattr(transcription, "_get_faster_whisper_model", lambda _: FakeModel())

    result = transcription.transcribe_with_faster_whisper(video, "small", "en_us")

    assert calls == {"path": str(audio), "word_timestamps": True, "language": "en"}
    assert result["text"] == "Hello world."
    assert result["language"] == "en"
    assert result["segments"][0]["words"][1] == {
        "word": " world.", "start": 0.5, "end": 1.0, "probability": 0.8,
    }

    lines = transcription.format_transcript_for_analysis(result)
    assert lines == ["[00:00 - 00:01] Hello world."]
    cached = transcription._whisper_result_to_transcript_data(result)
    assert [w["start"] for w in cached["words"]] == [0, 500]
    assert cached["language_code"] == "en"
