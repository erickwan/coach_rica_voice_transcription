"""Speech recognition (faster-whisper) plus per-segment speaker matching against a VoiceProfile."""
import re
from dataclasses import dataclass

import numpy as np

from .audio import slice_audio
from .speaker import VoiceProfile, embed

# Segments shorter than this are padded (centred) before embedding: very short
# clips give unreliable speaker embeddings.
MIN_EMBED_WINDOW = 1.5
# Whisper segments can span several speakers; re-split on pauses between words so each
# piece is (usually) one voice.
SPLIT_GAP = 0.5
MAX_SEGMENT = 8.0
CHUNK_SECONDS = 10


@dataclass
class Segment:
    start: float
    end: float
    text: str
    score: float | None = None
    label: str = ""
    negative_score: float | None = None


_models: dict = {}
# Whisper's stock phrases for noise (learned from video outros); never real coaching.
HALLUCINATIONS = re.compile(r"^(thank you|thanks)( so much)?( for watching)?[.! ]*$|for watching", re.I)


def whisper_model(name: str):
    if name not in _models:
        from faster_whisper import WhisperModel

        _models[name] = WhisperModel(name, device="auto", compute_type="int8")
    return _models[name]


def recognize(audio: np.ndarray, model_name: str, names: list[str], language: str = "en",
              on_progress=None) -> list[Segment]:
    """Transcribe and split into single-voice-ish segments. on_progress(fraction) is
    called as recognition advances through the audio."""
    from faster_whisper import BatchedInferencePipeline

    # Speech is found with VAD and each short chunk is decoded independently. Whisper's
    # normal sequential decoding over 30 s windows can abandon the rest of a window on
    # noisy field audio, silently dropping whole passages of speech.
    segments, info = BatchedInferencePipeline(whisper_model(model_name)).transcribe(
        audio,
        language=language,
        batch_size=8,
        chunk_length=CHUNK_SECONDS,
        word_timestamps=True,
        condition_on_previous_text=False,
        # Whisper mishears uncommon names on noisy field audio (e.g. "Vikram" -> "Big Grub").
        hotwords=" ".join(names) if names else None,
    )
    words = []
    for s in segments:
        words += s.words
        if on_progress and info.duration:
            on_progress(min(1.0, s.end / info.duration))
    return [s for s in split_on_pauses(words) if s.text and not HALLUCINATIONS.search(s.text)]


def split_on_pauses(words) -> list[Segment]:
    out: list[Segment] = []
    current: list = []
    for w in words:
        if current and (w.start - current[-1].end > SPLIT_GAP or w.end - current[0].start > MAX_SEGMENT):
            out.append(_join(current))
            current = []
        current.append(w)
    if current:
        out.append(_join(current))
    return out


def _join(words) -> Segment:
    return Segment(words[0].start, words[-1].end, "".join(w.word for w in words).strip())


def embed_segment(audio: np.ndarray, seg: Segment) -> np.ndarray | None:
    mid = (seg.start + seg.end) / 2
    half = max((seg.end - seg.start) / 2 + 0.15, MIN_EMBED_WINDOW / 2)
    return embed(slice_audio(audio, max(0.0, mid - half), mid + half))


def score_segments(audio: np.ndarray, segments: list[Segment], profile: VoiceProfile) -> None:
    for seg in segments:
        seg.score, seg.negative_score, seg.label = profile.classify(embed_segment(audio, seg))
