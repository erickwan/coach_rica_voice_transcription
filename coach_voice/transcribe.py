"""Speech recognition (faster-whisper) plus per-segment speaker matching against a VoiceProfile."""
from dataclasses import dataclass

import numpy as np

from .audio import slice_audio
from .speaker import VoiceProfile, embed

# Segments shorter than this are padded (centred) before embedding: very short
# clips give unreliable speaker embeddings.
MIN_EMBED_WINDOW = 1.5
# Scores in [threshold - UNCERTAIN_MARGIN, threshold) are reported as "uncertain".
UNCERTAIN_MARGIN = 0.06
# Whisper segments can span several speakers; re-split on pauses between words so each
# piece is (usually) one voice.
SPLIT_GAP = 0.5
MAX_SEGMENT = 8.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    score: float | None = None
    label: str = ""


def recognize(audio: np.ndarray, model_name: str, names: list[str], language: str = "en") -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, _ = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        word_timestamps=True,
        # Carrying context across segments makes Whisper loop on repetitive shouting.
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        # Whisper mishears uncommon names on noisy field audio (e.g. "Vikram" -> "Big Grub").
        hotwords=" ".join(names) if names else None,
    )
    return split_on_pauses([w for s in segments for w in s.words])


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


def score_segments(audio: np.ndarray, segments: list[Segment], profile: VoiceProfile) -> None:
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        half = max((seg.end - seg.start) / 2 + 0.15, MIN_EMBED_WINDOW / 2)
        emb = embed(slice_audio(audio, max(0.0, mid - half), mid + half))
        seg.score = None if emb is None else profile.score(emb)
        if seg.score is None:
            seg.label = "uncertain"
        elif seg.score >= profile.threshold:
            seg.label = "match"
        elif seg.score >= profile.threshold - UNCERTAIN_MARGIN:
            seg.label = "uncertain"
        else:
            seg.label = "other"
