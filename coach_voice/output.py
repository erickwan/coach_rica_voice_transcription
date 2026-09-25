from pathlib import Path

from .transcribe import Segment


def _ts(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(segments: list[Segment], path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{_ts(seg.start)} --> {_ts(seg.end)}", seg.text, ""]
    path.write_text("\n".join(lines))


def write_txt(segments: list[Segment], path: Path) -> None:
    path.write_text("".join(f"[{_ts(s.start, '.')[3:]}] {s.text}\n" for s in segments))


def write_review(segments: list[Segment], path: Path) -> None:
    """Every recognised segment with its speaker score, for checking and tuning the threshold."""
    rows = ["start\tend\tscore\tlabel\ttext"]
    for s in segments:
        score = "" if s.score is None else f"{s.score:.3f}"
        rows.append(f"{s.start:.2f}\t{s.end:.2f}\t{score}\t{s.label}\t{s.text}")
    path.write_text("\n".join(rows) + "\n")
