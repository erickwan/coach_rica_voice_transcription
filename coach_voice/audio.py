import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio/video file to mono 16 kHz float32."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)],
            check=True,
        )
        audio, _ = sf.read(wav, dtype="float32")
    return audio


def slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    return audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
