"""Speaker embeddings (SpeechBrain ECAPA-TDNN, trained on VoxCeleb) and the saved voice profile."""
import json
from pathlib import Path

import numpy as np
import torch

from .audio import SAMPLE_RATE

MIN_SECONDS = 0.5

_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "coach_voice" / "ecapa"),
            run_opts={"device": "cpu"},
        )
    return _encoder


def embed(clip: np.ndarray) -> np.ndarray | None:
    """L2-normalised speaker embedding, or None if the clip is too short to be reliable."""
    if len(clip) < MIN_SECONDS * SAMPLE_RATE:
        return None
    with torch.no_grad():
        e = encoder().encode_batch(torch.from_numpy(clip).unsqueeze(0)).squeeze().numpy()
    return e / np.linalg.norm(e)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class VoiceProfile:
    """A set of enrolled embeddings for one speaker. Scoring uses the best match against
    the centroid and each individual enrollment, which is more robust than the centroid
    alone when the voice varies (shouting vs. talking)."""

    def __init__(self, name: str, embeddings: list[np.ndarray], threshold: float, sources: list[dict]):
        self.name = name
        self.embeddings = np.stack(embeddings)
        self.threshold = threshold
        self.sources = sources

    @property
    def centroid(self) -> np.ndarray:
        c = self.embeddings.mean(axis=0)
        return c / np.linalg.norm(c)

    def score(self, emb: np.ndarray) -> float:
        sims = [cosine(emb, self.centroid)] + [cosine(emb, e) for e in self.embeddings]
        top = sorted(sims, reverse=True)[:3]
        return float(np.mean(top))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.embeddings)
        path.with_suffix(".json").write_text(json.dumps(
            {"name": self.name, "threshold": self.threshold, "sources": self.sources}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VoiceProfile":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        embs = list(np.load(path.with_suffix(".npy")))
        return cls(meta["name"], embs, meta["threshold"], meta["sources"])

    def extend(self, embeddings: list[np.ndarray], sources: list[dict]) -> None:
        self.embeddings = np.concatenate([self.embeddings, np.stack(embeddings)])
        self.sources += sources
