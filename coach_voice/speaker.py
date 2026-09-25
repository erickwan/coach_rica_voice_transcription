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


DEFAULT_THRESHOLD = 0.18
# Scores in [threshold - UNCERTAIN_MARGIN, threshold) are reported as "uncertain".
UNCERTAIN_MARGIN = 0.06
# How much closer a segment must be to a "not the speaker" sample than to the nearest
# speaker samples before it is rejected regardless of its score.
NEGATIVE_MARGIN = 0.03


def _top_mean(sims: list[float], k: int = 3) -> float:
    return float(np.mean(sorted(sims, reverse=True)[:k]))


class VoiceProfile:
    """Enrolled embeddings for one speaker (positives), plus optional embeddings of other
    voices that were confused with them (negatives).

    Scoring uses the best matches against the centroid and each individual enrollment,
    which is more robust than the centroid alone when the voice varies (shouting vs.
    talking). A segment that sounds more like a known negative than like the speaker is
    rejected even above the threshold.
    """

    def __init__(self, name: str, embeddings: list[np.ndarray], threshold: float, sources: list[dict],
                 negatives: list[np.ndarray] | None = None, negative_sources: list[dict] | None = None,
                 auto_threshold: bool = False):
        self.name = name
        self.embeddings = list(embeddings)
        self.threshold = threshold
        self.sources = list(sources)
        self.negatives = list(negatives or [])
        self.negative_sources = list(negative_sources or [])
        self.auto_threshold = auto_threshold

    @property
    def centroid(self) -> np.ndarray:
        c = np.mean(self.embeddings, axis=0)
        return c / np.linalg.norm(c)

    def score(self, emb: np.ndarray, exclude: int | None = None) -> float | None:
        embs = [e for i, e in enumerate(self.embeddings) if i != exclude]
        if not embs:
            return None
        c = np.mean(embs, axis=0)
        sims = [cosine(emb, c)] + [cosine(emb, e) for e in embs]
        return _top_mean(sims)

    def nearest_score(self, emb: np.ndarray, exclude: int | None = None) -> float:
        """Similarity to the closest speaker samples, comparable with negative_score."""
        return _top_mean([cosine(emb, e) for i, e in enumerate(self.embeddings) if i != exclude], k=2)

    def negative_score(self, emb: np.ndarray, exclude: int | None = None) -> float | None:
        negs = [e for i, e in enumerate(self.negatives) if i != exclude]
        if not negs:
            return None
        return _top_mean([cosine(emb, e) for e in negs], k=2)

    def classify(self, emb: np.ndarray | None, exclude: int | None = None,
                 exclude_negative: int | None = None) -> tuple[float | None, float | None, str]:
        """Return (score, negative_score, label) where label is match / uncertain / other.
        exclude / exclude_negative leave out one enrolled embedding (for held-out scoring)."""
        if emb is None:
            return None, None, "uncertain"
        score = self.score(emb, exclude=exclude)
        neg = self.negative_score(emb, exclude=exclude_negative)
        if score is None:
            return None, neg, "uncertain"
        if neg is not None and neg > self.nearest_score(emb, exclude) + NEGATIVE_MARGIN:
            return score, neg, "other"
        if score >= self.threshold:
            return score, neg, "match"
        if score >= self.threshold - UNCERTAIN_MARGIN:
            return score, neg, "uncertain"
        return score, neg, "other"

    def calibrate(self) -> dict:
        """Estimate how well the profile separates the speaker from the labelled negatives,
        using leave-one-out scores, and (if auto_threshold) pick the threshold that best
        separates them."""
        pos = [self.score(e, exclude=i) for i, e in enumerate(self.embeddings)]
        pos = [p for p in pos if p is not None]
        neg = [self.score(e) for e in self.negatives] if self.embeddings else []
        stats = {"positive_scores": pos, "negative_scores": neg, "accuracy": None}
        if len(pos) >= 2 and neg:
            if self.auto_threshold:
                self.threshold = _best_threshold(pos, neg)
            tp = sum(p >= self.threshold for p in pos) / len(pos)
            tn = sum(n < self.threshold for n in neg) / len(neg)
            stats["accuracy"] = (tp + tn) / 2
        return stats

    def remove_source(self, key: dict) -> None:
        """Drop any positive or negative whose source matches all fields in key."""
        def keep(src):
            return not all(src.get(k) == v for k, v in key.items())
        pos = [(e, s) for e, s in zip(self.embeddings, self.sources) if keep(s)]
        neg = [(e, s) for e, s in zip(self.negatives, self.negative_sources) if keep(s)]
        self.embeddings, self.sources = [e for e, _ in pos], [s for _, s in pos]
        self.negatives, self.negative_sources = [e for e, _ in neg], [s for _, s in neg]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), np.array(self.embeddings).reshape(len(self.embeddings), -1))
        neg_path = path.with_suffix(".neg.npy")
        if self.negatives:
            np.save(neg_path, np.stack(self.negatives))
        elif neg_path.exists():
            neg_path.unlink()
        path.with_suffix(".json").write_text(json.dumps({
            "name": self.name, "threshold": self.threshold, "auto_threshold": self.auto_threshold,
            "sources": self.sources, "negative_sources": self.negative_sources}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VoiceProfile":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        embs = list(np.load(path.with_suffix(".npy")))
        neg_path = path.with_suffix(".neg.npy")
        negs = list(np.load(neg_path)) if neg_path.exists() else []
        return cls(meta["name"], embs, meta["threshold"], meta["sources"], negs,
                   meta.get("negative_sources", []), meta.get("auto_threshold", False))

    def extend(self, embeddings: list[np.ndarray], sources: list[dict]) -> None:
        self.embeddings += list(embeddings)
        self.sources += sources

    def add_negatives(self, embeddings: list[np.ndarray], sources: list[dict]) -> None:
        self.negatives += list(embeddings)
        self.negative_sources += sources


def _best_threshold(pos: list[float], neg: list[float]) -> float:
    """Threshold maximising balanced accuracy; ties go to the widest gap between classes."""
    values = sorted(set(pos + neg))
    candidates = [(a + b) / 2 for a, b in zip(values, values[1:])] or [DEFAULT_THRESHOLD]
    best, best_key = DEFAULT_THRESHOLD, None
    for t in candidates:
        acc = sum(p >= t for p in pos) / len(pos) + sum(n < t for n in neg) / len(neg)
        gap = min([p - t for p in pos if p >= t] + [1]) + min([t - n for n in neg if n < t] + [1])
        key = (acc, gap)
        if best_key is None or key > best_key:
            best, best_key = t, key
    return float(min(max(best, 0.05), 0.6))
