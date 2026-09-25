"""Local web UI for building a voice profile interactively.

Videos are transcribed and every segment is embedded once (cached under the data dir).
Labelling a segment adds or removes its embedding from the profile, so re-scoring every
segment after each click is just a few dot products and happens instantly.
"""
import json
import queue
import re
import shutil
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from ..audio import SAMPLE_RATE, load_audio
from ..output import _ts
from ..speaker import DEFAULT_THRESHOLD, VoiceProfile
from ..transcribe import Segment, embed_segment, recognize, whisper_model

STATIC = Path(__file__).parent / "static"
LABELS = ("coach", "other", "mixed")


class App:
    def __init__(self, profile_path: Path, data_dir: Path, model: str, names: list[str], speaker_name: str):
        self.profile_path = profile_path
        self.data_dir = data_dir
        self.model = model
        self.names = names
        self.lock = threading.Lock()
        self.meta_lock = threading.Lock()
        self.jobs: queue.Queue[str] = queue.Queue()
        self.embeddings: dict[str, np.ndarray] = {}
        (data_dir / "videos").mkdir(parents=True, exist_ok=True)
        if profile_path.with_suffix(".json").exists():
            self.profile = VoiceProfile.load(profile_path)
        else:
            self.profile = VoiceProfile(speaker_name, [], DEFAULT_THRESHOLD, [], auto_threshold=True)
        self.labels = self._load_labels()
        for vid in self.video_ids():
            if self.meta(vid)["status"] not in ("ready", "error"):
                self.jobs.put(vid)  # resume work interrupted by a restart
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- storage -------------------------------------------------------------------
    def vdir(self, vid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", vid):
            raise HTTPException(404, "unknown video")
        d = self.data_dir / "videos" / vid
        if not d.exists():
            raise HTTPException(404, "unknown video")
        return d

    def video_ids(self) -> list[str]:
        dirs = [d for d in (self.data_dir / "videos").iterdir() if (d / "meta.json").exists()]
        return [d.name for d in sorted(dirs, key=lambda d: json.loads((d / "meta.json").read_text())["created"])]

    def meta(self, vid: str) -> dict:
        return json.loads((self.vdir(vid) / "meta.json").read_text())

    def set_meta(self, vid: str, **updates) -> None:
        path = self.data_dir / "videos" / vid / "meta.json"
        with self.meta_lock:
            meta = json.loads(path.read_text()) if path.exists() else {}
            meta.update(updates)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta))
            tmp.replace(path)

    def segments(self, vid: str) -> list[dict]:
        path = self.vdir(vid) / "segments.json"
        return json.loads(path.read_text()) if path.exists() else []

    def segment_embeddings(self, vid: str) -> np.ndarray:
        if vid not in self.embeddings:
            self.embeddings[vid] = np.load(self.vdir(vid) / "embeddings.npy")
        return self.embeddings[vid]

    def _labels_path(self) -> Path:
        return self.profile_path.with_suffix(".labels.json")

    def _load_labels(self) -> dict:
        path = self._labels_path()
        return json.loads(path.read_text()) if path.exists() else {}

    # ---- processing ----------------------------------------------------------------
    def add_video(self, name: str) -> str:
        vid = uuid.uuid4().hex[:12]
        (self.data_dir / "videos" / vid).mkdir(parents=True)
        self.set_meta(vid, name=name, status="queued", progress=0.0, created=time.time(), error=None)
        return vid

    def _worker(self) -> None:
        while True:
            vid = self.jobs.get()
            try:
                self._process(vid)
            except Exception as e:  # keep the worker alive; surface the error in the UI
                traceback.print_exc()
                self.set_meta(vid, status="error", error=str(e))

    def _process(self, vid: str) -> None:
        d = self.data_dir / "videos" / vid
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("url") and not list(d.glob("source.*")):
            self.set_meta(vid, status="downloading", progress=0.0)
            download(meta["url"], d / "source.mp4", lambda f: self.set_meta(vid, progress=f))
        source = next(d.glob("source.*"))

        self.set_meta(vid, status="extracting audio", progress=0.0)
        audio = load_audio(source)
        sf.write(d / "audio.wav", audio, SAMPLE_RATE)
        self.set_meta(vid, duration=len(audio) / SAMPLE_RATE)

        self.set_meta(vid, status="loading speech model", progress=0.0)
        whisper_model(self.model)
        self.set_meta(vid, status="transcribing", progress=0.0)
        segs = recognize(audio, self.model, self.names, on_progress=lambda f: self.set_meta(vid, progress=f))

        self.set_meta(vid, status="analysing voices", progress=0.0)
        embs = []
        for i, seg in enumerate(segs):
            e = embed_segment(audio, seg)
            embs.append(np.full(192, np.nan, dtype=np.float32) if e is None else e.astype(np.float32))
            self.set_meta(vid, progress=(i + 1) / len(segs))
        np.save(d / "embeddings.npy", np.stack(embs) if embs else np.zeros((0, 192), np.float32))
        (d / "segments.json").write_text(json.dumps(
            [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in segs]))
        self.set_meta(vid, status="ready", progress=1.0)

    # ---- scoring and labelling ------------------------------------------------------
    def scored(self, vid: str) -> list[dict]:
        segs = self.segments(vid)
        if not segs:
            return []
        embs = self.segment_embeddings(vid)
        out = []
        for i, (seg, e) in enumerate(zip(segs, embs)):
            emb = None if np.isnan(e).any() else e
            # A labelled segment is scored without its own embedding, so its score shows
            # how well the rest of the profile would have recognised it.
            score, neg, predicted = self.profile.classify(
                emb, exclude=self._own_index(self.profile.sources, vid, i),
                exclude_negative=self._own_index(self.profile.negative_sources, vid, i))
            out.append({**seg, "i": i, "score": score, "negative_score": neg, "predicted": predicted,
                        "label": self.labels.get(f"{vid}:{i}")})
        return out

    @staticmethod
    def _own_index(sources: list[dict], vid: str, i: int) -> int | None:
        for idx, s in enumerate(sources):
            if s.get("video_id") == vid and s.get("seg") == i:
                return idx
        return None

    def set_label(self, vid: str, i: int, label: str | None) -> None:
        segs = self.segments(vid)
        if not 0 <= i < len(segs):
            raise HTTPException(404, "unknown segment")
        key = f"{vid}:{i}"
        self.profile.remove_source({"video_id": vid, "seg": i})
        self.labels.pop(key, None)
        if label:
            self.labels[key] = label
            e = self.segment_embeddings(vid)[i]
            src = {"file": self.meta(vid)["name"], "video_id": vid, "seg": i,
                   "start": segs[i]["start"], "end": segs[i]["end"]}
            if not np.isnan(e).any():
                if label == "coach":
                    self.profile.extend([e], [src])
                elif label == "other":
                    self.profile.add_negatives([e], [src])
        self.save()

    def save(self) -> None:
        self.profile.calibrate()
        if self.profile.embeddings:
            self.profile.save(self.profile_path)
        self._labels_path().write_text(json.dumps(self.labels, indent=1))

    def summary(self) -> dict:
        stats = self.profile.calibrate()
        return {
            "name": self.profile.name,
            "threshold": self.profile.threshold,
            "auto_threshold": self.profile.auto_threshold,
            "positives": len(self.profile.embeddings),
            "negatives": len(self.profile.negatives),
            "enrolled_outside_ui": sum("video_id" not in s for s in self.profile.sources),
            "accuracy": stats["accuracy"],
            "profile_path": str(self.profile_path.with_suffix(".json")),
        }

    def export(self, vid: str, fmt: str, include_uncertain: bool) -> str:
        keep = {"match", "uncertain"} if include_uncertain else {"match"}
        lines = []
        for s in self.scored(vid):
            # Your labels override the prediction.
            is_coach = s["label"] == "coach" if s["label"] else s["predicted"] in keep
            if is_coach:
                lines.append(Segment(s["start"], s["end"], s["text"]))
        if fmt == "srt":
            return "".join(f"{n}\n{_ts(s.start)} --> {_ts(s.end)}\n{s.text}\n\n" for n, s in enumerate(lines, 1))
        return "".join(f"[{_ts(s.start, '.')[3:]}] {s.text}\n" for s in lines)


def drive_download_url(url: str) -> str:
    m = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if "drive.google.com" in url and m:
        return f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t"
    return url


def download(url: str, dest: Path, on_progress) -> None:
    req = urllib.request.Request(drive_download_url(url), headers={"User-Agent": "coach-voice"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError("link returned a web page, not a video; make sure it is shared as "
                               "'Anyone with the link'")
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                on_progress(done / total)


def create_app(profile_path: str, data_dir: str = "data", model: str = "medium.en",
               names: list[str] | None = None, speaker_name: str = "Coach") -> FastAPI:
    state = App(Path(profile_path), Path(data_dir), model, names or [], speaker_name)
    api = FastAPI(title="Coach voice trainer")

    @api.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @api.get("/api/state")
    def get_state():
        videos = [{"id": v, **state.meta(v)} for v in state.video_ids()]
        for v in videos:
            v["labelled"] = sum(k.startswith(v["id"] + ":") for k in state.labels)
        return {"profile": state.summary(), "videos": videos, "names": state.names}

    @api.post("/api/videos")
    async def add_video(file: UploadFile | None = File(None), url: str = Form(""), path: str = Form("")):
        if file is not None and file.filename:
            vid = state.add_video(file.filename)
            suffix = Path(file.filename).suffix or ".mp4"
            with open(state.vdir(vid) / f"source{suffix}", "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif url.strip():
            vid = state.add_video(url.strip().split("?")[0].rstrip("/").split("/")[-1] or "download")
            if "drive.google.com" in url:
                state.set_meta(vid, name="Google Drive video")
            state.set_meta(vid, url=url.strip())
        elif path.strip():
            src = Path(path.strip()).expanduser()
            if not src.is_file():
                raise HTTPException(400, f"file not found: {src}")
            vid = state.add_video(src.name)
            (state.vdir(vid) / f"source{src.suffix}").symlink_to(src.resolve())
        else:
            raise HTTPException(400, "choose a file, a link or a path")
        state.jobs.put(vid)
        return {"id": vid}

    @api.patch("/api/videos/{vid}")
    def rename_video(vid: str, body: dict = Body(...)):
        state.vdir(vid)
        state.set_meta(vid, name=str(body.get("name", "")).strip() or state.meta(vid)["name"])
        return {"ok": True}

    @api.get("/api/videos/{vid}")
    def get_video(vid: str):
        return {"id": vid, **state.meta(vid), "segments": state.scored(vid) if
                state.meta(vid)["status"] == "ready" else []}

    @api.get("/api/videos/{vid}/media")
    def media(vid: str):
        source = next(state.vdir(vid).glob("source.*"), None)
        if source is None:
            raise HTTPException(404, "not downloaded yet")
        return FileResponse(source)

    @api.post("/api/videos/{vid}/segments/{i}/label")
    def label(vid: str, i: int, body: dict = Body(...)):
        value = body.get("label")
        if value not in (*LABELS, None):
            raise HTTPException(400, f"label must be one of {LABELS} or null")
        with state.lock:
            state.set_label(vid, i, value)
            return {"profile": state.summary(), "segments": state.scored(vid)}

    @api.post("/api/threshold")
    def threshold(body: dict = Body(...)):
        with state.lock:
            if body.get("auto"):
                state.profile.auto_threshold = True
            else:
                state.profile.auto_threshold = False
                state.profile.threshold = float(body["value"])
            state.save()
            return {"profile": state.summary()}

    @api.get("/api/videos/{vid}/export.{fmt}")
    def export(vid: str, fmt: str, uncertain: bool = False):
        if fmt not in ("srt", "txt"):
            raise HTTPException(404)
        name = Path(state.meta(vid)["name"]).stem
        return PlainTextResponse(state.export(vid, fmt, uncertain), headers={
            "Content-Disposition": f'attachment; filename="{name}.coach.{fmt}"'})

    return api
