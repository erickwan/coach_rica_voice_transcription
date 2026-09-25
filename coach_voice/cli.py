"""
Usage:
  python -m coach_voice enroll REFERENCE_VIDEO --segment 11.0-12.1 --segment 20.9-25.0 \
      --name "Coach Rica" --profile profiles/coach_rica
  python -m coach_voice sample VIDEO --profile profiles/coach_rica --out samples/
  python -m coach_voice enroll VIDEO --pick 3,5,8 --sample-dir samples/VIDEO_STEM --profile profiles/coach_rica --add
  python -m coach_voice transcribe VIDEO [VIDEO ...] --profile profiles/coach_rica --out transcripts/
"""
import argparse
import sys
from pathlib import Path

from .audio import load_audio, slice_audio
from .output import write_review, write_srt, write_txt
from .speaker import VoiceProfile, embed

DEFAULT_THRESHOLD = 0.18
# Long reference segments are split into chunks so the profile captures more variation.
ENROLL_CHUNK = 3.0


def parse_range(text: str) -> tuple[float, float]:
    def secs(t: str) -> float:
        parts = [float(p) for p in t.split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

    start, end = text.split("-")
    return secs(start), secs(end)


def picked_ranges(sample_dir: str, picks: str) -> list[tuple[float, float]]:
    rows = {}
    for line in (Path(sample_dir) / "segments.tsv").read_text().splitlines()[1:]:
        num, start, end, *_ = line.split("\t")
        rows[int(num)] = (float(start), float(end))
    wanted = [int(n) for n in picks.split(",") if n.strip()]
    missing = [n for n in wanted if n not in rows]
    if missing:
        sys.exit(f"segment numbers not in {sample_dir}/segments.tsv: {missing}")
    return [rows[n] for n in wanted]


def enroll(args: argparse.Namespace) -> None:
    ranges = [parse_range(r) for r in args.segment or []]
    if args.pick:
        if not args.sample_dir:
            sys.exit("--pick needs --sample-dir (the folder written by the sample command)")
        ranges += picked_ranges(args.sample_dir, args.pick)
    if not ranges:
        sys.exit("give --segment and/or --pick")
    audio = load_audio(args.reference)
    embeddings, sources = [], []
    for start, end in ranges:
        n = max(1, round((end - start) / ENROLL_CHUNK))
        step = (end - start) / n
        for i in range(n):
            s, e = start + i * step, start + (i + 1) * step
            emb = embed(slice_audio(audio, s, e))
            if emb is None:
                print(f"skipping {s:.2f}-{e:.2f}: too short", file=sys.stderr)
                continue
            embeddings.append(emb)
            sources.append({"file": Path(args.reference).name, "start": round(s, 2), "end": round(e, 2)})
    if not embeddings:
        sys.exit("no usable reference segments")

    profile_json = Path(args.profile).with_suffix(".json")
    if args.add and profile_json.exists():
        profile = VoiceProfile.load(args.profile)
        profile.extend(embeddings, sources)
        if args.threshold is not None:
            profile.threshold = args.threshold
    else:
        threshold = DEFAULT_THRESHOLD if args.threshold is None else args.threshold
        profile = VoiceProfile(args.name, embeddings, threshold, sources)
    profile.save(args.profile)
    print(f"saved {profile.name!r} with {len(profile.embeddings)} embeddings to {profile_json}")


def transcribe(args: argparse.Namespace) -> None:
    from .transcribe import recognize, score_segments

    profile = VoiceProfile.load(args.profile)
    if args.threshold is not None:
        profile.threshold = args.threshold
    names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video in args.videos:
        print(f"== {video}", file=sys.stderr)
        audio = load_audio(video)
        segments = recognize(audio, args.model, names, args.language)
        score_segments(audio, segments, profile)
        keep = {"match", "uncertain"} if args.include_uncertain else {"match"}
        coach = [s for s in segments if s.label in keep]

        stem = out_dir / Path(video).stem
        write_srt(coach, stem.with_suffix(".coach.srt"))
        write_txt(coach, stem.with_suffix(".coach.txt"))
        write_review(segments, stem.with_suffix(".review.tsv"))
        for s in segments:
            score = "  -  " if s.score is None else f"{s.score:.2f}"
            print(f"  {s.start:7.2f}-{s.end:7.2f}  {score}  {s.label:9s}  {s.text}")
        print(f"  -> {len(coach)}/{len(segments)} segments attributed to {profile.name}; "
              f"wrote {stem}.coach.srt/.coach.txt/.review.tsv", file=sys.stderr)


def sample(args: argparse.Namespace) -> None:
    """Transcribe a clip into numbered, listenable snippets so you can pick the target
    speaker's lines for enrollment."""
    import soundfile as sf

    from .audio import SAMPLE_RATE
    from .transcribe import recognize, score_segments

    names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else []
    audio = load_audio(args.video)
    segments = recognize(audio, args.model, names, args.language)
    profile = VoiceProfile.load(args.profile) if args.profile else None
    if profile:
        score_segments(audio, segments, profile)

    out = Path(args.out) / Path(args.video).stem
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = ["num\tstart\tend\tscore\tlabel\ttext"]
    for i, seg in enumerate(segments, 1):
        clip = f"{i:03d}_{seg.start:.1f}-{seg.end:.1f}.wav"
        sf.write(out / "clips" / clip, slice_audio(audio, max(0.0, seg.start - 0.2), seg.end + 0.2), SAMPLE_RATE)
        score = "" if seg.score is None else f"{seg.score:.3f}"
        rows.append(f"{i}\t{seg.start:.2f}\t{seg.end:.2f}\t{score}\t{seg.label}\t{seg.text}")
        print(f"  {i:3d}  {seg.start:7.2f}-{seg.end:7.2f}  {score or '  -  ':5s}  {seg.label:9s}  {seg.text}")
    (out / "segments.tsv").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {len(segments)} snippets to {out}/clips/. Listen, then enroll the ones that are only "
          f"the target speaker:\n  python -m coach_voice enroll {args.video} --sample-dir {out} "
          f"--pick 1,2,3 --profile {args.profile or 'profiles/NAME'}" + (" --add" if profile else ""),
          file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="coach_voice", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="build (or extend with --add) a voice profile from reference segments")
    e.add_argument("reference", help="audio/video file containing the target speaker")
    e.add_argument("--segment", action="append",
                   help="time range spoken only by the target, e.g. 11.0-12.1 or 0:20.9-0:25")
    e.add_argument("--sample-dir", help="folder written by the sample command")
    e.add_argument("--pick", help="comma-separated segment numbers from the sample folder")
    e.add_argument("--profile", required=True, help="profile path prefix, e.g. profiles/coach_rica")
    e.add_argument("--name", default="target speaker")
    e.add_argument("--threshold", type=float, default=None)
    e.add_argument("--add", action="store_true", help="append to an existing profile")
    e.set_defaults(func=enroll)

    m = sub.add_parser("sample", help="cut a clip into numbered snippets to pick enrollment samples from")
    m.add_argument("video")
    m.add_argument("--profile", help="existing profile, to show how well each snippet matches")
    m.add_argument("--out", default="samples")
    m.add_argument("--names", default="")
    m.add_argument("--model", default="medium.en")
    m.add_argument("--language", default="en")
    m.set_defaults(func=sample)

    t = sub.add_parser("transcribe", help="transcribe only the enrolled speaker in one or more videos")
    t.add_argument("videos", nargs="+")
    t.add_argument("--profile", required=True)
    t.add_argument("--out", default="transcripts")
    t.add_argument("--names", default="", help="comma-separated names to help Whisper spell them")
    t.add_argument("--model", default="medium.en",
                   help="faster-whisper model: small.en (fast), medium.en (default), large-v3 (best)")
    t.add_argument("--language", default="en")
    t.add_argument("--threshold", type=float, default=None, help="override the profile's match threshold")
    t.add_argument("--include-uncertain", action="store_true")
    t.set_defaults(func=transcribe)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
