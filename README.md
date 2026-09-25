# Coach voice transcription

Transcribe only what one specific person (Coach Rica) says in game/practice videos, ignoring other voices.

## How it works

1. **Enroll**: you give time ranges from a reference clip where only the coach is speaking. Each range is turned
   into a speaker embedding with [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
   (a pretrained speaker-recognition model). The embeddings are saved as a voice profile in `profiles/`.
   Nothing is fine-tuned: a few seconds of clean speech per sample is enough, and more samples make matching more reliable.
2. **Transcribe**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes the new video in short
   speech chunks, which avoids Whisper skipping whole passages of noisy field audio. The transcript is split at
   pauses between words, and each piece is scored against the profile by cosine similarity.
   Pieces above the threshold are kept as the coach's lines.
3. **Improve**: label lines as coach / not coach in the web page. "Coach" lines are added to the profile. "Not coach" lines
   teach it which similar voices to reject, and they set the threshold automatically.

## Setup

```bash
pip install -r requirements.txt
```

The Whisper and ECAPA models download automatically on first run. A CPU is enough: `medium.en` takes about 1–2 minutes per
3 minutes of video on 4 cores.

## Web page (recommended)

```bash
python -m coach_voice ui          # then open http://127.0.0.1:8000
```

1. **Add a video**: upload a file, paste a Google Drive share link (shared as "Anyone with the link"), or give a path
   on your computer. It is transcribed in the background, which takes about a minute per 3 minutes of video.
2. **Review lines**: click a line to watch that moment. Mark it with a button or a key:
   - **Coach** (<kbd>C</kbd>): only the coach is speaking
   - **Not coach** (<kbd>X</kbd>): someone else
   - **Mixed** (<kbd>M</kbd>): you can't tell, or several voices
3. **Keep going**: every label updates the profile and all scores refresh at once. The list is sorted so the lines the
   profile is least sure about come first, and after each label it moves to the next one and plays it.
4. **Check progress**: the header shows:
   - **Held-out accuracy**: each labelled line is scored by a profile built without it, and this is how often that
     score agrees with your label
   - **Threshold**: the cut-off score, auto-tuned from your labels unless you drag the slider
5. **Export**: download the coach's lines for any video as subtitles (.srt) or text.

Labels are saved in `profiles/coach_rica.labels.json`. Processed videos are cached in `data/`. The saved profile is the
same one the command line uses. Options: `--profile`, `--names` (player names to help Whisper), `--model`, `--port`.

## Command line

```bash
# 1. Build a profile. The included profile was built from the reference clip
#    ("go Vikram, keep going Vikram, Joshua, Joshua, hey I want you to stay close to 4 and 5"):
python -m coach_voice enroll reference.mp4 --name "Coach Rica" --profile profiles/coach_rica \
    --segment 11.0-12.1 --segment 14.1-15.6 --segment 15.9-16.9 --segment 20.9-25.0

# 2. Transcribe the coach in other videos
python -m coach_voice transcribe game1.mp4 game2.mp4 --profile profiles/coach_rica \
    --names "Vikram,Joshua,Ryan,Zoey,Dash,Kendra" --out transcripts/
```

For each video, `transcribe` writes these files:

| file | contents |
|---|---|
| `NAME.coach.srt` | the coach's lines as subtitles, which you can load in a video player alongside the video |
| `NAME.coach.txt` | the coach's lines as plain text with timestamps |
| `NAME.review.tsv` | every recognised line with its score and label (`match` / `uncertain` / `other`) |

`--names` passes player names to Whisper as hotwords. Without it, "Vikram" came out as "Big Grub".

## Adding more samples (recommended)

The profile starts from only about 7 seconds of speech. You can add more samples from any video:

```bash
# Cut the video into numbered snippets, each scored against the current profile
python -m coach_voice sample practice2.mp4 --profile profiles/coach_rica --names "Vikram,Joshua" --out samples/

# Listen to samples/practice2/clips/*.wav and check samples/practice2/segments.tsv, then add
# the snippets that contain only the coach's voice:
python -m coach_voice enroll practice2.mp4 --sample-dir samples/practice2 --pick 1,3,4 \
    --profile profiles/coach_rica --add
```

Only pick snippets where the coach is the only voice; one wrong sample hurts more than a missing one.
Samples from different videos and conditions (wind, distance, calm talk and shouting) help the most.

## Tuning

- Scores on the reference clip:
  - the coach's lines scored about 0.25–0.65
  - other adults' voices scored 0.05–0.12
  - the default threshold is `0.18`
- Override the threshold per run with `--threshold`, or save a new one in the profile with `enroll ... --add --threshold 0.2`.
  Check `review.tsv` to choose a value.
- Pass `--include-uncertain` to also keep lines scoring just below the threshold.
- `--model large-v3` is slower but more accurate on noisy audio. `small.en` is fast but missed speech on the reference clip.

## Limitations

- When another person speaks right after the coach without a pause, both can end up in one line.
  That line gets a mixed score.
- Lines of one or two words (under about 1s) have less reliable scores.
- Whisper can still invent words over crowd noise. Passing player names as hints sometimes makes it write out a list
  of those names ("Ollie Ryan Zoey Vikram"). Mark such lines **Mixed**.
  Adding more samples improves speaker matching, but not Whisper's recognition errors.
