# Test Media Sources

The repo should not commit third-party videos. Put downloaded media under
`test-media/` and keep generated reports under `reports/`.

## Synthetic Stress Media

Use the local generator when you need predictable small mismatches, missing
subtitles, timing boundaries, long videos, or broad Indic script coverage:

```bash
python -m pip install -e '.[fixtures]'
python scripts/build_stress_fixture.py /tmp/burnsub-stress/bundle --target-duration 900
PYTHONPATH=src python -m burnin_subtitle_checker.cli --quiet check \
  /tmp/burnsub-stress/bundle/video.mp4 \
  --transcript-json /tmp/burnsub-stress/bundle/transcript.json \
  --reference-srt /tmp/burnsub-stress/bundle/reference.srt \
  --output-dir /tmp/burnsub-stress/report \
  --ocr-languages "$(cat /tmp/burnsub-stress/bundle/ocr_languages.txt)" \
  --crop-bottom-percent 34 \
  --frame-offsets 0 \
  --formats html,json,csv \
  --save-artifacts
```

Edit `fixtures/stress/spec.json` or the generated `burned_subtitles.srt`, then
rerun the generator. Use `--target-duration 3600` for an hour-long fixture. The
generator uses ffmpeg's `subtitles` filter when available and falls back to
Pillow frame rendering when installed.

## Kannada Captioned Sources

Wikimedia Commons has a small but useful Kannada closed-caption category:

- https://commons.wikimedia.org/wiki/Category:Files_with_closed_captioning_in_Kannada
- https://commons.wikimedia.org/wiki/File:Cassini%27s_Grand_Finale.ogv
- https://commons.wikimedia.org/wiki/File:Creative_Commons_and_Commerce.ogv
- https://commons.wikimedia.org/wiki/File:Open_Letter_for_Free_Access_to_Wikipedia.webm
- https://commons.wikimedia.org/wiki/File:Rudy_Mancuso_%26_Maia_Mitchell_-_Magic_(Official_Music_Video).webm
- https://commons.wikimedia.org/wiki/File:%C2%BFQu%C3%A9_es_Wikipedia%3F.ogv

Example download for a local Kannada burn-in test:

```bash
mkdir -p test-media/commons-cassini
curl -L -o test-media/commons-cassini/source.ogv \
  "https://commons.wikimedia.org/wiki/Special:Redirect/file/Cassini%27s_Grand_Finale.ogv"
curl -L -o test-media/commons-cassini/kn.srt \
  "https://commons.wikimedia.org/w/index.php?title=TimedText:Cassini%27s_Grand_Finale.ogv.kn.srt&action=raw"
cp test-media/commons-cassini/kn.srt test-media/commons-cassini/kn-mismatch.srt
```

Edit `kn-mismatch.srt`, then burn it into a new video:

```bash
ffmpeg -hide_banner -y \
  -i test-media/commons-cassini/source.ogv \
  -vf "subtitles=test-media/commons-cassini/kn-mismatch.srt:force_style='FontName=Noto Sans Kannada,FontSize=42,Outline=3,MarginV=48'" \
  -c:a copy \
  test-media/commons-cassini/burnin-kn-mismatch.mp4
```

## Broader Indic Sources

The Commons subtitle index is the best starting point for reusable samples:

- All subtitle categories: https://commons.wikimedia.org/wiki/Category:Videos_with_subtitles
- Hindi: https://commons.wikimedia.org/wiki/Category:Videos_with_Hindi_subtitles
- Tamil: https://commons.wikimedia.org/wiki/Category:Videos_with_Tamil_subtitles
- Telugu: https://commons.wikimedia.org/wiki/Category:Videos_with_Telugu_subtitles
- Malayalam: https://commons.wikimedia.org/wiki/Category:Videos_with_Malayalam_subtitles
- Marathi: https://commons.wikimedia.org/wiki/Category:Videos_with_Marathi_subtitles
- Gujarati: https://commons.wikimedia.org/wiki/Category:Videos_with_Gujarati_subtitles
- Punjabi: https://commons.wikimedia.org/wiki/Category:Videos_with_Punjabi_subtitles

For long multilingual subtitles, this Commons file is useful but large:

- https://commons.wikimedia.org/wiki/File:How_I_Forced_Elon_Musk_To_Hug_Me.webm

Government/OpenSpeaks samples are useful for real-world cadence and licensing
variety:

- https://commons.wikimedia.org/wiki/File:Prime_Minister_Narendra_Modi%27s_virtual_address_to_the_75th_United_Nations_General_Assembly_session,_2020_(Hindi_with_English_subtitles).webm
- https://commons.wikimedia.org/wiki/File:PM_Narendra_Modi_on_125th_birth_of_Prabhupada_(Hindi_with_English_subtitles_).webm
- https://commons.wikimedia.org/wiki/File:OpenSpeaks-Gbm-Jaunpuri-Sampati-Information_Access_and_Aadhaar.webm

## Licensing Notes

Check each file page before redistributing anything. Commons category pages mix
licenses. NASA-origin media may be public domain, Commons timed-text pages are
usually under Commons text licensing, GODL-India files require attribution and
non-endorsement care, and CC BY-SA/CC BY-NC-SA sources can affect derivatives.
For open-source fixtures, prefer generated synthetic media or document optional
download steps instead of committing third-party videos.
