import pytest

from burnin_subtitle_checker.exceptions import ConfigError
from burnin_subtitle_checker.srt import load_reference_srt, merge_cue_text, parse_srt_text

SAMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,500
वो कहाँ गई थी

2
00:00:04,000 --> 00:00:06,000
ಅವಳು ಎಲ್ಲಿಗೆ ಹೋದಳು

3
00:00:07,250 --> 00:00:09,750
<i>good night</i>
{\\an8}second line
"""


def test_parse_srt_text_extracts_timed_cues():
    cues = parse_srt_text(SAMPLE_SRT)

    assert [cue.start for cue in cues] == [1.0, 4.0, 7.25]
    assert [cue.end for cue in cues] == [3.5, 6.0, 9.75]
    assert cues[0].text == "वो कहाँ गई थी"
    assert cues[2].text == "good night second line"


def test_parse_srt_text_handles_period_separator_and_bom():
    cues = parse_srt_text("﻿1\n00:00:00.000 --> 00:00:01.000\nhello\n\n")
    assert cues[0].text == "hello"
    assert cues[0].start == 0.0
    assert cues[0].end == 1.0


def test_load_reference_srt_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_reference_srt(tmp_path / "missing.srt")


def test_load_reference_srt_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.srt"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_reference_srt(path)


def test_merge_cue_text_joins_in_order():
    cues = parse_srt_text(SAMPLE_SRT)
    text = merge_cue_text(cues)
    assert "वो कहाँ गई थी" in text
    assert "ಅವಳು ಎಲ್ಲಿಗೆ ಹೋದಳು" in text
    assert "good night second line" in text


def test_load_reference_srt_round_trip(tmp_path):
    path = tmp_path / "ref.srt"
    path.write_text(SAMPLE_SRT, encoding="utf-8")
    cues = load_reference_srt(path)
    assert len(cues) == 3
    assert cues[1].start == 4.0
