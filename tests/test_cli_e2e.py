import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compare_cli_end_to_end(tmp_path):
    out_dir = tmp_path / "report"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "burnin_subtitle_checker.cli",
            "compare",
            str(ROOT / "fixtures/e2e/transcript.json"),
            str(ROOT / "fixtures/e2e/ocr.json"),
            "--output-dir",
            str(out_dir),
            "--threshold",
            "0.75",
            "--formats",
            "html,json,csv",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Compared 4 segment(s); 2 need review." in completed.stdout
    assert (out_dir / "report.html").is_file()
    assert (out_dir / "report.csv").is_file()
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 4
    assert payload["summary"]["ok"] == 2
    assert payload["summary"]["review"] == 2
