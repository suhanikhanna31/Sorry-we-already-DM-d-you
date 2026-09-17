"""
test_reproducibility.py

The golden set (eval/golden_set.csv) and the judge scores
(eval/results/judge_scores.csv) are both *generated* files -- the real
source of truth is eval/golden_labels.py (hand-assigned labels) +
eval/golden_candidates.csv, and eval/judge_scores_manual_data.py +
eval/results/judge_sample.csv, respectively (see eval/make_golden_set.py
and eval/build_judge_scores.py).

This test regenerates both into a scratch copy of the repo and diffs them
against what's checked in, so that if someone edits golden_labels.py (or
judge_scores_manual_data.py) without re-running the generator script, CI
catches the drift immediately -- without needing the ~500MB raw dataset or
any network access.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_repo_subset(tmp_path: Path) -> Path:
    dest = tmp_path / "repo"
    dest.mkdir()
    for rel in ["eval", "src"]:
        shutil.copytree(REPO_ROOT / rel, dest / rel)
    return dest


def test_golden_set_matches_hand_authored_labels(tmp_path):
    repo = _copy_repo_subset(tmp_path)
    result = subprocess.run(
        [sys.executable, "eval/make_golden_set.py"],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    regenerated = pd.read_csv(repo / "eval" / "golden_set.csv")
    checked_in = pd.read_csv(REPO_ROOT / "eval" / "golden_set.csv")
    pd.testing.assert_frame_equal(regenerated, checked_in)


def test_judge_scores_match_hand_authored_scores(tmp_path):
    repo = _copy_repo_subset(tmp_path)
    result = subprocess.run(
        [sys.executable, "eval/build_judge_scores.py"],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    regenerated = pd.read_csv(repo / "eval" / "results" / "judge_scores.csv")
    checked_in = pd.read_csv(REPO_ROOT / "eval" / "results" / "judge_scores.csv")
    pd.testing.assert_frame_equal(regenerated, checked_in)
