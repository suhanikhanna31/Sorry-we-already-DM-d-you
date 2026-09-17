# MANIFEST — what's in here and what isn't

This is the full repo: source code, the checked-in eval artifacts, tests,
CI, and the design write-up (`report/REPORT.md`). See `README.md` for setup
and exact reproduction commands — this file is about provenance: which
files are hand-authored vs. generated, and what row counts to expect if you
regenerate something.

## What's included (small, exact, worth keeping as-is)

- `eval/golden_set.csv` — the actual 210 hand-labeled examples referenced
  throughout the design doc (§5, §6). Columns: `id, customer_tweet_id,
  customer_text, reference_agent_reply, weak_intent_seed, gold_intent,
  gold_should_escalate, gold_escalate_reason`.
- `eval/golden_candidates.csv` — the 210-row stratified sample *before*
  hand-labeling (weak-label seed only). Paired with `golden_labels.py`
  (the hand-assigned labels, keyed by row position) via `make_golden_set.py`
  to produce `golden_set.csv` above.
- `eval/labeling_notes.md` — the sampling/labeling methodology writeup.
- `eval/results/predictions.csv` — every one of the 3 systems'
  (trivial/simple/main_agent) predictions on all 210 golden examples.
- `eval/results/metrics.json` — every number quoted in the design doc's
  §4 and §6 (intent accuracy/F1, confusion matrix, escalation P/R/F1,
  ROUGE-1, missed-escalation examples), computed by `eval/run_eval.py`.
- `eval/results/judge_sample.csv` + `judge_scores.csv` — the 40-example
  sample scored against the LLM-judge rubric, and
  `judge_scores_manual_data.py` — the actual hand-applied scores + rationale
  for each (see design doc §5 for why these are manually-applied rather than
  live-API scores in this environment).
- `src/*.py` — all pipeline code: `prepare_data.py` (raw → cleaned pairs),
  `build_silver_corpus.py` (+ weak-label column), `intents.py` (taxonomy +
  weak labeler), `retrieval.py` (TF-IDF grounding index), `classifier.py`
  (intent model), `draft.py` (reply adaptation), `escalation.py`
  (rule-based policy), `agent.py` (orchestration), `llm_agent.py` +
  `llm_client.py` (optional LLM-mode path).
- `tests/*.py` — unit tests for the dependency-free rule modules
  (`escalation.py`, `draft.py`, `intents.py`) plus a reproducibility
  regression test (`test_reproducibility.py`) that regenerates
  `eval/golden_set.csv` and `eval/results/judge_scores.csv` from their
  hand-authored sources and diffs against what's checked in.
- `.github/workflows/ci.yml` + `.github/scripts/make_fake_twcs.py` — CI
  that lints, runs the tests above, and runs the *entire real pipeline*
  end-to-end against a small synthetic dataset with the real Kaggle schema
  (no Kaggle credentials needed in CI).
- `README.md`, `requirements.txt`, `.gitignore` — setup, dependencies, and
  what's intentionally excluded from version control (see below).

## What's NOT included, and why

- `data_raw/archive-2/twcs/twcs.csv` (the raw ~516MB Kaggle dump) — gated,
  licensed data; not redistributable and not something a setup script
  should auto-fetch with embedded credentials. See `README.md` "Step 1" for
  exactly how to get it (Kaggle CLI command or manual download + exact
  target path).
- `data/spotify_pairs.csv`, `data/spotify_pairs_weak.csv` (the cleaned
  37,923-row corpus), `data/intent_classifier.pkl` (trained model),
  `data/retrieval_index.pkl` (TF-IDF index) — these are all large,
  **deterministically regeneratable** from the raw file by running, in
  order: `src/prepare_data.py` → `src/build_silver_corpus.py` (applies
  `intents.weak_label`) → `src/classifier.py` (train) → `src/retrieval.py`
  (build index). See `README.md` "Step 3" for exact commands. Excluded from
  version control purely to keep the repo lean; nothing here depends on
  randomness that would make a regenerated copy differ from the original,
  aside from the `train_test_split` in `classifier.py` (`random_state=13`,
  so it's actually fully deterministic too).

## Quick sanity check after regenerating

If you rebuild `data/spotify_pairs.csv` from scratch, the exact row counts
from the original run were: 43,265 raw SpotifyCares tweets → 43,092 paired
→ 39,610 after English filter → **37,923** final rows. If your rebuild
doesn't match these numbers closely, something in the pipeline diverged —
check `src/prepare_data.py` first.
