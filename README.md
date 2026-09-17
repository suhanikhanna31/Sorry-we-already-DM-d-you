# SpotifyCares AI Support Agent

An AI support agent for **SpotifyCares** (Spotify's Twitter support account),
built from the `thoughtvector/customer-support-on-twitter` Kaggle dataset.
Given a customer message, it (1) classifies intent into one of six taxonomy
classes, (2) drafts a reply grounded in real historical SpotifyCares replies
via retrieval, and (3) decides auto-handle vs. escalate-to-human with a
stated reason.

Full design rationale, results vs. two baselines, failure analysis, the
"what's misleading about my headline number" section, and the decision log
live in **[`report/REPORT.md`](report/REPORT.md)** — read that for the *why*.
This README is the *how*: how to get the data and reproduce the numbers.

**Two clocks, not one.** Getting the raw dataset onto your machine (Step 1)
requires a free Kaggle account and isn't something a setup script should
auto-fetch with embedded credentials — that's a normal, one-time, human step
and is **not** part of the reproducibility claim. Once `twcs.csv` is in
place, **Step 2 onward runs start-to-finish in well under 15 minutes** on a
laptop, no GPU, no API key, fully offline.

---

## Step 1: get the dataset

The raw data is **not** included in this repo (it's ~516MB and Kaggle's
license doesn't permit redistribution) — `data_raw/` is gitignored.

**Option A — Kaggle CLI (recommended if you have an API token):**

```bash
pip install kaggle
# requires ~/.kaggle/kaggle.json — see https://www.kaggle.com/docs/api
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data_raw --unzip
mkdir -p data_raw/archive-2/twcs
mv data_raw/twcs.csv data_raw/archive-2/twcs/twcs.csv
```

**Option B — manual download:**

1. Download the dataset from
   <https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>
   (Kaggle account required, free).
2. Unzip it and place the CSV so the final path is exactly:

   ```
   data_raw/archive-2/twcs/twcs.csv
   ```

**Verify you're set up correctly:**

```bash
ls -la data_raw/archive-2/twcs/twcs.csv   # should be ~516MB
head -1 data_raw/archive-2/twcs/twcs.csv  # should show the schema header below
```

Expected header:
`tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id`

> If you're picking this project up inside an environment with no direct
> Kaggle access (e.g. a sandboxed agent session), don't assume it can fetch
> the file on its own — provide `twcs.csv` as an upload, or point to wherever
> it's already on disk, and place/symlink it at the path above.

---

## Step 2: install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Only four real runtime dependencies: `pandas`, `numpy`, `scikit-learn`,
`langdetect`. Everything else in `requirements.txt` is dev/CI tooling
(`pytest`, `flake8`).

---

## Step 3: build the corpus, train, index — the reproducible pipeline

Run these **from the repo root**, in order. Each regenerates a file already
described in [`MANIFEST.md`](MANIFEST.md); expected row counts are noted
there so you can sanity-check your run.

```bash
python src/prepare_data.py          # twcs.csv -> data/spotify_pairs.csv (~37,923 rows)
python src/build_silver_corpus.py   # + weak_intent column -> data/spotify_pairs_weak.csv
python src/classifier.py            # trains -> data/intent_classifier.pkl
python src/retrieval.py             # builds -> data/retrieval_index.pkl
```

Timing on a laptop: `prepare_data.py` is the slow one (language-detecting
~40K rows, ~1–2 minutes); the rest is seconds. **Total: a few minutes**, well
inside the 15-minute budget counted from when `twcs.csv` is already on disk.

Try the agent on a single message:

```bash
echo "the app keeps crashing every time I open it on my phone" | python src/agent.py
```

```json
{
  "customer_text": "the app keeps crashing every time I open it on my phone",
  "intent": "technical_playback",
  "intent_confidence": 0.92,
  "draft_reply": "Hey! Sorry to hear that. Can you tell us your device, OS and app version?",
  "grounded_in": "...",
  "grounding_similarity": 0.77,
  "should_escalate": false,
  "escalate_reason": "routine request matching a known safe-to-automate pattern"
}
```

---

## Step 4: reproduce the headline eval numbers

The hand-labeled golden set (`eval/golden_set.csv`, 210 examples) is checked
into the repo already — you don't need to relabel anything to reproduce the
headline numbers, only Steps 2–3 above (the trained classifier + retrieval
index).

```bash
python eval/run_eval.py
```

Writes `eval/results/predictions.csv` and `eval/results/metrics.json` —
these are the exact numbers quoted in `report/REPORT.md` §4 and §6
(intent accuracy/macro-F1, full confusion matrix, escalation
precision/recall/F1, ROUGE-1 vs. reference). Console output shows a quick
summary per system (`trivial`, `simple`, `main_agent`).

### Regenerating the golden set itself (optional — not required for headline numbers)

`eval/golden_set.csv` is *generated* from hand-authored source files, and
that generation is itself reproducible and covered by CI (see
`tests/test_reproducibility.py`):

```bash
python eval/build_golden_candidates.py   # needs data/spotify_pairs_weak.csv from Step 3
python eval/make_golden_set.py           # eval/golden_candidates.csv + eval/golden_labels.py -> eval/golden_set.csv
```

### Optional: LLM mode + live LLM-as-judge (requires `ANTHROPIC_API_KEY`)

Not required for the headline numbers (those are 100% local/offline — see
`report/REPORT.md` §5 for why the checked-in judge scores were produced by a
human applying the exact rubric, not a live model call, in the environment
this was built in).

```bash
export ANTHROPIC_API_KEY=sk-...
echo "some customer message" | python src/agent.py --mode llm
python eval/llm_judge.py --n 40 --out eval/results/judge_scores_live.csv
```

---

## Running the test suite

```bash
pytest tests/ -v
```

Covers the dependency-free rule modules (`src/escalation.py`,
`src/draft.py`, `src/intents.py`) and a reproducibility regression check
that regenerates `eval/golden_set.csv` and `eval/results/judge_scores.csv`
from their hand-authored source files and diffs them against what's checked
in — so a hand-edit to a label file that isn't followed by re-running the
generator script gets caught in CI, not discovered later. None of this
needs the raw dataset.

CI (`.github/workflows/ci.yml`) runs this on every push/PR, plus a
full-pipeline smoke test against a small synthetic dataset with the same
schema as the real one (`.github/scripts/make_fake_twcs.py`) — this
exercises `prepare_data.py` → `build_silver_corpus.py` → `classifier.py` →
`retrieval.py` → `agent.py` → `run_eval.py` end-to-end on every push without
needing Kaggle credentials in CI. It produces meaningless model numbers (the
synthetic data is tiny and repetitive) — its job is to catch integration
breaks (a renamed column, a broken import), not to validate quality.

---

## Project layout

```
data_raw/            # gitignored — put twcs.csv here (Step 1)
data/                # gitignored — regenerated by Step 3 (corpus, model, index)
src/                 # pipeline code
  prepare_data.py        # raw twcs.csv -> cleaned (customer, agent_reply) pairs
  build_silver_corpus.py # + weak_intent labels (silver, not ground truth)
  intents.py              # 6-class taxonomy + weak-label heuristic
  classifier.py            # TF-IDF + LogisticRegression intent classifier
  retrieval.py              # TF-IDF grounding index over historical replies
  draft.py                   # adapts a retrieved reply into a new draft
  escalation.py                # rule-based auto-handle/escalate policy
  agent.py                      # orchestrates the four steps above (CLI)
  llm_agent.py / llm_client.py  # optional --mode llm path (needs API key)
eval/                 # golden set, baselines, eval harness, LLM-judge
  golden_candidates.csv / golden_labels.py / make_golden_set.py
  golden_set.csv            # the 210 hand-labeled examples (checked in)
  baselines.py                # trivial + simple baselines
  run_eval.py                   # computes all headline metrics
  llm_judge.py / judge_scores_manual_data.py / build_judge_scores.py
  results/                        # predictions.csv, metrics.json, judge_scores.csv (checked in)
report/
  REPORT.md             # full write-up: framing, results, failure analysis,
                         # "what's misleading about my headline number",
                         # decision log
tests/                # unit tests + reproducibility regression tests
.github/workflows/ci.yml          # lint + tests + full-pipeline smoke test
.github/scripts/make_fake_twcs.py # synthetic dataset generator, CI-only
MANIFEST.md           # exact file-by-file provenance + expected row counts
```

## Why TF-IDF and not embeddings?

Deliberate choice for a fully offline, sub-15-minute, reproducible pipeline
with no GPU and no downloaded model weights. `report/REPORT.md` §8 names the
natural upgrade path (SetFit, a contrastive sentence-transformer few-shot
approach) explicitly, and why it wasn't used here given the reproducibility
constraint.
