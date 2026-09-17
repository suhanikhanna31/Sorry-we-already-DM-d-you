# Golden set: sampling & labeling notes

**File:** `eval/golden_set.csv` (210 rows)

## Sampling

1. Ran the keyword `weak_label` heuristic (`src/intents.py`) over the full
   37,923-row cleaned SpotifyCares corpus. This heuristic is deliberately
   crude (regex keyword buckets) -- it exists only to keep the sample from
   being 90% "vague complaint" (the majority natural class), not to define
   ground truth.
2. Stratified: drew 35 candidates per weak-label bucket (6 buckets x 35 =
   210), and within each bucket spread the sample across months so a single
   viral tweet or outage day couldn't dominate a bucket with near-duplicates.
   Script: `eval/build_golden_candidates.py`.
3. Every one of the 210 candidates was kept (no further filtering) -- see
   `eval/golden_candidates.csv` for the raw pool.

## Labeling

Every row was read by hand (customer message + the real historical agent
reply, for context) and assigned, from scratch, using the taxonomy defined in
`src/intents.py` and the escalation policy documented at the top of
`eval/golden_labels.py`. The weak label was *not* trusted as ground truth --
of the 210 rows, the human-assigned gold intent disagreed with the weak-label
seed bucket in several dozen cases (mostly `general_complaint_vague`
seeds that turned out, on reading, to actually be clean `content_catalog` or
`feature_request_feedback` messages -- expected, since that bucket is the
heuristic's catch-all default).

Two labels were assigned per example:
- `gold_intent`: one of the six taxonomy classes.
- `gold_should_escalate` (+ `gold_escalate_reason` when true): whether a
  human agent should handle this rather than the bot auto-posting a reply.
  This is a policy the author defined and applied consistently (see the
  docstring in `golden_labels.py`) -- it is **not** simply "did the real
  agent ask for a DM," because SpotifyCares asks for a DM on almost every
  account-related tweet regardless of severity. Escalation here means
  something narrower: security compromise, money that has already moved
  incorrectly, PII exposed in the public tweet, sustained anger/profanity
  signaling churn or PR risk, or messages too ambiguous to act on with
  confidence.

## Known limitations of this golden set (be skeptical of it)

- **One labeler, no independent second pass.** There is no true inter-rater
  reliability number for the *gold* labels themselves -- only for the LLM
  judge vs. a human on reply quality (see `eval/human_agreement_notes.md`).
  Boundary cases (e.g. "is a sarcastic thank-you `praise_smalltalk_closed`
  or `feature_request_feedback`?") were resolved by a documented rule of
  thumb, but a second labeler would likely disagree on 5-10% of the harder
  cases -- account_billing vs. general_complaint_vague and
  content_catalog vs. technical_playback (both often say "this song isn't
  available") are the two fuzziest boundaries.
- **Escalation labels encode one policy's risk appetite.** A more
  risk-averse brand would escalate more of the `account_billing` bucket by
  default; a more automation-forward one would escalate less. The 12%
  escalate rate is a property of the policy in `golden_labels.py`, not a
  law of nature -- change the policy and the "escalation precision/recall"
  numbers move accordingly (see report/REPORT.md, "what's misleading about
  my headline number").
- **Class imbalance mirrors the sampling design, not the true population.**
  Real-world traffic is much more `general_complaint_vague`-shaped/skewed
  than this golden set (which was stratified to 35/bucket). Accuracy
  numbers on this set should not be read as "accuracy on a random day of
  SpotifyCares' mentions" -- see the report for a reweighted estimate.
