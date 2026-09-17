# SpotifyCares AI Support Agent

An AI support agent for **SpotifyCares**, built from the official Kaggle
`thoughtvector/customer-support-on-twitter` dataset.

Given a customer message, the system:

1.  classifies intent into one of six handling-oriented classes,
2.  retrieves similar historical SpotifyCares customer/reply pairs,
3.  adapts a real historical reply into a grounded draft, and
4.  decides **auto-handle vs. human escalation** with a stated reason.

The default pipeline is local/offline: TF-IDF + Logistic Regression +
TF-IDF cosine retrieval + transparent escalation rules. An optional LLM
path uses the same retrieval index for grounding.

> **Scope note:** this README documents the project as implemented.
> Proposed improvements are explicitly labeled as next steps; they are
> not presented as implemented features.

------------------------------------------------------------------------

## Submission package

The repository contains the contents of the project handoff bundle. The
submission package is named **DATA**; it is the project/code bundle, not
the raw Kaggle dataset. The raw dataset remains outside version control
and is fetched separately using Step 1 below.

> **Important:** `data_raw/` and the regenerated `data/*.csv` /
> `data/*.pkl` artifacts are intentionally gitignored. The checked-in
> `eval/golden_set.csv` is the 210-example hand-labeled evaluation set
> used by the reported results.

------------------------------------------------------------------------

## 1. Problem framing: what "good" means for this brand

For a support brand, "good" is not simply maximum intent accuracy or
maximum automation.

The key asymmetry in this project is:

-   A false **auto-handle** on a security-compromise or financial-harm
    message can create real harm: account exposure, financial loss, or
    additional customer frustration.
-   A false **escalation** mainly costs a human agent a short review.

Therefore, the escalation gate should be evaluated with particular
attention to **recall**, not only F1 or precision.

The project consequently separates:

-   **intent classification** --- what kind of handling strategy the
    message appears to require;
-   **reply grounding** --- whether the draft is based on how
    SpotifyCares has actually replied to similar messages;
-   **escalation safety** --- whether the message should reach a human
    before automation.

### Intent taxonomy

The six classes were chosen around **distinct handling strategies**,
rather than simply creating many topical categories:

  -----------------------------------------------------------------------
  Intent                              Handling strategy
  ----------------------------------- -----------------------------------
  `account_billing`                   Account/payment issues; private
                                      verification is generally required

  `technical_playback`                Public troubleshooting first;
                                      device/OS/version details may be
                                      needed

  `content_catalog`                   Public informational response about
                                      availability/licensing/catalog

  `feature_request_feedback`          Acknowledge feedback or point
                                      toward the ideas/community
                                      mechanism

  `general_complaint_vague`           Insufficient information; ask a
                                      clarifying question

  `praise_smalltalk_closed`           No substantive action required
  -----------------------------------------------------------------------

The weak-label heuristic is used only to create **silver training data /
retrieval metadata**. The golden evaluation set is held out from
training.

### What I chose not to build

These are deliberate scope boundaries:

-   **Multi-turn conversation state:** the current agent is
    single-message/stateless. It does not reconstruct a customer's
    entire conversation or know whether they already supplied
    information in an earlier turn.
-   **Fine-grained sub-intents:** the six classes map to downstream
    handling strategies. Splitting them into dozens of narrower classes
    would fragment the training data without necessarily changing the
    action taken.
-   **PII redaction before external LLM calls:** identified as a real
    security gap and left for the next iteration rather than pretending
    it is solved.
-   **Hosted/cloud demo:** the target was a locally reproducible
    pipeline, not a cloud deployment with credentials, uptime and
    infrastructure dependencies.
-   **Fine-tuned embedding/generation model:** TF-IDF was deliberately
    retained for fast, deterministic, offline reproducibility.
-   **HNSW/ANN retrieval:** the current corpus is \~38K rows, so
    brute-force TF-IDF cosine retrieval was kept. HNSW is a scaling
    option if the corpus becomes much larger.

------------------------------------------------------------------------

## 2. System design

``` text
Customer message
      |
      v
Input validation / length cap
      |
      v
[1] Intent classifier
    TF-IDF (20K features, 1–2 grams)
    + Logistic Regression
      |
      v
[2] Historical retrieval
    TF-IDF cosine similarity
    over 37,923 customer/reply pairs
    + predicted-intent filtering
    + mild penalty for DM-only boilerplate
      |
      v
[3] Draft composer
    Strip agent signatures
    Remove some personalized greetings
    Add intent-specific generic opener
    Preserve substantive historical reply content
      |
      v
[4] Escalation policy
    Security / billing harm / public PII /
    sustained anger + unresolved issue / churn /
    low confidence / too-short vague messages
      |
      v
[5] Fail-safe behavior
    Pipeline errors -> escalate to human
```

The final output contains the intent, confidence, draft, retrieval
provenance/similarity, escalation decision, and stated reason.

### Why retrieval rather than free generation?

The central drafting choice is **reuse/adaptation of real historical
support replies** rather than unrestricted generation. This makes the
draft grounded in the brand's observed support behavior.

This is consistent with the broader retrieval-grounded dialogue
literature: Shuster et al. found that retrieval augmentation can reduce
hallucination in knowledge-grounded conversation. That paper is a
justification for the *design principle*, not evidence that this
specific system has eliminated hallucinations.

------------------------------------------------------------------------

## 3. Reproduction

### Step 1 --- obtain the raw dataset

The project uses the official Kaggle **Customer Support on Twitter**
dataset:

`thoughtvector/customer-support-on-twitter`

The raw dataset is **not included in the repository**. `data_raw/` is
gitignored because the dataset is large and is not redistributed with
this project. The expected local project path is:

``` text
data_raw/archive-2/twcs/twcs.csv
```

**Kaggle CLI:**

``` bash
pip install kaggle

mkdir -p data_raw
kaggle datasets download \
  -d thoughtvector/customer-support-on-twitter \
  -p data_raw \
  --unzip
```

After extraction, verify that the project can see:

``` bash
ls -lh data_raw/archive-2/twcs/twcs.csv
head -1 data_raw/archive-2/twcs/twcs.csv
```

The expected header is:

``` text
tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
```

If the Kaggle CLI extracts the CSV somewhere different on your machine,
place/copy the extracted `twcs.csv` at the project path above.

Official dataset page:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Expected schema:

``` text
tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
```

### Step 2 --- dependencies

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3 --- build the corpus, model and retrieval index

Run from the repository root:

``` bash
python src/prepare_data.py
python src/build_silver_corpus.py
python src/classifier.py
python src/retrieval.py
```

The original pipeline produced:

``` text
43,265 raw SpotifyCares tweets
43,092 paired
39,610 after English filter
37,923 final rows
```

### Step 4 --- reproduce the evaluation

``` bash
python eval/run_eval.py
```

This evaluates:

-   the trivial baseline,
-   the simple baseline,
-   the main agent.

It writes predictions and metrics under `eval/results/`.

### Optional LLM mode

The project also contains an optional LLM path requiring
`ANTHROPIC_API_KEY`:

``` bash
export ANTHROPIC_API_KEY=...
echo "some customer message" | python src/agent.py --mode llm
```

The local rule engine remains an escalation floor: the final decision is
effectively **local escalation OR LLM escalation**, never AND.

------------------------------------------------------------------------

# 4. Results vs. two baselines

The evaluation uses the checked-in **210-example golden set**
(`eval/golden_set.csv`). It is part of the project handoff and does not
need to be downloaded from Kaggle.

### Baselines

**Trivial baseline**

-   Always predicts the most common intent: `general_complaint_vague`
-   Uses one fixed generic reply
-   Never escalates

**Simple baseline**

-   Uses the same trained TF-IDF + Logistic Regression classifier as the
    main agent
-   Uses one fixed canned reply per intent
-   Escalates only when predicted intent is `account_billing`

The same classifier is intentionally shared by the Simple and Main
systems. This isolates the effect of **retrieval-grounded drafting +
richer escalation policy** rather than silently changing the classifier
between systems.

### Main results

  ------------------------------------------------------------------------------------------
  System         Intent      Intent   Escalation   Escalation   Escalation   Reply/reference
               accuracy    macro-F1    precision       recall           F1           ROUGE-1
  --------- ----------- ----------- ------------ ------------ ------------ -----------------
  Trivial          4.8%       0.015          ---         0.00        0.000             0.155

  Simple          67.6%       0.648        0.429         0.60        0.500             0.234

  **Main      **67.6%**   **0.648**    **0.917**     **0.44**    **0.595**         **0.858**
  agent**                                                                  
  ------------------------------------------------------------------------------------------

The identical intent metrics for Simple and Main are expected: both use
the same classifier. The difference comes after classification.

### Per-class signal

For the Main agent:

-   `account_billing`: precision 0.886, recall 0.674, F1 0.765
-   `content_catalog`: precision 0.795, recall 0.674, F1 0.729
-   `feature_request_feedback`: precision 0.694, recall 0.568, F1 0.625
-   `general_complaint_vague`: precision 0.250, recall 0.800, F1 0.381
-   `praise_smalltalk_closed`: precision 0.457, recall 0.889, F1 0.604
-   `technical_playback`: precision 0.939, recall 0.674, F1 0.785

The catch-all `general_complaint_vague` class is particularly important:
its 0.80 recall comes with only 0.25 precision.

### Reply-quality rubric

A random sample of 40 Main-agent drafts was evaluated using a rubric of:

-   relevance,
-   groundedness,
-   tone,
-   safety,
-   `auto_sendable`.

The checked-in scores are:

-   relevance: **4.20 / 5**
-   groundedness: **4.03 / 5**
-   tone: **4.00 / 5**
-   safety: **4.45 / 5**
-   auto-sendable: **65% (26/40)**

Important evaluation limitation: in the delivered environment there was
no outbound Anthropic API access, so these 40 scores were produced by
the author manually applying the exact rubric. They are **not**
independent live LLM-judge results.

The evaluation script is present for a live judge run when an API key is
available.

------------------------------------------------------------------------

# 5. Failure analysis: top 5 real failure modes

These were found by running the pipeline against the golden set.

## 1. False action claims

A retrieved historical reply can contain a statement such as:

> "We've just sent you a DM."

The current bot did not actually send that DM. Reusing the historical
wording can therefore create a false claim of action.

This is the most serious observed drafting failure because a reply can
look highly grounded while still being factually wrong about what the
current system did.

**Mitigation currently present:** DM-only historical replies receive a
mild retrieval penalty.

**Residual issue:** the penalty does not eliminate the pattern when
better substantive candidates are unavailable.

Example from the judge sample: customer ID **46** received a draft
claiming a DM had already been sent, even though no such action
occurred.

------------------------------------------------------------------------

## 2. Cross-contamination of identity / name leakage

One draft addressed a new customer as **"Stephen"**, because that name
appeared inside the retrieved historical reply.

The current greeting-cleaning regex handles names at the beginning of a
reply, but not every mid-sentence personalization pattern.

This is not a claim that the name is the customer's real identity; it is
a concrete example of historical text leaking into a new response.

Example: customer ID **162**.

------------------------------------------------------------------------

## 3. Silent truncation and thread artifacts from data cleaning

Two related data-cleaning problems were visible in generated replies:

-   URL stripping can leave dangling text such as **"add your email
    at"** or **"There's more info here:"** when the useful content was
    originally the removed URL.
-   Unstitched multi-tweet replies can leave artifacts such as **"1:"**
    or **"3:"** in the final draft.

Examples include IDs **108**, **169**, **207**, **15**, and **99** in
the judge sample.

The important lesson is that preprocessing can create output-quality
failures that no classifier metric will reveal.

------------------------------------------------------------------------

## 4. Classification errors cascade into tone mismatches

An upstream intent error changes the downstream opener and reply
strategy.

Example: customer ID **98** is a `technical_playback` case, but the
classifier routed it as `feature_request_feedback`.

The resulting draft began with a feature-request style opener:

> "Thanks for the suggestion..."

while the rest of the response was troubleshooting-oriented.

The failure is therefore not merely "one wrong class." It becomes a
visible **tone + action mismatch** in the final customer-facing reply.

------------------------------------------------------------------------

## 5. Escalation brittleness under paraphrase

The richer rule-based escalation policy is transparent and auditable,
but keyword/regex policies can miss semantically equivalent wording.

A particularly important measurement surprise was that the richer policy
initially had lower escalation recall than the naive `account_billing`
baseline. The baseline happened to cover many gold escalation cases
simply because many escalation-worthy examples belonged to the
account/billing class.

After generalizing the regex patterns and adding the short-message
trigger, the Main agent reached:

-   escalation precision: **0.917**
-   escalation recall: **0.44**
-   escalation F1: **0.595**

The remaining gap is real.

Examples include paraphrased financial-harm language such as:

-   "did the wrong payment help us out"
-   "avoid double charge"
-   "charged me ... after cancellation"

This is exactly where semantic classification is a possible future
improvement.

------------------------------------------------------------------------

# 6. "What is misleading about my headline number?"

## 67.6% intent accuracy is not the whole story

The golden set was intentionally stratified across six intent buckets.
It is therefore not a natural-traffic distribution.

The catch-all class has:

-   precision **0.25**
-   recall **0.80**

So a model can achieve a 67.6% overall accuracy while still frequently
assigning the catch-all class to messages that belong elsewhere.

The headline number should therefore be read together with macro-F1 and
per-class precision/recall.

## 0.858 ROUGE-1 is especially misleading

The Main agent's ROUGE-1 score is **0.858**, compared with 0.234 for the
Simple baseline.

But the Main agent is explicitly designed to adapt historical reply
text. High lexical overlap is therefore partly a consequence of the
architecture.

It means roughly:

> "The draft is lexically close to a real historical reply."

It does **not** prove:

> "The retrieved historical reply was the correct precedent for this
> customer."

The false-DM and name-leakage failures demonstrate why these are
different questions.

ROUGE-1 is consequently reported as a diagnostic, not as a standalone
reply-quality score.

## Escalation F1 hides the precision/recall trade-off

Main agent:

-   precision: **0.917**
-   recall: **0.44**

Simple baseline:

-   precision: **0.429**
-   recall: **0.60**

The blended F1 number does not tell the whole safety story. For this
project's stated risk framing, a missed escalation is more consequential
than an unnecessary human review, so recall needs to remain visible
rather than being collapsed into F1.

## The 4.x/5 judge scores are not independent validation

The 40-example rubric scores were manually produced by the author
because a live external LLM call was unavailable in the build
environment.

They should therefore be treated as a structured qualitative audit, not
as an independent benchmark.

## 11.9% is not "the percentage of SpotifyCares traffic that needs a human"

The golden set contains 25 escalation-positive examples out of 210, or
**11.9%**.

That is a property of this particular hand-labeled sample under this
particular escalation policy. It is not a measurement of SpotifyCares'
actual production escalation rate.

------------------------------------------------------------------------

# 7. Evaluation design and limitations

## Golden set

The golden set contains **210 examples**.

Sampling was:

-   generated from the full 37,923-row corpus using the weak-label
    heuristic,
-   stratified to 35 candidates per intent bucket,
-   spread across months within each bucket to reduce
    near-duplicate/outage-day concentration.

Each example was manually assigned:

-   `gold_intent`
-   `gold_should_escalate`
-   escalation reason

The labeling policy was defined around:

-   security compromise,
-   financial harm that already occurred,
-   publicly exposed PII,
-   sustained anger/profanity with a long-unresolved issue,
-   explicit churn after a long unresolved issue,
-   ambiguity / insufficient information where safe automation is not
    possible.

The project does **not** claim independent inter-rater reliability for
the gold labels: there was one labeler.

## LLM-as-judge context

The project includes an LLM-judge rubric and script. The delivered
`judge_scores.csv` is based on manual application of that rubric in the
no-network environment.

For external context, Zheng et al. (2023) found that strong LLM judges
can reach over 80% agreement with human preferences in their MT-Bench /
Chatbot Arena experiments, while also documenting judge biases and
limitations. That published result is a benchmark about LLM judging
generally, **not a measurement of this project's judge-human
agreement**.

------------------------------------------------------------------------

# 8. What I'd do next with one more week

### 1. Get genuine independent evaluation

Run the existing LLM-judge script against a live model and add a second
independent human labeler on at least a 50-example overlap.

Report:

-   intent agreement,
-   escalation agreement,
-   Cohen's kappa,
-   judge-human agreement bucketed by score gap.

This addresses the largest credibility limitation in the current
evaluation.

### 2. Add PII redaction before external calls

The current local pipeline does not send customer data outside the
machine.

The optional LLM path does.

A production-oriented next step is to add a PII-redaction pass
**before** any external LLM call, using email/phone detection at minimum
and a tool such as Microsoft Presidio for broader entity detection.

This is a proposed next step, not an implemented feature.

### 3. Evaluate on a second, unstratified sample

The current 210 examples are intentionally balanced by intent. A second
natural-distribution sample would make the reported accuracy more
representative of the actual corpus distribution.

### 4. Upgrade intent semantics with SetFit

SetFit is a natural candidate for improving semantic generalization
without requiring an LLM call per message.

For example, a semantic classifier could better connect paraphrases such
as:

-   "would be great if you added X"
-   "any chance we could get X"

The current bag-of-words TF-IDF representation does not inherently model
that semantic equivalence.

### 5. Fix the observed retrieval/data-cleaning bugs

Specifically:

-   catch mid-sentence name leakage,
-   add a sentence-completeness check for dangling URL references,
-   remove thread-number artifacts before a draft reaches the
    customer-facing layer,
-   add stronger safeguards around action claims such as "we've sent you
    a DM."

### 6. Add multi-turn state and idempotency

A real support integration needs:

-   thread/customer state,
-   processed-message IDs,
-   duplicate/retry handling,
-   awareness of information already supplied earlier in the
    conversation.

### 7. Scale retrieval only when scale requires it

At \~38K rows, brute-force TF-IDF retrieval is deliberately simple.

If the corpus grows to millions of messages, **HNSW** becomes relevant:
it represents items as nodes in a navigable proximity graph and searches
by moving through neighboring nodes rather than scanning every vector.

Product quantization is another possible scaling technique for
compressed approximate nearest-neighbor search.

These are scaling directions, not changes made to the current
implementation.

------------------------------------------------------------------------

# 9. A small but important inspiration: the Stanford Bunny and "search by navigation"

One of the unusual ideas that influenced the thinking around this
project came from the **Stanford Bunny**, the classic 3D scanned mesh
associated with Greg Turk and Marc Levoy's *Zippered Polygon Meshes from
Range Images*.

The Bunny itself has **nothing to do with text classification** and is
not used in the implementation.

The useful conceptual question it triggered was:

> If a complex object can be represented as connected local
> neighborhoods, can search also work by navigating from one nearby node
> to another instead of checking everything?

That question leads to a genuinely relevant text-search lineage:
**navigable small-world graphs** and **HNSW (Hierarchical Navigable
Small World)**.

HNSW represents vectors as nodes connected to nearby vectors and
performs approximate nearest-neighbor search by navigating through the
graph. Its hierarchy provides coarse-to-fine navigation across distance
scales.

So the conceptual chain for this project is:

``` text
Stanford Bunny / mesh topology
        |
        | inspiration: connected neighborhoods
        v
"Can search navigate between nearby nodes?"
        |
        v
Navigable Small-World graph idea
        |
        v
HNSW
        |
        v
Future ANN retrieval option for a much larger
SpotifyCares corpus
```

The distinction matters:

-   **Bunny mesh:** physical 3D geometry and surface reconstruction.
-   **HNSW:** vector-space nearest-neighbor search.
-   **This project:** text support messages represented with TF-IDF
    vectors.

The Bunny is therefore an **inspiration for the node-to-node navigation
intuition**, not a technical component of the current support agent.

------------------------------------------------------------------------

# 10. Decision log --- non-obvious decisions

1.  **Kept the silver training corpus separate from the golden
    evaluation set.**\
    The classifier never trains on golden examples, preventing
    evaluation leakage.

2.  **Defined intents by handling strategy rather than topic.**\
    Six classes were chosen because they correspond to meaningfully
    different downstream actions.

3.  **Defined escalation independently from historical agent
    behavior.**\
    "Should escalate" was not defined as "did the historical agent ask
    for a DM," because that would make the label too dependent on
    routine account-related DM behavior.

4.  **Used a transparent rule engine for escalation.**\
    Auditability and explicit reasons were prioritized over treating
    safety as an opaque learned score.

5.  **Made the system fail closed.**\
    Unhandled pipeline errors default to `should_escalate=True`.

6.  **Used OR composition between local escalation rules and optional
    LLM escalation.**\
    An LLM can add an escalation but cannot suppress a local rule that
    already requires human review.

7.  **Stored retrieval provenance with every draft.**\
    The draft records which historical example grounded it and the
    similarity score, enabling auditing.

8.  **Generated a draft even for escalated messages.**\
    The human agent gets a suggested starting point rather than a blank
    compose box; the draft is not treated as automatically sendable.

9.  **Penalized DM-only boilerplate instead of deleting it.**\
    Such replies are legitimate historical behavior for some account
    cases, so a soft penalty preserves useful signal.

10. **Used the same classifier for the Simple baseline and Main
    agent.**\
    This isolates retrieval/drafting and escalation effects.

11. **Generalized escalation regexes rather than copying missed
    golden-set phrases verbatim.**\
    This was intended to avoid improving the reported metric through
    direct phrase memorization.

12. **Stratified golden sampling by month within intent buckets.**\
    This reduces the chance that a viral event or outage day dominates
    an intent bucket.

13. **Reported ROUGE while explicitly calling out its limitations.**\
    Since retrieval reuse naturally increases lexical overlap, the
    metric is useful diagnostically but misleading as a standalone
    quality measure.

14. **Added the short-message escalation trigger after observing the
    failure.**\
    Very short generic messages could receive high classifier confidence
    while still lacking enough information to act safely.

15. **Did not add HNSW just because it is a production technique.**\
    The current corpus is only \~38K rows, so brute-force TF-IDF
    retrieval keeps the project simpler and reproducible. HNSW is
    reserved for a real scaling need.

------------------------------------------------------------------------

# 11. Security and privacy boundary

The default local mode keeps customer text inside the local pipeline.

The optional LLM mode introduces a different boundary: customer text can
be sent to an external API.

The current project explicitly identifies this as a gap:

> **PII redaction before an external LLM call is not implemented yet.**

A production version should therefore redact or otherwise protect PII
before sending customer text to an external model.

Microsoft Presidio is one possible open-source direction for PII
detection/anonymization, but it is **not currently integrated into this
project**.

------------------------------------------------------------------------

# 12. Repository structure

``` text
data_raw/              raw Kaggle data (not committed)
data/                  regenerated corpus/model/index (not committed)
src/
  prepare_data.py
  build_silver_corpus.py
  intents.py
  classifier.py
  retrieval.py
  draft.py
  escalation.py
  agent.py
  llm_agent.py
  llm_client.py

eval/
  golden_candidates.csv
  golden_labels.py
  make_golden_set.py
  golden_set.csv
  baselines.py
  run_eval.py
  llm_judge.py
  results/

report/
  REPORT.md

tests/
.github/
MANIFEST.md
README.md
requirements.txt
```

------------------------------------------------------------------------

# 13. Research and citations

The following references informed design decisions or future-work
directions. They are **research grounding**, not claims that every cited
method is implemented in the current code.

### Retrieval-grounded dialogue

1.  **Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J.
    (2021).**\
    *Retrieval Augmentation Reduces Hallucination in Conversation.*
    Findings of EMNLP 2021, 3784--3803.\
    https://aclanthology.org/2021.findings-emnlp.320/\
    Used to ground the rationale for retrieval-backed conversational
    responses and hallucination mitigation.

### Approximate nearest-neighbor search / navigation

2.  **Malkov, Y. A. & Yashunin, D. A. (2018/2020).**\
    *Efficient and Robust Approximate Nearest Neighbor Search Using
    Hierarchical Navigable Small World Graphs.* IEEE TPAMI, 42(4),
    824--836.\
    https://doi.org/10.1109/TPAMI.2018.2889473\
    Relevant to the proposed future HNSW retrieval path and the
    node-to-node navigation idea.

3.  **Kleinberg, J. (2000).**\
    *The Small-World Phenomenon: An Algorithmic Perspective.*
    Proceedings of STOC 2000.\
    Relevant conceptual lineage for navigable small-world graphs and
    decentralized node-to-node search.

### Vector compression

4.  **Jégou, H., Douze, M., & Schmid, C. (2011).**\
    *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI,
    33(1), 117--128.\
    https://doi.org/10.1109/TPAMI.2010.57\
    Relevant to future memory-efficient approximate nearest-neighbor
    retrieval; not implemented in the current project.

### Semantic classification

5.  **Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D.,
    Wasserblat, M., & Pereg, O. (2022).**\
    *Efficient Few-Shot Learning Without Prompts.*\
    https://arxiv.org/abs/2209.11055\
    Relevant to the proposed SetFit upgrade for semantic intent
    generalization; not implemented in the current project.

### PII protection

6.  **Microsoft Presidio.**\
    Open-source framework for identifying and anonymizing sensitive/PII
    data in text.\
    https://microsoft.github.io/presidio/\
    Considered as a future redaction layer before external LLM calls;
    not integrated into the current code.

### LLM evaluation

7.  **Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023).**\
    *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.*
    EMNLP 2023, 2511--2522.\
    https://aclanthology.org/2023.emnlp-main.153/\
    Relevant to rubric-based LLM evaluation and the use of structured
    criteria for judging generated text.

8.  **Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023).**\
    *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*\
    https://arxiv.org/abs/2306.05685\
    Used as external context for discussing LLM-as-a-judge and its
    agreement/limitations. The published benchmark is not a result from
    this project.

### Stanford Bunny / 3D mesh inspiration

9.  **Turk, G. & Levoy, M. (1994).**\
    *Zippered Polygon Meshes from Range Images.* SIGGRAPH '94,
    311--318.\
    https://doi.org/10.1145/192161.192241\
    This is the source of the Stanford Bunny / mesh-processing
    connection discussed in §9. The project does not use a 3D mesh
    model; the connection is conceptual inspiration for thinking about
    connected neighborhoods and navigation.

------------------------------------------------------------------------

## Final takeaway

The project deliberately optimizes for a different objective than "make
a chatbot that sounds good."

It treats support automation as a combination of:

**classification + retrieval grounding + explicit safety policy + human
escalation + auditability.**

The headline 67.6% intent accuracy is only one part of that story. The
more important engineering findings are the observed failure modes:
**false action claims, identity leakage, preprocessing artifacts,
classifier-to-tone cascades, and escalation brittleness under
paraphrase**.

The current implementation stays intentionally simple and reproducible.
The next technical frontier is not adding complexity for its own sake;
it is improving **semantic generalization, PII boundaries, independent
evaluation, retrieval correctness, and escalation recall**, while
preserving the auditability of the current system.
