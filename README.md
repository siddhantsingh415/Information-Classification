# Information-Classification
A Git Repo to track a misinformation detection project

# models4 — Structured Argumentation Graph for Fake News Detection

A Graph Attention Network (GAT) pipeline that detects fake news by constructing
typed argumentation graphs from articles and comparing their analysis and claim
layers against independently retrieved cross-source coverage of the same event.

---

## Table of Contents

1. [Motivation and Core Insight](#1-motivation-and-core-insight)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Step-by-Step Architecture](#3-step-by-step-architecture)
   - [3.1 Dataset Loading](#31-dataset-loading)
   - [3.2 Sentence Role Classification](#32-sentence-role-classification)
   - [3.3 Headline Search](#33-headline-search)
   - [3.4 Source Credibility Database](#34-source-credibility-database)
   - [3.5 Structured Argumentation Graph](#35-structured-argumentation-graph)
   - [3.6 Node Feature Vectors](#36-node-feature-vectors)
   - [3.7 Graph Attention Network](#37-graph-attention-network)
   - [3.8 Batched Graph Construction](#38-batched-graph-construction)
   - [3.9 Evidence Retry](#39-evidence-retry)
   - [3.10 Warm-Start Fine-Tuning](#310-warm-start-fine-tuning)
   - [3.11 Batched Training Loop](#311-batched-training-loop)
4. [Data Flow Diagram](#4-data-flow-diagram)
5. [Design Decisions and Trade-offs](#5-design-decisions-and-trade-offs)
6. [Known Limitations](#6-known-limitations)
7. [Improvement Proposals](#7-improvement-proposals)
   - [7.1 GATv2Conv — High Impact, Trivial Change](#71-gatv2conv--high-impact-trivial-change)
   - [7.2 Edge Features in Attention — Medium Impact](#72-edge-features-in-attention--medium-impact)
   - [7.3 Richer Sentence Encoder — Medium Impact](#73-richer-sentence-encoder--medium-impact)
   - [7.4 Z-Score Normalisation — Low-Medium Impact](#74-z-score-normalisation--low-medium-impact)
   - [7.5 Heterogeneous Graph Formulation — High Impact, More Work](#75-heterogeneous-graph-formulation--high-impact-more-work)
   - [7.6 Role-Prediction Auxiliary Loss — Low-Medium Impact](#76-role-prediction-auxiliary-loss--low-medium-impact)
   - [7.7 Larger Search Index — Medium Impact](#77-larger-search-index--medium-impact)
   - [7.8 Temporal Edge Weighting — Low Impact](#78-temporal-edge-weighting--low-impact)
8. [File and Directory Layout](#8-file-and-directory-layout)
9. [Quickstart](#9-quickstart)

---

## 1. Motivation and Core Insight

Standard text classifiers treat each article as an independent bag of tokens.
This works when the signal is stylistic (writing patterns, vocabulary), but
falls apart against sophisticated misinformation that is stylistically
indistinguishable from real journalism.

The core bet of this pipeline is that **misinformation is detectable at the
relational level**: fake articles tend to make analytical claims (interpretations,
conclusions, framings) that contradict or are unsupported by what other
independent sources say about the same event. Real articles tend to have their
analysis corroborated.

Making that bet computable requires three things:

1. A way to distinguish *what an article claims* from *how it interprets* those
   claims — the role classification layer.
2. A way to find other sources covering the same event — headline search.
3. A structured representation that lets a model reason over the relationships
   between all of those sentences across sources — the argumentation graph.

---

## 2. Pipeline Overview

```
Raw article (title + body)
        │
        ▼
┌───────────────────┐
│  Sentence         │  sent_tokenize → NLI role classifier
│  Role Labelling   │  → claim / evidence / analysis / background
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Headline Search  │  DuckDuckGo (headline query, k=10)
│  + Doc Scoring    │  → DBSCAN consensus × domain credibility
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Graph Building   │  intra-article NLI edges
│                   │  + cross-source role-matched NLI edges
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Node Features    │  20-dim hand-crafted feature vector per node
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  GAT Classifier   │  4-layer GAT + JK aggregation + dual pooling
│                   │  → binary logit (real / fake)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Credibility DB   │  Bayesian update per domain using model confidence
│  Update           │  → feeds back into next batch's doc scoring
└───────────────────┘
```

---

## 3. Step-by-Step Architecture

### 3.1 Dataset Loading

**Source:** ISOT Fake News Dataset (`Fake.csv`, `True.csv`).

The two files are concatenated, shuffled, and binary-labelled (`fake=1`, `real=0`).
One preprocessing step is critical: real articles from Reuters carry a dateline
prefix (`WASHINGTON (Reuters) —`) that is a near-perfect label leak. It is stripped
with a regex before any further processing.

**Why shuffle?** The two CSV files are temporally ordered within their respective
sources. Training on a sorted concatenation would let the model exploit temporal
drift instead of semantic content.

---

### 3.2 Sentence Role Classification

Each article body is tokenised into sentences with NLTK's `sent_tokenize`. Every
sentence is then classified into one of four rhetorical roles using **zero-shot NLI**:

| Role | Definition |
|------|-----------|
| `claim` | A specific factual assertion |
| `evidence` | Cited data, quotes, or supporting facts |
| `analysis` | Interpretation, opinion, or conclusion drawn from facts |
| `background` | Contextual or general information |

Classification works by running the NLI model (`cross-encoder/nli-deberta-v3-small`)
against four defining hypothesis strings and taking the role whose hypothesis scores
highest on entailment probability. All sentences × all hypotheses are batched into a
single `predict()` call for efficiency.

**Why zero-shot NLI rather than a fine-tuned classifier?** A fine-tuned classifier
needs a role-labelled news corpus that doesn't readily exist, and would risk
overfitting to surface patterns of specific publications. Zero-shot NLI generalises
across topics without labelled role data.

**Why this role distinction matters:** Misinformation rarely fabricates raw facts
outright. It manipulates the *analysis layer* — presenting selective evidence,
drawing unsupported conclusions, or framing neutral facts with loaded interpretation.
Labelling that layer explicitly gives the GAT the right signal to learn from.

---

### 3.3 Headline Search

Rather than extracting entity queries from the article body, the pipeline searches
using the article's **headline** directly via `duckduckgo_search`.

**Why headlines?**
- Headlines are written to be findable — they are the journalist's best distillation
  of the article's core claim.
- A headline search retrieves articles covering the *same event* from different
  publishers, which is the cross-source comparison the model needs.
- Entity queries from body text match individual entities that appear in completely
  unrelated contexts (the models2 failure mode: emojipedia and dndbeyond appearing
  as evidence for political articles).
- This reduces search calls from O(n\_claims) per article to O(1).

Publication suffixes (`| Reuters`, `- BBC`) are stripped from headlines before
searching because they bias results back toward the same publisher.

Results are cached on disk by MD5 hash of the query string to avoid redundant API
calls across runs.

---

### 3.4 Source Credibility Database

A SQLite database maintains a per-domain **Bayesian credibility score** using a
Beta distribution:

- Every domain starts with a Beta(2, 2) prior — a score of 0.5, meaning no
  information either way.
- The score at any point is `alpha / (alpha + beta)`, the mean of the Beta
  distribution.
- **Model updates** increment `alpha` (real signal) or `beta` (fake signal) by
  `model_weight × model_confidence × document_relevance_score` (weight = 0.3).
- **User updates** use weight 1.0 so that explicit human labels dominate model
  inference.

This score is blended with the DBSCAN consensus score when ranking retrieved
documents, so high-credibility sources get proportionally more edge weight in
the graph.

**Why Bayesian?** The Beta-Bernoulli model naturally handles cold-start (new domains
start uncertain, not zero), accumulates evidence gracefully, and produces calibrated
probability estimates rather than raw counts.

---

### 3.5 Structured Argumentation Graph

Each article becomes a `networkx.Graph` with two kinds of nodes and three kinds of
edges.

**Node types:**

| Type | Source | Description |
|------|--------|-------------|
| `input` | Target article | One node per sentence, labelled with its role |
| `evidence` | Retrieved articles | One node per external sentence that passes the role-match and similarity threshold |

**Edge types:**

| Type | Connects | Condition |
|------|----------|-----------|
| Intra-article | `input ↔ input` | Cosine similarity > 0.60, then NLI-typed |
| Cross-source analysis | `input_analysis ↔ ext_analysis` | Cosine similarity > 0.55, then NLI-typed |
| Cross-source claim | `input_claim ↔ ext_evidence/claim` | Cosine similarity > 0.55, then NLI-typed |

**Why role-matched cross-source edges only?** Comparing a background sentence from
the target article to an analysis sentence from a retrieved article produces
meaningless NLI scores — their semantic roles are incompatible. Restricting edges
to role-matched pairs keeps the cross-source signal clean.

Each edge carries: cosine similarity, NLI edge type (entailment / contradiction /
neutral), NLI confidence score, document relevance weight, and a cross-source flag.

**NLI edge typing:** The NLI model produces a 3-way probability distribution over
contradiction, entailment, and neutral. The argmax determines the edge type; the
winning probability is stored as confidence. This means a single cross-source edge
encodes not just *that* two sentences are semantically related but *how* — whether
one corroborates or disputes the other.

---

### 3.6 Node Feature Vectors

Each node is described by a **20-dimensional hand-crafted feature vector**, normalised
column-wise by the maximum value.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `n_evidence_nbrs` | Count of external evidence neighbours |
| 1 | `n_input_nbrs` | Count of intra-article neighbours |
| 2 | `mean_ev_weight` | Mean document relevance score of evidence neighbours |
| 3 | `max_ev_weight` | Max document relevance score of evidence neighbours |
| 4 | `mean_sim` | Mean edge cosine similarity |
| 5 | `ev_ratio` | Fraction of neighbours that are external |
| 6 | `is_evidence` | 1 if this node is from a retrieved source |
| 7 | `node_weight` | Document credibility / relevance score |
| 8 | `n_entailing` | Count of entailment edges |
| 9 | `n_contradicting` | Count of contradiction edges |
| 10 | `ent_wsum` | Weighted entailment sum (weight × confidence) |
| 11 | `cont_wsum` | Weighted contradiction sum |
| 12 | `ent_ratio` | Entailment fraction of evidence neighbours |
| 13 | `cont_ratio` | Contradiction fraction of evidence neighbours |
| 14–17 | `role_onehot` | One-hot encoding of role (claim / evidence / analysis / background) |
| 18 | `cross_ent_w` | Cross-source entailment weight (corroboration signal) |
| 19 | `cross_cont_w` | Cross-source contradiction weight (dispute signal) |

Features 14–19 are the additions in models3/4 over models2. Features 18 and 19 are
the most directly diagnostic: a high `cross_ent_w` on an analysis node means that
node's interpretation is corroborated by multiple independent sources; a high
`cross_cont_w` means it is disputed.

---

### 3.7 Graph Attention Network

The classifier is a 4-layer **Graph Attention Network** implemented in PyTorch
Geometric.

**Architecture:**

```
Input (20-dim node features)
        │
        ▼
Linear input projection → hidden_dim (64)       # decouple feature scale from hidden dim
        │
        ▼
GATConv layer 1   (64 → 64,  4 heads, concat)   + residual
        │
        ▼
GATConv layer 2   (64 → 128, 4 heads, concat)   no residual (dim changes)
        │
        ▼
GATConv layer 3   (128 → 128, 4 heads, concat)  + residual
        │
        ▼
GATConv layer 4   (128 → 64, 4 heads, mean)     + residual (truncated)
        │
        ▼
JK concatenation: [x1 ‖ x2 ‖ x3 ‖ x4]          # Jumping Knowledge
        │
        ▼
global_mean_pool ‖ global_max_pool               # dual graph-level summary
        │
        ▼
MLP: 832 → 256 → 64 → 1                          # with BatchNorm + Dropout
        │
        ▼
BCEWithLogitsLoss
```

**Key design choices and why:**

**Input projection:** A dedicated linear layer maps 20-dim features into hidden space
before any message passing. This prevents the graph convolutions from having to
simultaneously rescale features and learn relational patterns.

**Residual connections:** Added wherever the hidden dimension stays constant (layers
1, 3, 4). Layer 2 changes dimension, so no residual is possible there. Residuals
prevent vanishing gradients and let the model skip layers that add no useful signal.

**Jumping Knowledge (JK) aggregation:** All four layer outputs are concatenated before
pooling. Layer 1 captures immediate neighbourhoods (are my direct neighbours
corroborating me?); layer 4 captures longer-range structure (does the whole graph
pattern look like a fake article?). Concatenating all four lets the readout draw on
whichever scale is most informative.

**Dual pooling:** Mean pool + max pool concatenated. Mean captures average node
behaviour across the graph; max captures the most extreme signal anywhere (e.g. a
single highly-contradicted analysis node may be the clearest fake signal).

**Training:** Adam with cosine annealing LR schedule, gradient clipping at 1.0, and
early stopping on validation loss with patience 20 (cold start) / 8 (fine-tuning).

---

### 3.8 Batched Graph Construction

`build_batch` replaces the monolithic `build_dataset` with a function that operates
on a single slice of the dataframe and returns:

- `pyg_data` — PyG graphs for articles that successfully retrieved evidence
- `all_scored` — aligned scored document lists
- `failed_items` — `FailedItem` dataclass instances for articles with zero evidence

The `FailedItem` dataclass preserves the pre-augmentation graph state (base graph,
sentence list, role list), meaning the expensive NLI role-classification pass does
not need to be repeated during retry. Only the cheaper retrieval and augmentation
steps are re-run.

**Why separate failed items rather than just dropping them?** Silently dropping
failed articles introduces a systematic bias — articles whose headlines happen to be
search-unfriendly are excluded from training. In models2 this was 262/500 articles.
Preserving and retrying them keeps the dataset representative.

---

### 3.9 Evidence Retry

`retry_failed_evidence` runs up to three progressively broader fallback search
strategies on each `FailedItem`, stopping as soon as one produces evidence:

**Strategy 1 — Truncated headline:** First 6 words of the cleaned headline.
Rare proper nouns in long or complex headlines often cause DuckDuckGo to return
unrelated pages. Truncating focuses the query on the most common terms.

**Strategy 2 — NER keyword query:** spaCy extracts named entities (PERSON, ORG,
GPE, EVENT…) and noun chunk roots from the title + first 300 characters of the
body. This tends to work when the headline is too vague to search directly.

**Strategy 3 — Lead sentence:** First substantive sentence of the article body,
capped at 120 characters. News articles follow the inverted pyramid structure,
so the lead sentence typically contains the most searchable concrete claim.

Articles that fail all three strategies are included in the dataset as **base-graph-only**
(intra-article edges only, no cross-source edges). This keeps the dataset balanced
and avoids the systematic exclusion problem, at the cost of weaker signal for those
articles.

---

### 3.10 Warm-Start Fine-Tuning

`finetune_gat` fine-tunes an existing model on new batch data rather than
re-initialising. It differs from `train_gat` in three ways:

1. **No re-initialisation** — continues from current weights.
2. **Lower LR** — defaults to 1e-4 vs 5e-4 for cold start, to avoid catastrophic
   forgetting of patterns learned in earlier batches.
3. **Fresh cosine-annealing restart** — `T_max` is set to the fine-tuning epoch
   count, giving a full cosine curve per batch rather than continuing a stale
   schedule from the cold start.

Early stopping patience is tighter (8 vs 20) because the model is already partially
trained and should converge quickly on new data.

---

### 3.11 Batched Training Loop

`batched_train_loop` orchestrates the full iterative pipeline:

```
Before batching:
  ├── Reserve a fixed global validation pool (never touched during training)
  └── Build val pool graphs once

For each batch i:
  ├── build_batch()              → graphs + failed_items
  ├── retry_failed_evidence()    → recover failed articles
  ├── split batch → batch-local train/val (for early stopping only)
  ├── if i == 0:  train_gat()     (cold start, more epochs)
  │   else:       finetune_gat()  (warm start, lower LR)
  ├── evaluate on fixed global val pool → record accuracy/F1/AUC
  ├── update credibility DB with this batch's predictions
  └── save checkpoint  fake_news_gat_v4_batch{i:03d}.pt

After all batches:
  └── plot learning curve (metrics + dataset growth per batch)
```

**Why a fixed global val pool?** If the val set changes each batch, accuracy numbers
are not comparable across batches — you cannot tell whether an improvement is real or
just caused by an easier val split. A fixed pool makes all metrics on the same scale.

**Why separate batch-local val from global val?** The batch-local val is used only
for early stopping during fine-tuning — it needs to be in-distribution with the
batch being trained on. The global val pool measures generalisation across the full
distribution.

**Compounding credibility DB:** Because the DB is updated after every batch, later
batches score retrieved documents with richer accumulated signal. The model's evidence
quality improves as it trains, which is the compounding feedback loop this design is
built around.

---

## 4. Data Flow Diagram

```
df (shuffled ISOT)
│
├─ val pool (fixed, ~60 articles) ──────────────────────────────────┐
│                                                                    │
└─ work_df                                                           │
    │                                                                │
    ├─ Batch 1 ──► build_batch ──► retry ──► train_gat (cold) ──► evaluate ──► DB update ──► checkpoint
    │                                                                │
    ├─ Batch 2 ──► build_batch ──► retry ──► finetune   ──────────► evaluate ──► DB update ──► checkpoint
    │                                                                │
    ├─ Batch 3 ──► build_batch ──► retry ──► finetune   ──────────► evaluate ──► DB update ──► checkpoint
    │                                                                │
    └─ ...                                                           │
                                                          batch_history (metrics per batch)
                                                          learning_curve.png
```

---

## 5. Design Decisions and Trade-offs

| Decision | Why | Trade-off |
|----------|-----|-----------|
| Headline search instead of entity queries | Retrieves same-event coverage, O(1) per article | Headlines for very old or niche articles may not return results |
| Zero-shot NLI for role classification | No labelled role corpus needed, generalises across topics | Slower than a fine-tuned classifier; hypotheses are hand-crafted |
| Role-matched cross-source edges only | Prevents noisy cross-role NLI comparisons | May miss some genuine signal between different role types |
| DBSCAN for document consensus scoring | Finds the majority narrative without assuming a fixed cluster count | Sensitive to `eps` hyperparameter; min_samples=2 can produce sparse clusters on small result sets |
| Bayesian domain credibility | Handles cold-start gracefully, calibrated uncertainty | Slow to update; requires many predictions before scores diverge meaningfully from 0.5 |
| Hand-crafted 20-dim features + column-max normalisation | Interpretable, no learned embedding required | Sensitive to outliers; misses distributional structure that learned features would capture |
| Batched training with warm starts | Faster feedback, compounding DB | Each batch is small; early batches may produce noisy gradient updates |
| Base-graph-only fallback for retry failures | Keeps dataset balanced | Base-only graphs are weaker signal and may confuse the model if they are a large fraction of the dataset |

---

## 6. Known Limitations

**Search reliability.** DuckDuckGo results vary by query and time of day. The caching
layer mitigates this for repeated runs but means stale cached results can persist.
For reproducible experiments, a static snapshot of retrieved documents is preferable.

**Snippet-only evidence.** Retrieved documents provide only the search result snippet
(~2 sentences), not the full article text. This limits the depth of cross-source
analysis and means very long or nuanced arguments in retrieved articles are not
captured.

**NLI model confidence.** `nli-deberta-v3-small` is reasonably capable but will
misclassify metaphor, sarcasm, and domain-specific language. Errors in role
classification propagate into graph structure and are not corrected later.

**Column-max normalisation.** A single outlier node with an extreme feature value
compresses all other nodes toward zero for that feature. This is the most likely
cause of sluggish convergence in early training.

**Small batch instability.** With `batch_size=50` and a `val_split=0.2`, the
batch-local training set is ~40 articles. BatchNorm layers can behave erratically
on batches this small, especially in early training before the model has seen
diverse graph structures.

---

## 7. Improvement Proposals

### 7.1 GATv2Conv — High Impact, Trivial Change

Standard `GATConv` computes attention weights before combining query and key
representations, making it equivalent to static attention in some graph structures.
`GATv2Conv` applies the non-linearity *after* concatenation, making attention
genuinely input-dependent. The API is identical — swap the import and class name,
no other changes required.

```python
from torch_geometric.nn import GATv2Conv

self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=n_heads,
                       concat=True, dropout=dropout)
```

### 7.2 Edge Features in Attention — Medium Impact

The graph carries rich edge attributes (NLI type, confidence, similarity,
cross-source flag) but they are currently ignored during message passing.
`GATv2Conv` accepts an `edge_dim` parameter that folds edge attributes into the
attention computation.

```python
EDGE_DIM = 7  # matches edge_attr dimension in graph_to_pyg()

self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=n_heads,
                       concat=True, dropout=dropout, edge_dim=EDGE_DIM)

# In forward(), pass edge attributes:
x1 = self.bn1(F.relu(self.lin1(
    self.conv1(x, edge_index, edge_attr=edge_attr)
))) + x
```

You also need to add `edge_attr` as a parameter to `forward()` and pass `b.edge_attr`
in the training loop. This is likely the single change with the best impact-to-effort
ratio after GATv2.

### 7.3 Richer Sentence Encoder — Medium Impact

`all-MiniLM-L6-v2` is fast but relatively weak for semantic nuance. The similarity
thresholds used for edge construction (0.55–0.60) were tuned for it. Upgrading
to `all-mpnet-base-v2` or `BAAI/bge-small-en-v1.5` would produce more accurate
embeddings for both edge construction and NLI role classification, at the cost of
~4–5× slower encoding.

```python
ENCODER = SentenceTransformer('BAAI/bge-small-en-v1.5')
```

Re-tune the similarity thresholds after switching encoders — the same threshold
values will not produce equivalent graph densities.

### 7.4 Z-Score Normalisation — Low-Medium Impact

The current column-max normalisation is sensitive to outliers and does not centre
the features. StandardScaler addresses both:

```python
from sklearn.preprocessing import StandardScaler

all_x  = torch.cat([d.x for d in pyg_data], dim=0).numpy()
scaler = StandardScaler().fit(all_x)

for d in pyg_data:
    d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype=torch.float)
```

Save the scaler alongside the model weights so inference-time graphs are normalised
consistently. In the batched loop, fit the scaler on the first batch and apply it
to all subsequent batches without re-fitting.

### 7.5 Heterogeneous Graph Formulation — High Impact, More Work

Currently, `input` and `evidence` nodes share the same embedding space and message
passing weights — only feature 6 (`is_evidence`) distinguishes them. PyG's
`HeteroData` assigns separate weight matrices to different node and edge types,
which is a more principled representation:

```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['input'].x    = input_features      # (n_input, feat_dim)
data['evidence'].x = evidence_features   # (n_evidence, feat_dim)

data['input',  'similar_to',     'input'].edge_index    = intra_edges
data['input',  'supported_by',   'evidence'].edge_index = entail_edges
data['input',  'contradicted_by','evidence'].edge_index = contra_edges
```

This is the largest refactor on the list but addresses a structural limitation of
the current design: the model has no mechanism to learn that `input→evidence`
entailment edges are fundamentally different from `input→input` similarity edges.

### 7.6 Role-Prediction Auxiliary Loss — Low-Medium Impact

If a small set of ground-truth role labels is available (or if `classify_sentence_roles`
is used as a noisy teacher to generate pseudo-labels), a node-level auxiliary loss
can be added to force the model to learn role-aware representations:

```python
# Additional head in FakeNewsGAT.__init__:
self.role_head = torch.nn.Linear(jk_dim, 4)

# In forward(), before pooling:
role_logits = self.role_head(xjk)   # (n_nodes, 4)

# In training loop:
loss = bce_loss(graph_logit, label) + 0.1 * ce_loss(role_logits, node_roles)
```

Multi-task learning often improves the primary task even with noisy auxiliary
supervision, because it regularises the learned representations.

### 7.7 Larger Search Index — Medium Impact

DuckDuckGo is convenient but limited to ~10 results per query. More results mean
more potential evidence nodes and a more robust DBSCAN consensus score. Replacing
or augmenting with the Google Custom Search API, Bing Web Search API, or a local
Elasticsearch index over a news corpus (e.g. CC-News or RealNews) would
substantially improve evidence quality and search reliability.

### 7.8 Temporal Edge Weighting — Low Impact

Articles retrieved as evidence may have been published before or after the target
article. Evidence published long after the target article cannot logically support
or refute it. If publication dates are available, edge weights could be attenuated
by temporal distance:

```python
days_diff   = abs((target_date - evidence_date).days)
time_weight = math.exp(-days_diff / 30)   # half-weight at 30 days
edge_weight = base_weight * time_weight
```

This adds a weak inductive bias that may improve precision on time-sensitive news
events.

---

## 8. File and Directory Layout

```
project/
├── data/
│   ├── Fake.csv                        # ISOT fake news articles
│   └── True.csv                        # ISOT real news articles
│
├── search_cache_v3/                    # DuckDuckGo result cache (MD5-keyed JSON)
│   └── <md5>.json
│
├── source_credibility_v3.db            # SQLite Bayesian credibility database
│
├── fake_news_gat_v4_batch001.pt        # Per-batch model checkpoints
├── fake_news_gat_v4_batch002.pt
├── ...
│
├── fake_news_gat_v4_learning_curve.png # Per-batch accuracy/F1/AUC + dataset growth
│
└── models4.ipynb                       # This notebook
```

---

## 9. Quickstart

**Requirements:** `torch`, `torch_geometric`, `sentence_transformers`,
`duckduckgo_search`, `spacy`, `nltk`, `networkx`, `scikit-learn`, `pandas`,
`matplotlib`, `tqdm`.

```bash
python -m spacy download en_core_web_sm
```

Place `Fake.csv` and `True.csv` in a `data/` folder, then run all cells in order
through Step 7 to define all functions. Then run:

```python
# Batched training (recommended)
final_model, history = batched_train_loop(
    df,
    batch_size    = 50,
    val_pool_size = 60,
    cold_epochs   = 80,
    warm_epochs   = 30,
    retry         = True,
)

# Single-pass training (original approach, for comparison)
pyg_dataset, all_scored_docs = build_dataset(df, sample=500)
model = train_gat(train_data, val_data, epochs=150, patience=20)
```

**Resuming after a kernel restart:**

```python
model = load_model('fake_news_gat_v4_batch003.pt')
# Slice work_df from row batch_size * 3 onward and continue
```
