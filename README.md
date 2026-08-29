# Information-Classification

A self-directed research project exploring graph-based misinformation detection using natural language processing, information retrieval, and Graph Neural Networks.

## models4 — Structured Argumentation Graph for Misinformation Detection

This project investigates whether misinformation can be detected by analyzing the relationships between an article's claims, evidence, and analysis rather than relying solely on the article's text.

The current system converts news articles into **Structured Argumentation Graphs**. Sentences are classified by rhetorical role, relevant external articles are retrieved, and relationships between claims and evidence are represented as graph edges. A Graph Attention Network then uses these relationships to classify articles as real or fake.

The project is actively under development, with current work focused on improving retrieval quality, model performance, and experimental evaluation.

---

## Motivation

Traditional text classifiers can perform well when misinformation contains distinctive linguistic or stylistic patterns. However, sophisticated misinformation can be written in a style that is difficult to distinguish from legitimate journalism.

This project explores a different approach:

> **Can relationships between claims, evidence, and analysis across independent sources provide useful signals for misinformation detection?**

The central hypothesis is that misinformation may be detectable at the **relational level**. An article may present plausible facts while drawing unsupported conclusions or interpretations that are contradicted by independent coverage of the same event.

To investigate this, the system:

1. Identifies the rhetorical role of sentences within an article.
2. Retrieves independent sources covering the same event.
3. Retrieves and processes the full text of relevant external articles.
4. Constructs a graph representing relationships between the target article and external sources.
5. Uses a Graph Attention Network to classify the resulting graph.

---

## Current System

### Pipeline

```text
                     Target Article
                           │
                           ▼
                  Sentence Tokenization
                           │
                           ▼
                  ┌───────────────────┐
                  │ Sentence Role     │
                  │ Classification    │
                  │                   │
                  │ claim             │
                  │ evidence          │
                  │ analysis          │
                  │ background        │
                  └─────────┬─────────┘
                            │
                            ▼
                    Headline Retrieval
                            │
                            ▼
                  External Articles
                            │
                            ▼
                 Full Article Retrieval
                            │
                            ▼
                 Document Scoring
              (consensus + credibility)
                            │
                            ▼
                 Cross-Source Analysis
                            │
                            ▼
                Structured Argumentation
                       Graph
                            │
                            ▼
                   Node Feature Matrix
                            │
                            ▼
                     GATv2Conv
                            │
                            ▼
                 Graph-Level Classification
                            │
                            ▼
                     Real / Fake
```

---

## Architecture

### 1. Sentence Role Classification

Each article is split into sentences and classified into one of four rhetorical roles using zero-shot Natural Language Inference (NLI):

| Role | Description |
|------|-------------|
| `claim` | A specific factual assertion |
| `evidence` | Data, quotations, or facts supporting a claim |
| `analysis` | Interpretation, opinion, or conclusion |
| `background` | General or contextual information |

The current implementation uses `cross-encoder/nli-deberta-v3-small`.

Separating these roles allows the graph to distinguish between what an article claims and how it interprets the underlying information.

---

### 2. External Article Retrieval

The system searches for external coverage using the target article's headline.

Headline-based retrieval was introduced to improve upon earlier entity-based searching, which frequently returned unrelated documents.

Retrieved documents are scored using:

- Semantic/document consensus
- Domain credibility
- Document relevance

Search results are cached locally to reduce repeated external requests.

---

### 3. Full-Article Retrieval

The current retrieval pipeline goes beyond search-result snippets by attempting to retrieve the **full text of external articles**.

This allows the system to construct richer evidence graphs and compare the target article against substantially more information from each external source.

Full-article retrieval is currently an active area of development, particularly around consistency and handling pages where the article body cannot be reliably extracted.

---

### 4. Source Credibility

A lightweight SQLite database maintains a Bayesian credibility estimate for individual domains.

Each domain begins with a neutral Beta prior and is updated using model predictions and human feedback.

The credibility estimate is incorporated into the ranking of retrieved documents, allowing evidence from sources with stronger accumulated reliability to receive greater weight.

---

### 5. Structured Argumentation Graph

Each article is represented as a graph containing:

**Input nodes**

- Sentences from the target article
- Each sentence is assigned a rhetorical role

**External nodes**

- Sentences extracted from retrieved articles
- Each sentence is also assigned a rhetorical role

**Edges**

- Intra-article semantic relationships
- Cross-source claim/evidence relationships
- Cross-source analysis relationships
- NLI-based entailment, contradiction, or neutral relationships

Cross-source comparisons are restricted to compatible rhetorical roles to reduce noise in the graph.

For example, analysis from the target article can be compared against analysis from independent sources to identify corroboration or contradiction.

---

### 6. Node Features

Each graph node currently contains a **20-dimensional hand-crafted feature vector**.

The features capture:

- Number and proportion of external neighbors
- Document relevance/credibility
- Semantic similarity
- Entailment and contradiction counts
- NLI confidence
- Sentence rhetorical role
- Cross-source corroboration
- Cross-source contradiction

The cross-source entailment and contradiction features are intended to capture whether an article's claims or analysis are supported or disputed by independent coverage.

---

### 7. GATv2 Graph Classifier

The current classifier uses **GATv2Conv** rather than the original `GATConv` implementation.

The model uses:

- 4 graph attention layers
- Multi-head attention
- Residual connections
- Jumping Knowledge aggregation
- Global mean pooling
- Global max pooling
- Multi-layer MLP classifier
- Binary cross-entropy loss

The transition from `GATConv` to `GATv2Conv` allows the attention mechanism to be more dynamically dependent on the representations of the nodes being compared.

---

## Experimental Results

The system has undergone several architectural iterations, with each version introducing changes to the graph construction, feature representation, and/or model architecture.

### Performance Across Model Versions

| Version | Accuracy | F1 Score | AUC-ROC | Early Stopping |
|---------|----------|----------|---------|----------------|
| `models1` | 0.467 | 0.636 | 0.500 | Epoch 21 |
| `models2` | 0.568 | 0.610 | 0.531 | Epoch 23 |
| `models3` | 0.743 | 0.716 | **0.859** | Epoch 28 |
| `models4` | **0.784** | **0.778** | 0.841 | Epoch 21 |

### models1

The initial implementation achieved limited performance on the held-out test set:

```text
Epoch 10: train=0.7343  val=0.7130  lr=4.95e-04
Epoch 20: train=0.7139  val=0.6946  lr=4.78e-04
Early stopping at epoch 21

Test Results:
Accuracy: 0.467
F1 Score: 0.636
AUC-ROC: 0.500
```

The AUC-ROC of 0.500 indicates that the initial model showed essentially no discriminative ability between the two classes according to this evaluation.

---

### models2

The second iteration improved classification accuracy:

```text
Epoch 10: train=0.7090  val=0.6893  lr=4.95e-04
Epoch 20: train=0.7044  val=0.7421  lr=4.78e-04
Early stopping at epoch 23

Test Results:
Accuracy: 0.568
F1 Score: 0.610
AUC-ROC: 0.531
```

Compared with `models1`, accuracy increased from 0.467 to 0.568 and AUC-ROC increased from 0.500 to 0.531, indicating an improvement in the model's ability to separate the two classes.

---

### models3

The third iteration produced a substantially larger improvement:

```text
Epoch 10: train=0.5339  val=0.6192  lr=4.95e-04
Epoch 20: train=0.5132  val=0.6269  lr=4.78e-04
Early stopping at epoch 28

Test Results:
Accuracy: 0.743
F1 Score: 0.716
AUC-ROC: 0.859
```

This represents a significant improvement over the first two iterations, particularly in AUC-ROC, which increased from 0.531 in `models2` to 0.859.

---

### models4

The current `models4` implementation further improved accuracy and F1 score:

```text
Test Results:
Accuracy: 0.784
F1 Score: 0.778
AUC-ROC: 0.841
```

`models4` should not be interpreted as a direct improvement over `models3` across every metric. While accuracy increased from 0.743 to 0.784 and F1 increased from 0.716 to 0.778, AUC-ROC decreased slightly from 0.859 to 0.841.

The results therefore represent an ongoing architectural development process rather than an unambiguous improvement across every evaluation metric.

---

### Overall Progression

The progression from `models1` through `models4` demonstrates substantial improvement over the initial architecture:

| Metric | models1 | models4 | Change |
|--------|---------|---------|--------|
| Accuracy | 0.467 | 0.784 | +0.317 |
| F1 Score | 0.636 | 0.778 | +0.142 |
| AUC-ROC | 0.500 | 0.841 | +0.341 |

The largest improvement occurred between `models2` and `models3`, suggesting that the architectural changes introduced during that iteration had a substantial effect on the model's ability to distinguish between truthful and false articles.

However, these results should be interpreted as **development results rather than a final benchmark**. The architecture and retrieval pipeline remain under active development, and additional experiments and ablation studies are needed to determine which individual changes are responsible for the observed improvements.

---

## Training

The project currently supports both conventional and batched training.

### Batched Training

The current training pipeline incrementally:

1. Builds graphs for a batch of articles.
2. Retries failed evidence retrieval.
3. Trains or fine-tunes the GAT model.
4. Evaluates against a fixed global validation pool.
5. Updates the source credibility database.
6. Saves a model checkpoint.
7. Continues with the next batch.

Subsequent batches use **warm-start fine-tuning** rather than reinitializing the model.

This allows the system to begin training before the entire dataset has been processed and allows the source credibility database to accumulate information throughout training.

---

## Current Research Questions

The project is currently focused on several questions.

### Does full-article retrieval improve classification?

The original system relied heavily on search-result snippets. The current version retrieves full article text, but an explicit experiment is still needed to determine whether the additional information actually improves model performance.

Planned comparisons include:

- No external retrieval
- Search-result snippets
- Full retrieved articles

with evaluation under consistent dataset splits and metrics.

### How reliable is external retrieval?

Search engines can return different results depending on query formulation and time. Full-article extraction can also fail for certain websites.

Improving retrieval consistency is therefore an important part of the current work.

### Which graph signals are actually useful?

The system contains several manually engineered features describing semantic similarity, NLI relationships, source credibility, and cross-source corroboration.

A major goal is to determine which of these signals provide meaningful predictive value rather than assuming that additional features necessarily improve the model.

### Why do different evaluation metrics change differently?

The `models3` and `models4` results provide an additional research question.

`models4` improved accuracy and F1 while producing a slightly lower AUC-ROC than `models3`.

Further experimentation is needed to determine whether this difference results from changes in model calibration, classification thresholds, graph construction, retrieval behavior, or other architectural changes.

---

## Known Limitations

### Retrieval Reliability

External search results are not guaranteed to be deterministic. Cached results improve repeatability, but experiments using live search may still vary over time.

### Full-Article Extraction

Not every retrieved webpage can be reliably converted into clean article text. Pages may use different layouts, dynamic rendering, paywalls, or other structures that complicate extraction.

### Zero-Shot Role Classification

The sentence-role classifier is not trained specifically for this task. Errors in role classification can propagate into the graph structure.

### Hand-Crafted Features

The current node representation relies heavily on manually designed features. This provides interpretability but may limit the amount of semantic information available to the GNN.

### Dataset Limitations

The current experiments use the ISOT Fake News Dataset. Dataset-specific characteristics may not represent modern misinformation encountered on the web, so performance on this dataset should not be interpreted as equivalent to real-world misinformation detection.

### Experimental Validation

The project is still in an experimental stage. In particular, the effect of full-article retrieval and individual architectural changes has not yet been isolated through comprehensive ablation studies.

---

## Project Structure

The codebase has been refactored so that the implementation is separated into modular pipeline components.

```text
.
├── models4.ipynb
├── data/
│   ├── Fake.csv
│   └── True.csv
│
└── pipeline/
    ├── config.py
    ├── data_loading.py
    ├── sentence_roles.py
    ├── retrieval.py
    ├── credibility.py
    ├── graph_building.py
    ├── augmentation.py
    ├── features.py
    ├── model.py
    ├── dataset_builder.py
    └── training_loop.py
```

### Where to Make Changes

| Component | File |
|-----------|------|
| Model configuration / pretrained models | `pipeline/config.py` |
| Dataset loading | `pipeline/data_loading.py` |
| Sentence role classification | `pipeline/sentence_roles.py` |
| Search and full-article retrieval | `pipeline/retrieval.py` |
| Source credibility | `pipeline/credibility.py` |
| Base graph construction | `pipeline/graph_building.py` |
| Retrieval/graph augmentation | `pipeline/augmentation.py` |
| Node features | `pipeline/features.py` |
| GAT architecture and training | `pipeline/model.py` |
| Dataset construction and evidence retry | `pipeline/dataset_builder.py` |
| Batched training | `pipeline/training_loop.py` |

`models4.ipynb` contains the high-level walkthrough, training workflow, and experimental results. Most reusable pipeline logic is implemented in the `pipeline/` package.

---

## Requirements

The project currently uses:

- Python
- PyTorch
- PyTorch Geometric
- Sentence Transformers
- Transformers
- NLTK
- spaCy
- NetworkX
- scikit-learn
- pandas
- NumPy
- matplotlib
- tqdm

Install the required spaCy model with:

```bash
python -m spacy download en_core_web_sm
```

The ISOT dataset should be placed in:

```text
data/
├── Fake.csv
└── True.csv
```

---

## Running the Project

The primary documentation and experimental workflow are contained in:

```text
models4.ipynb
```

The notebook imports the modular pipeline and can be used to construct graphs, train the model, evaluate performance, and inspect the resulting experiments.

---

## Future Work

Potential directions for future experimentation include:

- More reliable and reproducible retrieval
- Improved full-article extraction
- Ablation studies for retrieval and graph features
- Edge features incorporated directly into GATv2 attention
- Stronger sentence encoders
- Improved feature normalization
- Heterogeneous graph architectures
- Learned rather than entirely hand-crafted node representations
- Larger and more diverse news datasets
- Improved evaluation against contemporary misinformation
- Calibration and uncertainty estimation
- Human-facing explanations of the evidence used by the classifier

---

## Status

**Active research project**

The system is functional but remains experimental. Current development is focused on improving retrieval consistency, evaluating the effect of full-article retrieval, and determining which architectural components meaningfully improve model performance.
