# Information-Classification

A self-directed research project exploring graph-based misinformation detection using
natural language processing, information retrieval, and Graph Neural Networks.

## models4 — Structured Argumentation Graph for Misinformation Detection

This project investigates whether misinformation can be detected by analyzing the
relationships between an article's claims, evidence, and analysis rather than relying
solely on the article's text.

The current system converts news articles into **Structured Argumentation Graphs**.
Sentences are classified by rhetorical role, relevant external articles are retrieved,
and relationships between claims and evidence are represented as graph edges. A
Graph Attention Network then uses these relationships to classify articles as real
or fake.

The project is actively under development, with current work focused on improving
retrieval quality, model performance, and experimental evaluation.

---

## Motivation

Traditional text classifiers can perform well when misinformation contains
distinctive linguistic or stylistic patterns. However, sophisticated misinformation
can be written in a style that is difficult to distinguish from legitimate journalism.

This project explores a different approach:

> **Can relationships between claims, evidence, and analysis across independent
> sources provide useful signals for misinformation detection?**

The central hypothesis is that misinformation may be detectable at the
**relational level**. An article may present plausible facts while drawing
unsupported conclusions or interpretations that are contradicted by independent
coverage of the same event.

To investigate this, the system:

1. Identifies the rhetorical role of sentences within an article.
2. Retrieves independent sources covering the same event.
3. Retrieves and processes the full text of relevant external articles.
4. Constructs a graph representing relationships between the target article and
   external sources.
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


## Experimental Results

The system has undergone several architectural iterations, with each version
introducing changes to the graph construction, feature representation, and/or
model architecture.

### Performance Across Model Versions

| Version | Accuracy | F1 Score | AUC-ROC | Early Stopping |
|---------|----------|----------|---------|----------------|
| `models1` | 0.467 | 0.636 | 0.500 | Epoch 21 |
| `models2` | 0.568 | 0.610 | 0.531 | Epoch 23 |
| `models3` | 0.743 | 0.716 | **0.859** | Epoch 28 |
| `models4` | 0.784 | 0.778 | 0.841 | Epoch 21 |

models4 should not be interpreted as a direct improvement over models3
across every metric. While accuracy increased from 0.743 to 0.784 and F1
increased from 0.716 to 0.778, AUC-ROC decreased slightly from 0.859 to 0.841.

The results therefore represent an ongoing architectural development process
rather than an unambiguous improvement across every evaluation metric.