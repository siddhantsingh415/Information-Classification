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
