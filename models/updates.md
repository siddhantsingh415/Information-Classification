# Project State and Architectural Summary

## 1. Project Overview & Objective

The project is a **Graph Attention Network (GAT) pipeline for Fake News / Claim Verification**. It evaluates claims/articles by combining internal textual analysis with live external cross-source evidence retrieved via DuckDuckGo, forming a graph that runs through an NLI model and a PyTorch Geometric (PyG) GAT.

In this session, the project transitioned from relying on the **ISOT Dataset** (full-length CSV news articles) to the **AVeriTeC Dataset** (JSON-formatted claims, speakers, and 4-way veracity labels), while establishing full generalization for **live web article verification**.

---

## 2. Directory & Module Architecture

The codebase follows a modular Python package structure under the `pipeline/` directory, imported by root-level Jupyter Notebooks (`models4.ipynb` and the newly created `models5.ipynb`):

```text
.
├── models4.ipynb             # Legacy notebook (ISOT dataset baseline & training)
├── models5.ipynb             # New notebook (AVeriTeC experiment & GAT pipeline)
├── evidence_cache.db         # SQLite cache for web-scraped evidence pages
├── data/
│   ├── Fake.csv              # Legacy ISOT data
│   ├── True.csv              # Legacy ISOT data
│   └── averitec_dev.json     # Active AVeriTeC dataset split
└── pipeline/
    ├── config.py             # Shared models (spaCy 'nlp', ENCODER, NLI_MODEL, constants)
    ├── data_loading.py       # AVeriTeC & split data loading functions
    ├── sentence_roles.py     # Sentence role classification (claim/evidence/analysis/background)
    ├── retrieval.py          # Unified DuckDuckGo search, URL extraction, SQLite caching
    ├── credibility.py        # Per-domain source credibility SQLite database
    ├── graph_building.py     # NetworkX base graph construction (target claims & evidence nodes)
    ├── augmentation.py       # Search result augmentation & NLI edge weight generation
    ├── features.py           # 20-dim node feature extraction and NetworkX-to-PyG transformation
    ├── model.py              # FakeNewsGAT definition & training/evaluation helper functions
    ├── dataset_builder.py    # PyG dataset compilation utilities
    └── training_loop.py      # Batched PyG streaming training loop
```

---

## 3. Key Discoveries & Architectural Pivots

### A. Dataset Transition: ISOT -> AVeriTeC
* **ISOT Structure**: Full-length multi-sentence articles stored in CSV format with binary (`0`/`1`) labels. Intra-article graphs had multiple sentences as nodes.
* **AVeriTeC Structure**: JSON entries containing single-sentence `claim` strings, metadata (`speaker`), and 4-way labels (`Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`).
* **Binary Alignment**: To maintain compatibility with the GAT binary classifier, AVeriTeC labels are mapped as follows:
  * `Supported` -> `1` (Real / True)
  * `Refuted` -> `0` (Fake / False)
  * Ambiguous labels (`Not Enough Evidence`, `Conflicting Evidence`) are filtered out during dataset loading.

### B. Live Web Article Generalization
* **Problem**: In live production (e.g., verifying a web page or URL), metadata like explicit `speaker` or `claim` JSON keys are missing.
* **Solution**: Updated `pipeline/retrieval.py` with an automated extraction cascade using the shared `pipeline.config.nlp` spaCy model and `trafilatura`:
  1. *URL Title Clean-up*: Strips publisher branding (e.g., `" - BBC News"`) to focus searches on story headlines.
  2. *NER Extraction*: Extracts prominent entities (`PERSON`, `ORG`, `GPE`) as implicit "speakers".
  3. *Sentence Fallback*: Selects the lead sentence as the candidate claim if no headline exists.

---

## 4. File-by-File Changes Made

### 1. `pipeline/data_loading.py`
* **Changes**: Replaced ISOT CSV loading logic with `load_averitec_dataset()`.
* **Functionality**: Loads `averitec_dev.json`, filters non-binary labels, and returns a DataFrame with `text` (claim), `speaker`, and `label`. Includes `get_train_test_split()`.

### 2. `pipeline/retrieval.py`
* **Changes**: Updated to utilize `from pipeline.config import nlp` and support both dataset-driven and live URL searches.
* **Functionality**:
  * `extract_live_entities_and_claims()`: Uses spaCy NER to extract actors/speakers and lead sentences from raw text.
  * `generate_search_query()`: Formulates optimal DuckDuckGo search queries depending on available metadata (Dataset query vs. Cleaned URL title vs. NER entity + claim sentence).
  * `search_claim_evidence()`: Executes DuckDuckGo text searches and caches full page body scrapes in `evidence_cache.db` via `trafilatura`.

### 3. `pipeline/graph_building.py`
* **Changes**: Refactored `build_article_graph()` and added `add_evidence_nodes()`.
* **Functionality**:
  * `build_article_graph()`: Constructs a single-root NetworkX node for the target claim (`is_evidence=0`, `role='claim'`).
  * `add_evidence_nodes()`: Dynamically attaches external web pages returned by `retrieval.py` as evidence nodes (`is_evidence=1`, `role='evidence'`) with domain credibility weighting prior to cross-source NLI edge creation in `augmentation.py`.

### 4. `models5.ipynb`
* **Changes**: Created as the complete replacement notebook for running experiments on AVeriTeC.
* **Structure**:
  * **Cell 1**: Downloads `averitec_dev.json` to `data/` and builds `train_df` / `test_df`.
  * **Cell 2**: Fits a TF-IDF + Logistic Regression baseline on internal text features only (`ngram_range=(1,2)`, `max_features=5000`).
  * **Cell 3**: Defines `build_claim_graph()` to chain `build_article_graph` -> `search_claim_evidence` -> `add_evidence_nodes` -> `augment_with_cross_source` -> `graph_to_pyg`.
  * **Cell 4**: Instantiates `FakeNewsGAT(in_channels=20, hidden_channels=64, out_channels=1)` and executes PyG `DataLoader` batched training loops.

---

## 5. Current State & Execution Notes

* **Pipeline Status**: All modules under `pipeline/` are fully compatible with AVeriTeC JSON data and arbitrary web URLs.
* **Data Flow**:
  `JSON Claim / URL -> Base Node -> DDG Retrieval -> Evidence Nodes -> NLI Scoring -> PyG Graph (20-dim) -> FakeNewsGAT`
* **Notebook Operational Requirement**: Cells in `models5.ipynb` must be executed sequentially from Cell 1 down. Restarting the Jupyter kernel clears in-memory DataFrames (`train_df`, `test_df`) and requires re-running Cell 1 to avoid `NameError` exceptions.