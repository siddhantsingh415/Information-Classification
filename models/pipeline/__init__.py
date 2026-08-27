"""
fake-news detection pipeline

Each stage of the models4 pipeline lives in its own module here so it can be
edited independently without touching the training notebook:

    config.py            shared constants + the pretrained models everything else uses
    data_loading.py      ISOT dataset loading/preprocessing
    sentence_roles.py    zero-shot claim/evidence/analysis/background classification
    retrieval.py         DuckDuckGo headline search + full-article-text fetching (with on-disk caching)
    credibility.py       per-domain source-credibility SQLite database
    graph_building.py    build the base per-article graph (sentences -> nodes/edges)
    augmentation.py      fold retrieved articles into the graph as cross-source evidence
    features.py          node feature engineering + networkx -> PyG conversion
    model.py             the FakeNewsGAT model + train/finetune/evaluate/save/load
    dataset_builder.py   df -> list[Data]: build_dataset (one-shot) and
                          build_batch / retry_failed_evidence (batched, streaming)
    training_loop.py     batched_train_loop: interleaves dataset_builder + model training

See the notebook's "Project layout" cell for a short guide to what to edit where.
"""
