"""
Turning a dataframe of articles into a list of PyG graphs.

Two ways to do this:
  - build_dataset: one-shot — builds every graph in the dataframe before
    returning. Simple, but you wait for the whole dataset before training
    starts.
  - build_batch + retry_failed_evidence: batched — builds one slice at a
    time and separates "zero evidence found" failures out so they can be
    retried with progressively broader fallback queries, without blocking
    the rest of the batch. Used by training_loop.batched_train_loop.

Edit this file to change: what counts as a "failed" article, the fallback
query strategies used to retry evidence collection, or how a batch's
diagnostics are reported.
"""
from dataclasses import dataclass
import copy

from tqdm import tqdm

from .config import ROLE_NAMES, nlp
from .graph_building import build_article_graph
from .retrieval import collect_evidence_by_headline, clean_headline, ddg_search, enrich_with_full_text
from .augmentation import augment_with_cross_source
from .features import graph_to_pyg


def build_dataset(df, text_col='text', title_col='title',
                  label_col='label_binary', sample=None, use_nli=True,
                  fetch_full_text=True):
    if sample:
        df = df.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(sample // 2, random_state=42)
        ).reset_index(drop=True)

    pyg_data, all_scored = [], []
    n_aug, n_fall = 0, 0
    role_counts = {r: 0 for r in ROLE_NAMES}

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Building graphs'):
        text  = row[text_col]
        title = row.get(title_col, '')
        label = int(row[label_col])

        # Build base article graph with role-labeled nodes
        G, article_sents, article_roles = build_article_graph(text, use_nli=use_nli)
        if not article_sents:
            all_scored.append([])
            continue

        # Track role distribution for diagnostics
        for r in article_roles:
            role_counts[ROLE_NAMES[r]] += 1

        # Headline search + cross-source augmentation
        scored_docs = []
        try:
            docs = collect_evidence_by_headline(title, k=10, fetch_full_text=fetch_full_text)
            if docs:
                G, scored_docs = augment_with_cross_source(
                    G, article_sents, article_roles, docs, use_nli=use_nli
                )
                n_aug += 1
            else:
                n_fall += 1
        except Exception as e:
            print(f'  Augmentation failed: {e}')
            n_fall += 1

        pyg_data.append(graph_to_pyg(G, label=label))
        all_scored.append(scored_docs)

    total_roles = sum(role_counts.values())
    print(f'\nBuilt {len(pyg_data)} | Augmented: {n_aug} | Fallback: {n_fall}')
    print('Role distribution:', {k: f'{v/total_roles:.1%}' for k, v in role_counts.items()})
    return pyg_data, all_scored


@dataclass
class FailedItem:
    """Holds the pre-augmentation state of an article that returned zero evidence,
    so retry_failed_evidence can patch it without re-running role classification."""
    df_idx:        int
    title:         str
    text:          str
    label:         int
    G_base:        object     # nx.Graph before augmentation
    article_sents: list
    article_roles: list


def build_batch(
    batch_df,
    text_col:  str  = 'text',
    title_col: str  = 'title',
    label_col: str  = 'label_binary',
    use_nli:   bool = True,
    fetch_full_text: bool = True,
):
    """
    Build graphs for one batch of articles.

    fetch_full_text controls whether each retrieved doc's full article page is
    scraped (see retrieval.py) — set False to skip those network round-trips
    for a quicker run at the cost of thinner (snippet-only) evidence.

    Returns
    -------
    pyg_data     : list[Data]        — PyG graphs for augmented articles
    all_scored   : list[list]        — scored doc lists aligned with pyg_data
    failed_items : list[FailedItem]  — articles with zero evidence (can be retried)
    role_counts  : dict              — role distribution diagnostics
    """
    pyg_data, all_scored, failed_items = [], [], []
    n_aug = 0
    role_counts = {r: 0 for r in ROLE_NAMES}

    for df_idx, row in tqdm(batch_df.iterrows(), total=len(batch_df),
                            desc='Building batch', leave=False):
        text  = row[text_col]
        title = row.get(title_col, '')
        label = int(row[label_col])

        G_base, article_sents, article_roles = build_article_graph(text, use_nli=use_nli)
        if not article_sents:
            continue  # empty article — skip entirely

        for r in article_roles:
            role_counts[ROLE_NAMES[r]] += 1

        scored_docs  = []
        got_evidence = False
        try:
            docs = collect_evidence_by_headline(title, k=10, fetch_full_text=fetch_full_text)
            if docs:
                G_aug, scored_docs = augment_with_cross_source(
                    G_base, article_sents, article_roles, docs, use_nli=use_nli
                )
                pyg_data.append(graph_to_pyg(G_aug, label=label))
                all_scored.append(scored_docs)
                n_aug += 1
                got_evidence = True
        except Exception as e:
            print(f'  [build_batch] augmentation error idx={df_idx}: {e}')

        if not got_evidence:
            # Preserve pre-augmentation state for retry
            failed_items.append(FailedItem(
                df_idx=df_idx, title=title, text=text, label=label,
                G_base=copy.deepcopy(G_base),
                article_sents=article_sents,
                article_roles=article_roles,
            ))

    total_roles = max(sum(role_counts.values()), 1)
    print(f'  Built {len(pyg_data)} | Augmented: {n_aug} | Failed: {len(failed_items)}')
    print('  Role dist:', {k: f'{v/total_roles:.1%}' for k, v in role_counts.items()})
    return pyg_data, all_scored, failed_items, role_counts


def _fallback_queries(item: FailedItem) -> list:
    """
    Return up to three progressively broader fallback queries for a failed article,
    ordered from most specific to most generic.
    """
    queries = []

    # Strategy 1: truncate headline to first 6 words
    words = clean_headline(item.title).split()
    if len(words) > 3:
        queries.append(' '.join(words[:6]))

    # Strategy 2: spaCy NER — named entities + top noun chunks from title + lead text
    doc  = nlp(item.title + '. ' + item.text[:300])
    ents = [e.text for e in doc.ents
            if e.label_ in ('PERSON', 'ORG', 'GPE', 'EVENT', 'NORP', 'FAC', 'LOC')]
    chunks = [c.root.text for c in doc.noun_chunks if len(c.root.text) > 3]
    kw = list(dict.fromkeys(ents + chunks))[:5]   # deduplicate, preserve order
    if kw:
        queries.append(' '.join(kw))

    # Strategy 3: first substantive sentence of the article body
    sents = [s.strip() for s in item.text.split('.') if len(s.strip()) > 20]
    if sents:
        queries.append(sents[0][:120])

    return queries


def retry_failed_evidence(
    failed_items: list,
    use_nli: bool = True,
    fetch_full_text: bool = True,
):
    """
    Attempt to retrieve evidence for articles that returned zero results during
    build_batch, using up to three fallback query strategies per article.

    Articles that fail all strategies are included as base-graph-only (no cross-source
    edges) so no articles are silently dropped from the dataset.

    Returns
    -------
    new_pyg      : list[Data]        — graphs for all previously-failed articles
    new_scored   : list[list]        — aligned scored doc lists
    still_failed : list[FailedItem]  — articles that failed all strategies (base-only)
    """
    new_pyg, new_scored, still_failed = [], [], []

    for item in tqdm(failed_items, desc='Retrying evidence', leave=False):
        recovered = False
        for attempt, query in enumerate(_fallback_queries(item), start=1):
            try:
                docs = ddg_search(query, num_results=10)
                if not docs:
                    continue
                if fetch_full_text:
                    docs = enrich_with_full_text(docs)
                G_retry = copy.deepcopy(item.G_base)
                G_aug, scored_docs = augment_with_cross_source(
                    G_retry, item.article_sents, item.article_roles,
                    docs, use_nli=use_nli
                )
                if scored_docs:   # evidence nodes were actually added
                    new_pyg.append(graph_to_pyg(G_aug, label=item.label))
                    new_scored.append(scored_docs)
                    print(f'  [retry] idx={item.df_idx} recovered via '
                          f'strategy {attempt} ("{query[:50]}")')
                    recovered = True
                    break
            except Exception as e:
                print(f'  [retry] strategy {attempt} failed for idx={item.df_idx}: {e}')

        if not recovered:
            # Include base-graph (no cross-source edges) so article stays in dataset
            new_pyg.append(graph_to_pyg(item.G_base, label=item.label))
            new_scored.append([])
            still_failed.append(item)

    n_recovered = len(failed_items) - len(still_failed)
    print(f'[retry] Recovered: {n_recovered} | Still failed (base-only): {len(still_failed)}')
    return new_pyg, new_scored, still_failed
