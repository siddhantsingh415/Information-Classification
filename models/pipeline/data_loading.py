import json
import pandas as pd
from typing import Tuple

def load_averitec_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads AVeriTeC JSON format. Extracts claims and maps strict veracity labels
    to maintain parity with the binary GAT classification.

    Also extracts `source_url` -- the claim's own source/context article link
    (AVeriTeC's `cached_original_claim_url`, falling back to
    `original_claim_url` if no archived copy exists), used by
    `pipeline.retrieval.search_claim_evidence` to fetch the real article and
    search by its headline rather than the bare claim sentence. Only ~52% of
    AVeriTeC's dev split actually has one populated -- rows without it fall
    back to claim-text-based search in retrieval.py. The archived
    (`cached_...`) URL is preferred over the live one since it tends to
    survive longer than the original, which is the point of archiving it.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for item in data:
        label = item.get('label')

        # Drop ambiguous labels for strict True/False pipeline
        if label == "Supported":
            binary_label = 1
        elif label == "Refuted":
            binary_label = 0
        else:
            continue

        source_url = item.get('cached_original_claim_url') or item.get('original_claim_url') or ''

        records.append({
            'text': item.get('claim', ''),
            'speaker': item.get('speaker') or '',
            'label': binary_label,
            'source_url': source_url,
        })

    return pd.DataFrame(records)

def get_train_test_split(df: pd.DataFrame, train_frac: float = 0.8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.sample(frac=train_frac, random_state=random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df
