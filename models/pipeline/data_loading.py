import json
import pandas as pd
from typing import Tuple

def load_averitec_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads AVeriTeC JSON format. Extracts claims and maps strict veracity labels
    to maintain parity with the binary GAT classification.
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

        records.append({
            'text': item.get('claim', ''),
            'speaker': item.get('speaker', ''),
            'label': binary_label
        })

    return pd.DataFrame(records)

def get_train_test_split(df: pd.DataFrame, train_frac: float = 0.8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.sample(frac=train_frac, random_state=random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df